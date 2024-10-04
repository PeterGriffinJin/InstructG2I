from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from transformers.models.clip.configuration_clip import CLIPConfig

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from typing import List, Optional, Union

import os
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from .GraphAdapter import ImageProjModel
from .GraphQFormer import CLIPGraphTextModel
from .utils import get_generator

logger = logging.get_logger(__name__)

def _encode_prompt(
    pipeline,
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
    ):

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(pipeline, LoraLoaderMixin):
        pipeline._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if not USE_PEFT_BACKEND:
            adjust_lora_scale_text_encoder(pipeline.text_encoder, lora_scale)
        else:
            scale_lora_layers(pipeline.text_encoder, lora_scale)

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # textual inversion: process multi-vector tokens if necessary
        if isinstance(pipeline, TextualInversionLoaderMixin):
            prompt = pipeline.maybe_convert_prompt(prompt, pipeline.tokenizer)

        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = pipeline.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = pipeline.tokenizer.batch_decode(
                untruncated_ids[:, pipeline.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {pipeline.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = pipeline.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = pipeline.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = pipeline.text_encoder.text_model.final_layer_norm(prompt_embeds)

    if pipeline.text_encoder is not None:
        prompt_embeds_dtype = pipeline.text_encoder.dtype
    elif pipeline.unet is not None:
        prompt_embeds_dtype = pipeline.unet.dtype
    else:
        prompt_embeds_dtype = prompt_embeds.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        # textual inversion: process multi-vector tokens if necessary
        if isinstance(pipeline, TextualInversionLoaderMixin):
            uncond_tokens = pipeline.maybe_convert_prompt(uncond_tokens, pipeline.tokenizer)

        max_length = prompt_embeds.shape[1]
        uncond_input = pipeline.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        negative_prompt_embeds = pipeline.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    if isinstance(pipeline, LoraLoaderMixin) and USE_PEFT_BACKEND:
        # Retrieve the original scale by scaling back the LoRA layers
        unscale_lora_layers(pipeline.text_encoder, lora_scale)

    return prompt_embeds, negative_prompt_embeds, text_input_ids, uncond_input.input_ids


class InstructG2IPipeline:
    def __init__(self, sd_pipe, sd_model_dir, image_encoder_path, proj_ckpt, gnn_ckpt, num_tokens, cross_attention_freq, neighbor_num, device):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.proj_ckpt = proj_ckpt
        self.num_tokens = num_tokens
        self.cross_attention_freq = cross_attention_freq
        self.gnn_ckpt = gnn_ckpt
        self.sd_model_dir = sd_model_dir
        self.neighbor_num = neighbor_num

        self.pipe = sd_pipe.to(self.device)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        
        # image proj model
        self.image_proj_model, self.gnn = self.init_proj_and_gnn()
        self.load_proj_and_gnn()
        print(f'Successfully loaded weights from checkpoint {sd_model_dir}')

    @classmethod
    def from_pretrained(cls, sd_model_dir, neighbor_num, cache_dir='cache/', device='cpu'):
        if not os.path.isdir(sd_model_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=sd_model_dir, local_dir=os.path.join(cache_dir, sd_model_dir))
        sd_model_dir = os.path.join(cache_dir, sd_model_dir)
        
        from diffusers import StableDiffusionPipeline
        sd_init = StableDiffusionPipeline.from_pretrained(sd_model_dir, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False).to(device)
        proj_ckpt = torch.load(os.path.join(sd_model_dir, "image_proj_model", "ckpt.pt"), map_location="cpu")
        gnn_ckpt = torch.load(os.path.join(sd_model_dir, "gnn", "ckpt.pt"), map_location="cpu")
        num_tokens = int(proj_ckpt['proj.weight'].shape[0] / proj_ckpt['norm.weight'].shape[0])
        cross_attention_freq = int(sum(1 for s in gnn_ckpt if "self_attn.q_proj.weight" in s) / sum(1 for s in gnn_ckpt if "crossattention.q_proj.weight" in s))
        
        return cls(sd_init,
                    sd_model_dir,
                    os.path.join(sd_model_dir, 'image_encoder'),
                    os.path.join(sd_model_dir, "image_proj_model", "ckpt.pt"), 
                    os.path.join(sd_model_dir, "gnn", "ckpt.pt"),
                    num_tokens,
                    cross_attention_freq,
                    neighbor_num,
                    device
                    )
        
    def init_proj_and_gnn(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        
        # create the gnn model
        encoder_config = CLIPConfig.from_pretrained(
            self.sd_model_dir, subfolder="text_encoder", cache_dir="/shared/data/bowenj4/hf-cache"
        )
        encoder_config.encoder_width = self.pipe.unet.config.cross_attention_dim
        encoder_config.cross_attention_freq = self.cross_attention_freq
        Qformer = CLIPGraphTextModel.from_pretrained(
            self.sd_model_dir, subfolder="text_encoder", config=encoder_config, cache_dir="/shared/data/bowenj4/hf-cache"
        ).to(self.device, dtype=torch.float16)
        
        return image_proj_model, Qformer

    def load_proj_and_gnn(self):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_gnn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.gnn.parameters()]))
        
        state_dict = torch.load(self.proj_ckpt, map_location="cpu")
        gnn_state_dict = torch.load(self.gnn_ckpt, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict, strict=True)
        self.gnn.load_state_dict(gnn_state_dict, strict=True)
        del state_dict
        del gnn_state_dict

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_gnn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.gnn.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_gnn_sum != new_gnn_sum, "Weights of gnn did not change!"

        print(f"Successfully loaded weights from checkpoint {self.proj_ckpt} and {self.gnn_ckpt}")

    @torch.inference_mode()
    def get_image_embeds(self, neighbor_image=None, clip_image_embeds=None):
        reshape_flag = False
        if neighbor_image is not None:
            if isinstance(neighbor_image, Image.Image):
                neighbor_image = [neighbor_image]
            if isinstance(neighbor_image[0], List):
                assert len(neighbor_image[0]) == self.neighbor_num
                bz = len(neighbor_image)
                neighbor_image = sum(neighbor_image, [])
                reshape_flag = True
            clip_image = self.clip_image_processor(images=neighbor_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if len(clip_image_embeds.shape) == 2:
            clip_image_embeds = clip_image_embeds.unsqueeze(0)
            
        if reshape_flag:
            clip_image_embeds = clip_image_embeds.view(bz, self.neighbor_num, -1)
            
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))

        return image_prompt_embeds, uncond_image_prompt_embeds

    def __call__(
        self,
        neighbor_image=None,
        neighbor_mask=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=50,
        **kwargs,
    ):

        if neighbor_mask and not isinstance(neighbor_mask, torch.LongTensor):
            # Convert neighbor_mask to a LongTensor
            neighbor_mask = torch.LongTensor(neighbor_mask)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            neighbor_image=neighbor_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_, text_input_ids, negative_text_input_ids = _encode_prompt(
                self.pipe,
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
                        
            neighbor_mask = neighbor_mask.repeat(num_samples, 1)
            neighbor_mask = torch.repeat_interleave(neighbor_mask, self.num_tokens, dim=1)
            gnn_prompt_embeds = self.gnn(input_ids=text_input_ids.repeat(num_samples, 1).to(self.device), encoder_hidden_states=image_prompt_embeds, encoder_attention_mask=neighbor_mask.to(self.device)).last_hidden_state
            uncond_gnn_prompt_embeds = self.gnn(input_ids=negative_text_input_ids.repeat(num_samples, 1).to(self.device), encoder_hidden_states=uncond_image_prompt_embeds, encoder_attention_mask=neighbor_mask.to(self.device)).last_hidden_state

            prompt_embeds = torch.cat([prompt_embeds_, gnn_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_gnn_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images


class InstructG2IGuidePipeline:
    def __init__(self, sd_pipe, sd_model_dir, image_encoder_path, proj_ckpt, gnn_ckpt, num_tokens, cross_attention_freq, neighbor_num, device):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.proj_ckpt = proj_ckpt
        self.num_tokens = num_tokens
        self.cross_attention_freq = cross_attention_freq
        self.gnn_ckpt = gnn_ckpt
        self.sd_model_dir = sd_model_dir
        self.neighbor_num = neighbor_num

        self.pipe = sd_pipe.to(self.device)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        
        # image proj model
        self.image_proj_model, self.gnn = self.init_proj_and_gnn()
        self.load_proj_and_gnn()
        print(f'Successfully loaded weights from checkpoint {sd_model_dir}')

    @classmethod
    def from_pretrained(cls, sd_model_dir, neighbor_num, cache_dir='cache/', device='cpu'):
        if not os.path.isdir(sd_model_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=sd_model_dir, local_dir=os.path.join(cache_dir, sd_model_dir))
        sd_model_dir = os.path.join(cache_dir, sd_model_dir)
        
        from .customized_sd_pipeline import StableDiffusionPipeline
        sd_init = StableDiffusionPipeline.from_pretrained(sd_model_dir, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False).to(device)
        proj_ckpt = torch.load(os.path.join(sd_model_dir, "image_proj_model", "ckpt.pt"), map_location="cpu")
        gnn_ckpt = torch.load(os.path.join(sd_model_dir, "gnn", "ckpt.pt"), map_location="cpu")
        num_tokens = int(proj_ckpt['proj.weight'].shape[0] / proj_ckpt['norm.weight'].shape[0])
        cross_attention_freq = int(sum(1 for s in gnn_ckpt if "self_attn.q_proj.weight" in s) / sum(1 for s in gnn_ckpt if "crossattention.q_proj.weight" in s))
        
        return cls(sd_init,
                    sd_model_dir,
                    os.path.join(sd_model_dir, 'image_encoder'),
                    os.path.join(sd_model_dir, "image_proj_model", "ckpt.pt"), 
                    os.path.join(sd_model_dir, "gnn", "ckpt.pt"),
                    num_tokens,
                    cross_attention_freq,
                    neighbor_num,
                    device
                    )

    def init_proj_and_gnn(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        
        # create the gnn model
        encoder_config = CLIPConfig.from_pretrained(
            self.sd_model_dir, subfolder="text_encoder", cache_dir="/shared/data/bowenj4/hf-cache"
        )
        encoder_config.encoder_width = self.pipe.unet.config.cross_attention_dim
        encoder_config.cross_attention_freq = self.cross_attention_freq
        Qformer = CLIPGraphTextModel.from_pretrained(
            self.sd_model_dir, subfolder="text_encoder", config=encoder_config, cache_dir="/shared/data/bowenj4/hf-cache"
        ).to(self.device, dtype=torch.float16)
        
        return image_proj_model, Qformer

    def load_proj_and_gnn(self):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_gnn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.gnn.parameters()]))
        
        state_dict = torch.load(self.proj_ckpt, map_location="cpu")
        gnn_state_dict = torch.load(self.gnn_ckpt, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict, strict=True)
        self.gnn.load_state_dict(gnn_state_dict, strict=True)
        del state_dict
        del gnn_state_dict

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_gnn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.gnn.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_gnn_sum != new_gnn_sum, "Weights of gnn did not change!"

        print(f"Successfully loaded weights from checkpoint {self.proj_ckpt} and {self.gnn_ckpt}")

    @torch.inference_mode()
    def get_image_embeds(self, neighbor_image=None, clip_image_embeds=None):
        reshape_flag = False
        if neighbor_image is not None:
            if isinstance(neighbor_image, Image.Image):
                neighbor_image = [neighbor_image]
            if isinstance(neighbor_image[0], List):
                assert len(neighbor_image[0]) == self.neighbor_num
                bz = len(neighbor_image)
                neighbor_image = sum(neighbor_image, [])
                reshape_flag = True
            clip_image = self.clip_image_processor(images=neighbor_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if len(clip_image_embeds.shape) == 2:
            clip_image_embeds = clip_image_embeds.unsqueeze(0)
            
        if reshape_flag:
            clip_image_embeds = clip_image_embeds.view(bz, self.neighbor_num, -1)
            
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)

        return image_prompt_embeds

    def __call__(
        self,
        neighbor_image=None,
        neighbor_mask=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        graph_guidance_scale=1.5,
        num_inference_steps=50,
        **kwargs,
    ):

        if neighbor_mask and not isinstance(neighbor_mask, torch.LongTensor):
            # Convert neighbor_mask to a LongTensor
            neighbor_mask = torch.LongTensor(neighbor_mask)

        image_prompt_embeds = self.get_image_embeds(
            neighbor_image=neighbor_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_, text_input_ids, negative_text_input_ids = _encode_prompt(
                self.pipe,
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
                        
            neighbor_mask = neighbor_mask.repeat(num_samples, 1)
            neighbor_mask = torch.repeat_interleave(neighbor_mask, self.num_tokens, dim=1)
            gnn_prompt_embeds = self.gnn(input_ids=text_input_ids.repeat(num_samples, 1).to(self.device), encoder_hidden_states=image_prompt_embeds, encoder_attention_mask=neighbor_mask.to(self.device)).last_hidden_state
            uncond_gnn_prompt_embeds = torch.zeros_like(gnn_prompt_embeds)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds_,
            negative_prompt_embeds=negative_prompt_embeds_,
            graph_prompt_embeds=gnn_prompt_embeds,
            negative_graph_prompt_embeds=uncond_gnn_prompt_embeds,
            guidance_scale=guidance_scale,
            graph_guidance_scale=graph_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images


class InstructG2IMultiGuidePipeline:
    def __init__(self, sd_pipe, sd_model_dir, image_encoder_path, proj_ckpt, gnn_ckpt, num_tokens, cross_attention_freq, neighbor_num, device):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.proj_ckpt = proj_ckpt
        self.num_tokens = num_tokens
        self.cross_attention_freq = cross_attention_freq
        self.gnn_ckpt = gnn_ckpt
        self.sd_model_dir = sd_model_dir
        self.neighbor_num = neighbor_num

        self.pipe = sd_pipe.to(self.device)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        
        # image proj model
        self.image_proj_model, self.gnn = self.init_proj_and_gnn()
        self.load_proj_and_gnn()
        print(f'Successfully loaded weights from checkpoint {sd_model_dir}')
        
    @classmethod
    def from_pretrained(cls, sd_model_dir, neighbor_num, cache_dir='cache/', device='cpu'):
        if not os.path.isdir(sd_model_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=sd_model_dir, local_dir=os.path.join(cache_dir, sd_model_dir))
        sd_model_dir = os.path.join(cache_dir, sd_model_dir)
        
        from .customized_sd_pipeline_multi import StableDiffusionPipeline
        sd_init = StableDiffusionPipeline.from_pretrained(sd_model_dir, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False).to(device)
        proj_ckpt = torch.load(os.path.join(sd_model_dir, "image_proj_model", "ckpt.pt"), map_location="cpu")
        gnn_ckpt = torch.load(os.path.join(sd_model_dir, "gnn", "ckpt.pt"), map_location="cpu")
        num_tokens = int(proj_ckpt['proj.weight'].shape[0] / proj_ckpt['norm.weight'].shape[0])
        cross_attention_freq = int(sum(1 for s in gnn_ckpt if "self_attn.q_proj.weight" in s) / sum(1 for s in gnn_ckpt if "crossattention.q_proj.weight" in s))
        
        return cls(sd_init,
                    sd_model_dir,
                    os.path.join(sd_model_dir, 'image_encoder'),
                    os.path.join(sd_model_dir, "image_proj_model", "ckpt.pt"), 
                    os.path.join(sd_model_dir, "gnn", "ckpt.pt"),
                    num_tokens,
                    cross_attention_freq,
                    neighbor_num,
                    device
                    )
        
    def init_proj_and_gnn(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        
        # create the gnn model
        encoder_config = CLIPConfig.from_pretrained(
            self.sd_model_dir, subfolder="text_encoder", cache_dir="/shared/data/bowenj4/hf-cache"
        )
        encoder_config.encoder_width = self.pipe.unet.config.cross_attention_dim
        encoder_config.cross_attention_freq = self.cross_attention_freq
        Qformer = CLIPGraphTextModel.from_pretrained(
            self.sd_model_dir, subfolder="text_encoder", config=encoder_config, cache_dir="/shared/data/bowenj4/hf-cache"
        ).to(self.device, dtype=torch.float16)
        
        return image_proj_model, Qformer

    def load_proj_and_gnn(self):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_gnn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.gnn.parameters()]))
        
        state_dict = torch.load(self.proj_ckpt, map_location="cpu")
        gnn_state_dict = torch.load(self.gnn_ckpt, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict, strict=True)
        self.gnn.load_state_dict(gnn_state_dict, strict=True)
        del state_dict
        del gnn_state_dict

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_gnn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.gnn.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_gnn_sum != new_gnn_sum, "Weights of gnn did not change!"

        print(f"Successfully loaded weights from checkpoint {self.proj_ckpt} and {self.gnn_ckpt}")

    @torch.inference_mode()
    def get_image_embeds(self, neighbor_image=None, clip_image_embeds=None):
        reshape_flag = False
        if neighbor_image is not None:
            if isinstance(neighbor_image, Image.Image):
                neighbor_image = [neighbor_image]
            if isinstance(neighbor_image[0], List):
                assert len(neighbor_image[0]) == self.neighbor_num
                bz = len(neighbor_image)
                neighbor_image = sum(neighbor_image, [])
                reshape_flag = True
            clip_image = self.clip_image_processor(images=neighbor_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if len(clip_image_embeds.shape) == 2:
            clip_image_embeds = clip_image_embeds.unsqueeze(0)
            
        if reshape_flag:
            clip_image_embeds = clip_image_embeds.view(bz, self.neighbor_num, -1)
            
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)

        return image_prompt_embeds

    def __call__(
        self,
        neighbor_images=None,
        neighbor_masks=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        graph_guidance_scale=1.5,
        num_inference_steps=50,
        **kwargs,
    ):

        if neighbor_masks and not isinstance(neighbor_masks[0], torch.LongTensor):
            # Convert neighbor_mask to a LongTensor
            # neighbor_masks = torch.LongTensor(neighbor_masks)
            neighbor_masks = [torch.LongTensor(neighbor_mask) for neighbor_mask in neighbor_masks]

        all_image_prompt_embeds = []
        for neighbor_image in neighbor_images:
            image_prompt_embeds = self.get_image_embeds(
                neighbor_image=neighbor_image, clip_image_embeds=clip_image_embeds
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            all_image_prompt_embeds.append(image_prompt_embeds)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_, text_input_ids, negative_text_input_ids = _encode_prompt(
                self.pipe,
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
                        
            gnn_prompt_embeds = []
            for neighbor_mask, image_prompt_embeds in zip(neighbor_masks, all_image_prompt_embeds):
                neighbor_mask = neighbor_mask.repeat(num_samples, 1)
                neighbor_mask = torch.repeat_interleave(neighbor_mask, self.num_tokens, dim=1)
                gnn_prompt_embed = self.gnn(input_ids=text_input_ids.repeat(num_samples, 1).to(self.device), 
                                            encoder_hidden_states=image_prompt_embeds, 
                                            encoder_attention_mask=neighbor_mask.to(self.device)).last_hidden_state
                gnn_prompt_embeds.append(gnn_prompt_embed)
            uncond_gnn_prompt_embeds = torch.zeros_like(gnn_prompt_embed)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds_,
            negative_prompt_embeds=negative_prompt_embeds_,
            graph_prompt_embeds=gnn_prompt_embeds,
            negative_graph_prompt_embeds=uncond_gnn_prompt_embeds,
            guidance_scale=guidance_scale,
            graph_guidance_scale=graph_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )

        return images
