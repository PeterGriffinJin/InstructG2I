# This code is for loading the trained graph adapter model for pipeline inference.
import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import UNet2DConditionModel
from torchvision import transforms

from .utils import get_generator

class PadToSquare:
        def __init__(self, fill=0, padding_mode='constant'):
            """
            Initializes the transform.
            :param fill: Pixel fill value for padding. Default is 0 (black).
            :param padding_mode: Type of padding. Can be 'constant', 'edge', etc.
            """
            self.fill = fill
            self.padding_mode = padding_mode

        def __call__(self, img):
            """
            Applies the transform to the given image.
            :param img: PIL Image or torch.Tensor to be padded.
            :return: Padded Image.
            """
            # Convert to PIL Image if it's a tensor
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)

            # Calculate padding
            width, height = img.size
            max_side = max(width, height)
            padding_left = (max_side - width) // 2
            padding_right = max_side - width - padding_left
            padding_top = (max_side - height) // 2
            padding_bottom = max_side - height - padding_top

            # Apply padding
            padding = (padding_left, padding_top, padding_right, padding_bottom)
            return transforms.Pad(padding, fill=self.fill, padding_mode=self.padding_mode)(img)


class GraphAdapter(torch.nn.Module):
    """Graph-Adapter"""
    def __init__(self, unet, image_proj_model, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model

        if ckpt_path is not None:
            self.from_pretrained(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, image_mask):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def save_pretrained(self, output_dir: str, save_unet=True):
        if save_unet:
            self.unet.save_pretrained(os.path.join(output_dir, "unet"))
        img_proj_save_dir = os.path.join(output_dir, "image_proj_model")
        if not os.path.exists(img_proj_save_dir):
            os.mkdir(img_proj_save_dir)
        torch.save(self.image_proj_model.state_dict(), os.path.join(img_proj_save_dir, 'ckpt.pt'))

    def from_pretrained(self, ckpt_path: str, subfolder=None):
        if subfolder:
            ckpt_path = os.path.join(ckpt_path, subfolder)
        
        # Load unet checkpoints
        load_model = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
        self.unet.register_to_config(**load_model.config)
        self.unet.load_state_dict(load_model.state_dict())
        del load_model

        # load image feature projector
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        state_dict = torch.load(os.path.join(ckpt_path, 'image_proj_model', 'ckpt.pt'), map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict, strict=True)
        del state_dict

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        bz, neighbor_num, _ = image_embeds.shape
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens).view(bz, neighbor_num * self.clip_extra_context_tokens, self.cross_attention_dim)
        return clip_extra_context_tokens


class GraphAdapterPipeline:
    def __init__(self, sd_pipe, image_encoder_path, proj_ckpt, neighbor_num, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.proj_ckpt = proj_ckpt
        self.num_tokens = num_tokens
        self.neighbor_num = neighbor_num

        self.pipe = sd_pipe.to(self.device)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        
        # image proj model
        self.image_proj_model = self.init_proj()
        self.load_proj()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_proj(self):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        state_dict = torch.load(self.proj_ckpt, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict, strict=True)
        del state_dict

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"

        print(f"Successfully loaded weights from checkpoint {self.proj_ckpt}")

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
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=50,
        **kwargs,
    ):

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            neighbor_image=neighbor_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

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
