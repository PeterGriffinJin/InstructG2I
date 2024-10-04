import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

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

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    # w, h = imgs[0].size
    w, h = max([img.size[0] for img in imgs]), max([img.size[1] for img in imgs])
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def image_grid2(imgs, rows, cols, resolution):
    assert len(imgs) == rows*cols

    # w, h = imgs[0].size
    # w, h = max([img.size[0] for img in imgs]), max([img.size[1] for img in imgs])
    w, h = resolution, resolution
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    transform = transforms.Compose(
        [
            PadToSquare(fill=(resolution, resolution, resolution), padding_mode='constant'),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution)
        ]
    )
    
    for i, img in enumerate(imgs):
        grid.paste(transform(img), box=(i%cols*w, i//cols*h))
    return grid

def get_neighbor_transforms(resolution):
    return  transforms.Compose(
            [
                PadToSquare(fill=(resolution, resolution, resolution), padding_mode='constant'),
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution)
            ]
    )