import torch
import random
import numpy as np
from PIL import Image


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    resized_img = img.resize((max_edge_resolution, max_edge_resolution))
    return resized_img


def pyramid_noise_like(noise, device, iterations=6, discount=0.3):
    b, c, w, h = noise.shape
    u = torch.nn.Upsample(size=(w, h), mode='bilinear').to(device)
    for i in range(iterations):
        r = random.random()*2+2 # Rather than always going 2x,
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(device)) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance
