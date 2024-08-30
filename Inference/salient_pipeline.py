from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import BaseOutput

from utils.image_util import resize_max_res
from utils.ensemble import ensemble_masks
from lib.SalientBranch import SalientBranch_Res2Net101


class SalientPipelineOutput(BaseOutput):
    """
    Output class for salient pipeline.
    Args:
        salient_np (`np.ndarray`):
            Salient array, with values in the range of [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    salient_np: np.ndarray
    uncertainty: Union[None, np.ndarray]


class SalientEstimationPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    salient_latent_scale_factor = 0.18215

    def __init__(self,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 scheduler: DDIMScheduler,
                 ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(self,
                 input_image: Image,
                 input_x: Image,
                 denoising_steps: int = 10,
                 ensemble_size: int = 10,
                 processing_res: int = 384,
                 match_input_res: bool = True,
                 batch_size: int = 0,
                 res2net_path: str = '',
                 ensemble_kwargs: Dict = None,
                 ) -> SalientPipelineOutput:

        # inherit from thea Diffusion Pipeline
        device = self.device
        input_size = input_image.size

        self.diffSOD_backbone = SalientBranch_Res2Net101(pretrained=False)
        self.diffSOD_backbone.load_state_dict(torch.load(res2net_path), strict=True)
        self.diffSOD_backbone.to(self.device)

        # adjust the input resolution.
        if not match_input_res:
            assert (
                    processing_res is not None
            ), " Value Error: `resize_output_back` is only valid with "

        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
            input_x = resize_max_res(
                input_x, max_edge_resolution=processing_res
            )
        # Convert the image to RGB and Normalize
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, c] -> [c, H, W]
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # Convert the x to RGB and Normalize
        input_x = input_x.convert("RGB")
        image_x = np.array(input_x)
        rgb_x = np.transpose(image_x, (2, 0, 1))  # [H, W, c] -> [c, H, W]
        rgb_x_norm = rgb_x / 255.0
        rgb_x_norm = torch.from_numpy(rgb_x_norm).to(self.dtype)
        rgb_x_norm = rgb_x_norm.to(device)
        assert rgb_x_norm.min() >= 0.0 and rgb_x_norm.max() <= 1.0

        # ----------------- predicting salient -----------------
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        duplicated_rgb_x = torch.stack([rgb_x_norm] * ensemble_size)
        single_rgb_x_dataset = TensorDataset(duplicated_rgb_x)

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=batch_size, shuffle=False)
        single_rgb_x_loader = DataLoader(single_rgb_x_dataset, batch_size=batch_size, shuffle=False)

        loader = tzip(single_rgb_loader, single_rgb_x_loader)
        # predict the salient
        salient_pred_ls = []
        iterable_bar = tqdm(
            loader, desc=" " * 2 + "Inference batches", leave=False
        )

        for _, ((batched_image,), (batch_x,)) in enumerate(iterable_bar):
            salient_pred_raw = self.single_infer(
                rgb_in=batched_image,
                rgb_x=batch_x,
                num_inference_steps=denoising_steps,
            )
            salient_pred_ls.append(salient_pred_raw.detach().clone())

        salient_preds = torch.concat(salient_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            salient_pred, salient_uncert = ensemble_masks(salient_preds, **(ensemble_kwargs or {}))
        else:
            salient_pred = salient_preds
            salient_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(salient_pred)
        max_d = torch.max(salient_pred)
        salient_pred = (salient_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        salient_pred = salient_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(salient_pred)
            pred_img = pred_img.resize(input_size)
            salient_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        salient_pred = salient_pred.clip(0, 1)

        return SalientPipelineOutput(
            salient_np=salient_pred,
            uncertainty=salient_uncert)

    @torch.no_grad()
    def single_infer(self, rgb_in: torch.Tensor, rgb_x: torch.Tensor, num_inference_steps: int):
        device = rgb_in.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial salient (noise)
        salient_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype)

        fusion_feats = self.diffSOD_backbone.forward_encode(rgb_in, rgb_x)
        nn, hh, ww, cc = fusion_feats.shape
        fusion_feats = fusion_feats.reshape(nn, -1, cc)

        # Denoising loop
        iterable = tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            leave=False,
            desc=" " * 4 + "Diffusion denoising",
        )

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, salient_latent], dim=1)
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=fusion_feats).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            salient_latent = self.scheduler.step(noise_pred, t, salient_latent).prev_sample

        torch.cuda.empty_cache()
        salient = self.decode_salient(salient_latent)
        # clip prediction
        salient = torch.clip(salient, -1.0, 1.0)
        # shift to [0, 1]
        salient = (salient + 1.0) / 2.0

        return salient

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor

        return rgb_latent

    def decode_salient(self, salient_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent into salient map.
        """
        salient_latent = salient_latent / self.salient_latent_scale_factor
        z = self.vae.post_quant_conv(salient_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        salient_mean = stacked.mean(dim=1, keepdim=True)
        return salient_mean
