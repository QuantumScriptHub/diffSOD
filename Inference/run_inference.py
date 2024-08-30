import os
import sys
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL


sys.path.append("../")
from Inference.salient_pipeline import SalientEstimationPipeline
from utils.seed_all import seed_all


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run salient Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--stable_diffusion_repo_path",
        type=str,
        default='stable-diffusion-2',
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='None',
        help="path for unet",
    )
    parser.add_argument(
        "--res2net_model_path",
        type=str,
        default='None',
        help="path for res2net",
    )
    parser.add_argument(
        "--input_rgb_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--input_x_path",
        type=str,
        required=True,
        help="Path to the input modality x.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=384,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 384.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output mask at resized operating resolution. Default: False.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )

    args = parser.parse_args()
    checkpoint_path = args.pretrained_model_path
    input_image_path = args.input_rgb_path
    input_x_path = args.input_x_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    seed = args.seed
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = 1
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------Data----------------------------
    logging.info("Inference Image Path from {}".format(input_image_path))

    # -------------------- Model --------------------
    vae = AutoencoderKL.from_pretrained(os.path.join(args.stable_diffusion_repo_path, 'vae'))
    scheduler = DDIMScheduler.from_pretrained(os.path.join(args.stable_diffusion_repo_path, 'scheduler'))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, 'unet'),
                                                in_channels=8, sample_size=48,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)

    pipe = SalientEstimationPipeline(unet=unet,
                                     vae=vae,
                                     scheduler=scheduler)

    logging.info("loading pipeline whole successfully.")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers
    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for img_name in os.listdir(input_image_path):
            input_image_pil = Image.open(os.path.join(input_image_path, img_name))
            input_x_pil = Image.open(os.path.join(input_x_path, img_name))
            pipe_out = pipe(input_image_pil,
                            input_x_pil,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=match_input_res,
                            batch_size=batch_size,
                            res2net_path=args.res2net_model_path,
                            )
            salient_np = pipe_out.salient_np
            pred_save_path = os.path.join(output_dir, img_name.split('.jpg')[0] + '.png')
            if os.path.exists(pred_save_path):
                logging.warning(f"Existing file: '{pred_save_path}' will be overwritten")

            salient_pred = np.expand_dims(salient_np, 2)
            salient_pred = np.repeat(salient_pred, 3, 2)
            salient_pred = (salient_pred * 255).astype(np.uint8)
            salient_pred = Image.fromarray(salient_pred)
            salient_pred.save(pred_save_path)
