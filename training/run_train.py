import os
import sys
import math
import timm
import torch
import shutil
import logging
import argparse
import datasets
import transformers
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from packaging import version
import torch.nn.functional as F
from transformers.utils import ContextManagers

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import DiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL

sys.path.append("..")
from dataloader.dataset_configuration import prepare_dataset, gt_normalization
from Inference.salient_pipeline import SalientEstimationPipeline
from utils.image_util import pyramid_noise_like
from lib.SalientBranch import SalientBranch_Res2Net101


logger = get_logger(__name__, log_level="INFO")
epoch_val_losses = []


def log_validation(vae, unet, args, accelerator, scheduler, epoch):
    global epoch_val_losses
    denoise_steps = 10
    ensemble_size = 1
    match_input_res = True
    batch_size = 1
    logger.info("Running validation ... ")
    pipeline = SalientEstimationPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        scheduler=accelerator.unwrap_model(scheduler))

    pipeline = pipeline.to(accelerator.device)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    total_loss = 0.0
    with torch.no_grad():
        for img_name in os.listdir(args.val_img):
            input_image_path = os.path.join(args.val_img, img_name)
            input_x_path = os.path.join(args.val_x, img_name)
            gt_path = os.path.join(args.val_gt, img_name.split('.')[0] + '.png')
            input_gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.
            if len(input_gt.shape) == 3:
                input_gt = input_gt[:, :, 0]
            input_gt = torch.from_numpy(input_gt).to(accelerator.device)
            input_gt = input_gt.unsqueeze(0).unsqueeze(0)
            input_image_pil = Image.open(input_image_path)
            input_x_pil = Image.open(input_x_path)
            pipe_out = pipeline(input_image_pil,
                                input_x_pil,
                                denoising_steps=denoise_steps,
                                ensemble_size=ensemble_size,
                                processing_res=args.processing_res,
                                match_input_res=match_input_res,
                                batch_size=batch_size,
                                res2net_path=os.path.join(args.output_dir, f"res2net_ft_{epoch}.pth"),
                                )
            salient_pred = pipe_out.salient_np
            salient_pred = torch.from_numpy(salient_pred).to(accelerator.device)
            salient_pred = salient_pred.unsqueeze(0).unsqueeze(0)
            total_loss += F.l1_loss(input_gt, salient_pred).item()

        del pipeline
        torch.cuda.empty_cache()
    total_loss /= len(os.listdir(args.val_img))
    epoch_val_losses.append(total_loss)
    logger.info(f"Epoch {epoch} {total_loss}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion-Based Image Generators for salient Detection")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_img_list",
        type=str,
        default="img_list.txt",
        required=True,
        help="train file listing the training image files",
    )
    parser.add_argument(
        "--train_x_list",
        type=str,
        default="x_list.txt",
        required=True,
        help="train file listing the modality x files",
    )
    parser.add_argument(
        "--train_gt_list",
        type=str,
        default="gt_list.txt",
        required=True,
        help="train file listing the training gt files",
    )
    parser.add_argument(
        "--val_img",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_x",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_gt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=500)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=50,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="salient-detection",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)"
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations)"
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=384,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 384.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    
    
def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    # set the warning levels
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'scheduler'))
    logger.info("loading the noise scheduler from {}".format(args.pretrained_model_name_or_path), main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'vae'))
        unet = UNet2DConditionModel.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'unet'),
                                                    in_channels=8,
                                                    sample_size=48,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)

    diffSOD_backbone = SalientBranch_Res2Net101(pretrained=True)

    # Freeze vae and set unet to trainable.
    vae.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        list(unet.parameters()) + list(diffSOD_backbone.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    with accelerator.main_process_first():
        train_loader = prepare_dataset(
            train_img_list=args.train_img_list,
            train_x_list=args.train_x_list,
            train_gt_list=args.train_gt_list,
            batch_size=args.train_batch_size,
            datathread=args.dataloader_num_workers,
            processing_res=args.processing_res,
            logger=logger)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(unet, optimizer, train_loader, lr_scheduler)

    # scale factor.
    rgb_latent_scale_factor = 0.18215
    mask_latent_scale_factor = 0.18215

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move vae and diffSOD_backbone to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    diffSOD_backbone.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Here is the DDP training: actually is 32
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    epoch_train_losses = []
    min_loss = float('inf')
    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        diffSOD_backbone.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with (accelerator.accumulate(unet)):
                salient_image = batch['original']
                salient_x = batch['x']
                salient_gt = batch['gt']
                salient_gt = gt_normalization(salient_gt)

                # encode image to latents
                h_rgb = vae.encoder(salient_image.to(weight_dtype))
                moments_rgb = vae.quant_conv(h_rgb)
                mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                rgb_latents = mean_rgb * rgb_latent_scale_factor

                # encode mask to latents
                h_mask = vae.encoder(salient_gt.to(weight_dtype))
                moments_mask = vae.quant_conv(h_mask)
                mean_mask, logvar_mask = torch.chunk(moments_mask, 2, dim=1)
                mask_latents = mean_mask * mask_latent_scale_factor
                
                noise = torch.randn_like(mask_latents)
                if args.multires_noise_iterations:
                    noise = pyramid_noise_like(noise, mask_latents.device, args.multires_noise_iterations, args.multires_noise_discount)
                bsz = mask_latents.shape[0]

                # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denoising.
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=mask_latents.device)
                timesteps = timesteps.long()
                
                # add noise to the mask lantents
                noisy_mask_latents = noise_scheduler.add_noise(mask_latents, noise, timesteps)

                # encode res2net to cross_kv
                fusion_feats = diffSOD_backbone.forward_encode(salient_image.to(weight_dtype), salient_x.to(weight_dtype))
                nn, hh, ww, cc = fusion_feats.shape
                fusion_feats = fusion_feats.reshape(nn, -1, cc)
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(mask_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # predict the noise residual and compute the loss.
                unet_input = torch.cat([rgb_latents, noisy_mask_latents], dim=1)
                noise_pred = unet(unet_input, timesteps, encoder_hidden_states=fusion_feats).sample
                loss = F.mse_loss(noise_pred.float(), target.float())
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                epoch_train_losses.append(loss.detach().item())
                # Back propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # saving the checkpoints
                if global_step % args.checkpointing_steps == 0 or (
                        loss.detach().item() < min_loss and loss.detach().item() < 0.01):
                    min_loss = loss.detach().item()
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            torch.save(diffSOD_backbone.state_dict(), os.path.join(args.output_dir, f"res2net_ft_{epoch}.pth"))
            log_validation(
                vae=vae,
                unet=unet,
                args=args,
                accelerator=accelerator,
                scheduler=noise_scheduler,
                epoch=epoch
            )
            last_value = epoch_val_losses[-1]
            is_min_value = last_value == min(epoch_val_losses)
            if is_min_value:
                os.makedirs(os.path.join(args.output_dir, "valcheckpoint"), exist_ok=True)
                save_valpath = os.path.join(args.output_dir, "valcheckpoint", f"checkpoint-{global_step}")
                shutil.rmtree(os.path.join(args.output_dir, "valcheckpoint"))
                accelerator.save_state(save_valpath)

            save_trainloss = os.path.join(args.output_dir, "train_losses.txt")
            with open(save_trainloss, 'w') as file:
                for loss in epoch_train_losses:
                    file.write(f"{loss}\n")
            print(f"Values overwritten to {save_trainloss}")

            save_valloss = os.path.join(args.output_dir, "val_losses.txt")
            with open(save_valloss, 'w') as file:
                for loss in epoch_val_losses:
                    file.write(f"{loss}\n")
            print(epoch_val_losses)
            print(f"Values overwritten to {save_valloss}")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    

if __name__ == "__main__":
    main()
