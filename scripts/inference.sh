inference_single_image(){
input_rgb_path='path for input rgb'
input_x_path='path for input modality x'
output_dir='../output'
stable_diffusion_repo_path='path for stable-diffusion-2'
pretrained_model_path='path for unet'
res2net_model_path='path for fine-tuned res2net'
ensemble_size=10

cd ..
cd Inference

CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --input_rgb_path $input_rgb_path \
    --input_x_path $input_x_path \
    --output_dir $output_dir \
    --stable_diffusion_repo_path $stable_diffusion_repo_path \
    --pretrained_model_path $pretrained_model_path \
    --res2net_model_path $res2net_model_path \
    --ensemble_size $ensemble_size \

}

inference_single_image
