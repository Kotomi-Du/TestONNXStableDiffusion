import time
import os
import numpy as np
import argparse
from diffusers import OnnxStableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler
import onnxruntime as ort

import re
def validateTitle(title):
    rstr = r"[\/\\\:\*\?\"\<\>\|.]"
    return re.sub(rstr, r"", title)

def run_inference(model_path, latents_path, prompt, num_images, batch_size, num_inference_steps, static_dims, enable_profiling, enable_intermediate_model,seed_value):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    if enable_profiling:
        sess_options.enable_profiling = True
    if enable_intermediate_model:
        sess_options.optimized_model_filepath = "test.onnx"

    if static_dims:
        # Not necessary, but helps DML EP further optimize runtime performance.
        # batch_size is doubled for sample & hidden state because of classifier free guidance:
        # https://github.com/huggingface/diffusers/blob/46c52f9b9607e6ecb29c782c052aea313e6487b7/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L672
        sess_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        sess_options.add_free_dimension_override_by_name("unet_sample_height", 64)
        sess_options.add_free_dimension_override_by_name("unet_sample_width", 64)
        #sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

        # vae_decoder
        # sess_options.add_free_dimension_override_by_name("decoder_batch", batch_size)
        # sess_options.add_free_dimension_override_by_name("decoder_channels", 4)
        # sess_options.add_free_dimension_override_by_name("decoder_height", 64)
        # sess_options.add_free_dimension_override_by_name("decoder_width", 64)

        #text_encoder
        sess_options.add_free_dimension_override_by_name("batch", 1)
        sess_options.add_free_dimension_override_by_name("sequence", 77 )
        #safe_checker
        # sess_options.add_free_dimension_override_by_name("sc_batch", 1)
        # sess_options.add_free_dimension_override_by_name("sc_channels", 3 )
        # sess_options.add_free_dimension_override_by_name("sc_input_height", 224 )
        # sess_options.add_free_dimension_override_by_name("sc_input_width", 224 )
        # sess_options.add_free_dimension_override_by_name("sc_img_width", 512 )
        # sess_options.add_free_dimension_override_by_name("sc_img_height", 512 )
        
    
    start_time = time.time()
    pipe = OnnxStableDiffusionPipeline.from_pretrained(model_path, provider="DmlExecutionProvider",sess_options=sess_options)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler") 
    pipe.scheduler = scheduler
    stage1_time = time.time()

    #generator = np.random.RandomState(2194967295)
    
    if not os.path.exists(latents_path):
        if seed_value is not None:
            print("tt")
            generator = np.random.seed(seed=seed_value)
            image = pipe(prompt,num_inference_steps = num_inference_steps, generator =generator).images[0]
        else:
            seed_value='random'
            image = pipe(prompt,num_inference_steps = num_inference_steps).images[0]
    else:
        latents = np.load(latents_path)#.astype(np.float16)
        image = pipe(prompt,num_inference_steps = num_inference_steps,  latents = latents).images[0]
    stage2_time = time.time()

    str_image = validateTitle(str(prompt))[:10] + '_seed_' + str(seed_value) + '_step_' + str(num_inference_steps)+'_time_'+str(int(time.time())) + ".png"

    image.save(str_image)
    print("Image generated from {}".format(model_path))
    print("Initialized Time: {:.2f}s".format(stage1_time - start_time))
    print("End to End Inference Time:{:.2f}s:".format(stage2_time - stage1_time))
    print("Check configuration:\n profiling:{} \n save_optimized_model: {} \n onnx optimization level: {}".format(
        sess_options.enable_profiling, 
        enable_intermediate_model,
        sess_options.graph_optimization_level))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default = r"C:\Users\GAME\Documents\Project\AIGC\Olive\olive0.2_sd1.5", type = str, help="folder of stable diffusion")
    parser.add_argument("--latents_path", default = r"C:\Users\GAME\Documents\Project\AIGC\latents_fp16.npy", type = str, help="file of latents")
    parser.add_argument("--seed_value", default=None, type=int, help="Number of seed")
    # property
    parser.add_argument("--prompt", default = "a photo of an astronaut riding a horse on mars", type = str, help="prompt")
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument("--num_inference_steps", default=20, type=int, help="Number of steps in diffusion process")
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")
    
    # profiling
    parser.add_argument("--enable_profiling", action="store_true", help="To save onnx profiling file")
    parser.add_argument("--enable_intermediate_model", action="store_true", help="To save intermediate graph optimized by onnxruntime")
  
    args = parser.parse_args()
    use_static_dims = False
    if not args.dynamic_dims:
        use_static_dims = True
    run_inference(
                args.model_dir,
                args.latents_path,
                args.prompt,
                args.num_images,
                args.batch_size,
                args.num_inference_steps,
                use_static_dims,
                args.enable_profiling,
                args.enable_intermediate_model,
                args.seed_value
            )
    
    
    