import time
from diffusers import StableDiffusionOnnxPipeline
import onnxruntime as rt

sess_options = rt.SessionOptions()
#sess_options.enable_profiling = True
#sess_options.optimized_model_filepath = "test.onnx"
start_time = time.time()
pipe = StableDiffusionOnnxPipeline.from_pretrained(r"C:\Users\GAME\Documents\Project\AIGC\Olive\optimized\runwayml\stable-diffusion-v1-5", provider="DmlExecutionProvider",sess_options=sess_options)
end_time1 = time.time()
print("Time :", end_time1 - start_time)
prompt = "a photo of an astronaut riding a horse on mars"

import numpy as np
#generator = np.random.RandomState(2194967295)
latents = np.load(r"C:\Users\GAME\Documents\Project\AIGC\latents_fp16.npy").astype(np.float16)

image = pipe(prompt,num_inference_steps = 20
        ,  latents = latents
             ).images[0]
end_time = time.time()
print("Time :", end_time - end_time1)
image.save("mansion.png")
