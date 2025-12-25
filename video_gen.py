import torch
from PIL import Image
from dit import VideoGeneration
from diffusers.utils import load_image, export_to_video

torch.cuda.set_device(0)
model_dtype, torch_dtype = 'bf16', torch.bfloat16   # Use bf16 (not support fp16 yet)

model = VideoGeneration(
    'PATH',                                         # The downloaded checkpoint dir
    model_name="pyramid_flux",
    model_dtype=model_dtype,
    model_variant='diffusion_transformer_768p',
)

model.vae.enable_tiling()
model.enable_sequential_cpu_offload()


prompt = "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors"

# used for 384p model variant
# width = 640
# height = 384

# used for 768p model variant
width = 1280
height = 768

with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch_dtype):
    frames = model.generate(
        prompt=prompt,
        num_inference_steps=[20, 20, 20],
        video_num_inference_steps=[10, 10, 10],
        height=height,     
        width=width,
        temp=16,                    # temp=16: 5s, temp=31: 10s
        guidance_scale=7.0,         # The guidance for the first frame, set it to 7 for 384p variant
        video_guidance_scale=5.0,   # The guidance for the other video latent
        output_type="pil",
        save_memory=True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
    )

export_to_video(frames, "./text_to_video_sample.mp4", fps=24)