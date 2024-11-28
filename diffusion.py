import numpy as np
import requests
import torch
from io import BytesIO
from PIL import Image
import cv2

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img_path = "/home/Nicholas/final_project/diffusion/images/ship/r_2.png"
img_path = "/home/Nicholas/final_project/diffusion/images/lego/r_6.png"

image = np.array(Image.open(img_path))
# size = (400, 400)
# size = (512, 512)
# image = cv2.resize(image, size)
# image = cv2.resize(image, (400, 400), cv2.INTER_AREA)
# image = cv2.resize(image, (800, 800), cv2.INTER_AREA)
image = (image / 255.0).astype(np.float32)

# NOTE: If editing a NeRF synthetic image, keep the next line uncommented
#       to convert from rgba to rgb with a white background.
#       If editing an rgb image, comment the next line out.
image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:]) # rgba -> rgb with white bkg
bkg_idxs = (image == 1.0).all(axis=2)

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", safety_checker=None, torch_dtype=torch.float32
).to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "turn the excavator pink" # default image guidance=1.5, text guidance=7.5 suffices
# prompt = "set the ship on fire" # use image guidance=1.45, text guidance=10 for more significant edits

image = pipe(
	prompt=prompt, 
	image=image, 
	output_type="np",
	num_inference_steps=20,    # 20 is default (probably should keep this too)
	image_guidance_scale=1.5,  # 1.5 is default
	guidance_scale=7.5         # 7.5 is default
).images[0]

# NOTE: If editing a NeRF synthetic image, keep the following line uncommented
image[bkg_idxs, :] = 1.0

image = Image.fromarray((image * 255.0).astype(np.uint8))
image.save("/home/Nicholas/final_project/diffusion/pink_lego.png")