# https://github.com/Woolverine94/biniou
# r_esrgan.py
import gradio as gr
import os
import PIL
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import huggingface_hub
# from realesrgan import RealESRGANModel as RE
from ressources.common import *
from ressources.gfpgan import *

device_label_resrgan, model_arch = detect_device()
device_resrgan = torch.device(device_label_resrgan)

model_path_resrgan = "./models/Real_ESRGAN/"
os.makedirs(model_path_resrgan, exist_ok=True)

model_list_resrgan = [
    "RealESRGAN_x2.pth",
    "RealESRGAN_x4.pth",
    "RealESRGAN_x8.pth",
]

@metrics_decoration
def image_resrgan(
    modelid_resrgan,
    scale_resrgan, 
    img_resrgan, 
    use_gfpgan_resrgan, 
    progress_resrgan=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Real ESRGAN 🔎]: starting module")

    huggingface_hub.cached_download = lambda *args, **kwargs: hf_hub_download("sberbank-ai/Real-ESRGAN", modelid_resrgan, local_dir=model_path_resrgan, **kwargs)
    from RealESRGAN import RealESRGAN

    hf_hub_download("sberbank-ai/Real-ESRGAN", "RealESRGAN_x2.pth", local_dir=model_path_resrgan)
    hf_hub_download("sberbank-ai/Real-ESRGAN", "RealESRGAN_x4.pth", local_dir=model_path_resrgan)
    hf_hub_download("sberbank-ai/Real-ESRGAN", "RealESRGAN_x8.pth", local_dir=model_path_resrgan)

    model_resrgan_path  = os.path.join(model_path_resrgan, modelid_resrgan)
    device = torch.device(device_resrgan)
    model_resrgan = RealESRGAN(device, scale=RESRGAN_SCALES[scale_resrgan])
    model_resrgan.load_weights(model_resrgan_path, download=True)
    image = Image.open(img_resrgan).convert('RGB')
    sr_image = model_resrgan.predict(image)
    final_image = [] 
    savename = name_image()
    if use_gfpgan_resrgan == True :
        sr_image = image_gfpgan_mini(sr_image)    
    sr_image.save(savename)
    final_image.append(savename)

    print(f">>>[Real ESRGAN 🔎]: generated 1 batch(es) of 1")
    reporting_resrgan = f">>>[Real ESRGAN 🔎]: "+\
        f"Settings : Model={modelid_resrgan} | "+\
        f"Scale={scale_resrgan} | "+\
        f"GFPGAN={use_gfpgan_resrgan} | "
    print(reporting_resrgan)

    exif_writer_png(reporting_resrgan, final_image)

    del model_resrgan, image, sr_image
    clean_ram()

    print(f">>>[Real ESRGAN 🔎]: leaving module")
    return final_image, final_image
