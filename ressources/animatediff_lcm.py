# https://github.com/Woolverine94/biniou
# animatediff_lcm.py
import gradio as gr
import os
import imageio
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import export_to_video
import numpy as np
import torch
import random
from ressources.common import *
from ressources.gfpgan import *
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import load_file
import tomesd

device_label_animatediff_lcm, model_arch = detect_device()
device_animatediff_lcm = torch.device(device_label_animatediff_lcm)

model_path_animatediff_lcm = "./models/Stable_Diffusion/"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)

adapter_path_animatediff_lcm = "./models/AnimateLCM/"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)

lora_path_animatediff_lcm = "./models/AnimateLCM/LoRA"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)


model_list_animatediff_lcm = [
    "emilianJR/epiCRealism",
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "stabilityai/sdxl-turbo",
#    "dataautogpt3/OpenDalleV1.1",
    "digiplay/AbsoluteReality_v1.8.1",
#    "segmind/Segmind-Vega",
#    "segmind/SSD-1B",
    "gsdf/Counterfeit-V2.5",
#    "ckpt/anything-v4.5-vae-swapped",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
]

model_list_adapters_animatediff_lcm = {
    "wangfuyun/AnimateLCM":("AnimateLCM_sd15_t2v_lora.safetensors", 0.8),
    "ByteDance/AnimateDiff-Lightning":("animatediff_lightning_4step_diffusers.safetensors", 1.0),
}

# Bouton Cancel
stop_animatediff_lcm = False

def initiate_stop_animatediff_lcm() :
    global stop_animatediff_lcm
    stop_animatediff_lcm = True

def check_animatediff_lcm(pipe, step_index, timestep, callback_kwargs) :
    global stop_animatediff_lcm
    if stop_animatediff_lcm == True :
        print(">>>[AnimateLCM 📼 ]: generation canceled by user")
        stop_animatediff_lcm = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
def video_animatediff_lcm(
    modelid_animatediff_lcm,
    modelid_adapters_animatediff_lcm,
    num_inference_step_animatediff_lcm,
    sampler_animatediff_lcm,
    guidance_scale_animatediff_lcm,
    seed_animatediff_lcm,
    num_frames_animatediff_lcm,
    num_fps_animatediff_lcm,
    height_animatediff_lcm,
    width_animatediff_lcm,
    num_videos_per_prompt_animatediff_lcm,
    num_prompt_animatediff_lcm,
    prompt_animatediff_lcm,
    negative_prompt_animatediff_lcm,
    nsfw_filter,
    use_gfpgan_animatediff_lcm,
    tkme_animatediff_lcm,
    progress_animatediff_lcm=gr.Progress(track_tqdm=True)
    ):

    print(">>>[AnimateLCM 📼 ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_animatediff_lcm, device_animatediff_lcm, nsfw_filter)

    if (modelid_adapters_animatediff_lcm == "wangfuyun/AnimateLCM"):
        adapter_animatediff_lcm = MotionAdapter.from_pretrained(
            modelid_adapters_animatediff_lcm,
            cache_dir=adapter_path_animatediff_lcm,
            torch_dtype=model_arch,
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    elif (modelid_adapters_animatediff_lcm == "ByteDance/AnimateDiff-Lightning"):
        if not os.path.isfile(f"{adapter_path_animatediff_lcm}/{model_list_adapters_animatediff_lcm[modelid_adapters_animatediff_lcm][0]}"):
            hf_hub_download(
                repo_id=modelid_adapters_animatediff_lcm,
                filename=model_list_adapters_animatediff_lcm[modelid_adapters_animatediff_lcm][0],
                repo_type="model",
                cache_dir=adapter_path_animatediff_lcm,
                local_dir=adapter_path_animatediff_lcm,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        adapter_animatediff_lcm = MotionAdapter()
        adapter_animatediff_lcm.load_state_dict(load_file(f"{adapter_path_animatediff_lcm}/{model_list_adapters_animatediff_lcm[modelid_adapters_animatediff_lcm][0]}"))

    pipe_animatediff_lcm = AnimateDiffPipeline.from_pretrained(
        modelid_animatediff_lcm,
        cache_dir=model_path_animatediff_lcm, 
        torch_dtype=model_arch, 
        motion_adapter=adapter_animatediff_lcm,
        use_safetensors=True, 
        safety_checker=nsfw_filter_final, 
        feature_extractor=feat_ex, 
        resume_download=True,
        local_files_only=True if offline_test() else None
    )

    if (modelid_adapters_animatediff_lcm == "wangfuyun/AnimateLCM"):
        pipe_animatediff_lcm.load_lora_weights(
            modelid_adapters_animatediff_lcm,
            weight_name=model_list_adapters_animatediff_lcm[modelid_adapters_animatediff_lcm][0],
            cache_dir=lora_path_animatediff_lcm,
            use_safetensors=True,
            adapter_name="adapter1",
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        pipe_animatediff_lcm.fuse_lora(lora_scale=model_list_adapters_animatediff_lcm[modelid_adapters_animatediff_lcm][1])
    #    pipe_animatediff_lcm.fuse_lora(lora_scale=lora_weight_animatediff_lcm)
    #    pipe_animatediff_lcm.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_animatediff_lcm)])
        pipe_animatediff_lcm.scheduler = LCMScheduler.from_config(pipe_animatediff_lcm.scheduler.config, beta_schedule="linear")
    elif (modelid_adapters_animatediff_lcm == "ByteDance/AnimateDiff-Lightning"):
        pipe_animatediff_lcm.scheduler = EulerDiscreteScheduler.from_config(pipe_animatediff_lcm.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

#    pipe_animatediff_lcm = schedulerer(pipe_animatediff_lcm, sampler_animatediff_lcm)
#    tomesd.apply_patch(pipe_animatediff_lcm, ratio=tkme_animatediff_lcm)
    if device_label_animatediff_lcm == "cuda" :
        pipe_animatediff_lcm.enable_sequential_cpu_offload()
    else : 
        pipe_animatediff_lcm = pipe_animatediff_lcm.to(device_animatediff_lcm)
    pipe_animatediff_lcm.enable_vae_slicing()

    if seed_animatediff_lcm == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_animatediff_lcm
    generator = []
    for k in range(num_prompt_animatediff_lcm):
        generator.append(torch.Generator(device_animatediff_lcm).manual_seed(final_seed + k))

    final_seed = []
    for i in range (num_prompt_animatediff_lcm):
        result = pipe_animatediff_lcm(
            prompt=prompt_animatediff_lcm,
            negative_prompt=negative_prompt_animatediff_lcm,
            num_frames=num_frames_animatediff_lcm,
            height=height_animatediff_lcm,
            width=width_animatediff_lcm,
            num_inference_steps=num_inference_step_animatediff_lcm,
            guidance_scale=guidance_scale_animatediff_lcm,
            video_length=num_frames_animatediff_lcm,
            num_videos_per_prompt=num_videos_per_prompt_animatediff_lcm,
            generator = generator[i],
            callback_on_step_end=check_animatediff_lcm,
            callback_on_step_end_tensor_inputs=['latents'],
        ).frames[0]

#        result = [(r * 255).astype("uint8") for r in result]

#        for n in range(len(result)):
#            if use_gfpgan_animatediff_lcm == True :
#                result[n] = image_gfpgan_mini(result[n])
#
#        a = 1
#        b = 0
#        for o in range(len(result)):
#            if (a < num_frames_animatediff_lcm):
#                a += 1
#            elif (a == num_frames_animatediff_lcm):
#                seed_id = random_seed + j*num_videos_per_prompt_animatediff_lcm + b if (seed_animatediff_lcm == 0) else seed_animatediff_lcm + j*num_videos_per_prompt_animatediff_lcm + b
#                savename = f"outputs/{seed_id}_{timestamper()}.mp4"
#                imageio.mimsave(savename, result, fps=num_fps_animatediff_lcm)
#                final_seed.append(seed_id)
#                a = 1
#                b += 1

        timestamp = time.time()
        seed_id = random_seed + i*num_videos_per_prompt_animatediff_lcm if (seed_animatediff_lcm == 0) else seed_animatediff_lcm + i*num_videos_per_prompt_animatediff_lcm
        savename = "outputs/tmp_animatelcm_out.mp4"
        savename_final = name_seeded_video(seed_id)
        export_to_video(result, savename, fps=num_fps_animatediff_lcm)
        os.rename(savename, savename_final)
        final_seed.append(seed_id)

    print(f">>>[AnimateLCM 📼 ]: generated {num_prompt_animatediff_lcm} batch(es) of {num_videos_per_prompt_animatediff_lcm}")
    reporting_animatediff_lcm = f">>>[AnimateLCM 📼 ]: "+\
        f"Settings : Model={modelid_animatediff_lcm} | "+\
        f"Adapter={modelid_adapters_animatediff_lcm} | "+\
        f"Sampler={sampler_animatediff_lcm} | "+\
        f"Steps={num_inference_step_animatediff_lcm} | "+\
        f"CFG scale={guidance_scale_animatediff_lcm} | "+\
        f"Video length={num_frames_animatediff_lcm} frames | "+\
        f"FPS={num_fps_animatediff_lcm} frames | "+\
        f"Size={width_animatediff_lcm}x{height_animatediff_lcm} | "+\
        f"GFPGAN={use_gfpgan_animatediff_lcm} | "+\
        f"Token merging={tkme_animatediff_lcm} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_animatediff_lcm} | "+\
        f"Negative prompt={negative_prompt_animatediff_lcm} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_animatediff_lcm) 

    del nsfw_filter_final, feat_ex, pipe_animatediff_lcm, generator, result
    clean_ram()

    print(f">>>[AnimateLCM 📼 ]: leaving module")
    return savename_final
