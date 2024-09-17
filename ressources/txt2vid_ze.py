# https://github.com/Woolverine94/biniou
# txt2vid_ze.py
import gradio as gr
import os
import imageio
from diffusers import TextToVideoZeroPipeline, TextToVideoZeroSDXLPipeline
import numpy as np
import torch
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_txt2vid_ze, model_arch = detect_device()
device_txt2vid_ze = torch.device(device_label_txt2vid_ze)

model_path_txt2vid_ze = "./models/Stable_Diffusion/"
os.makedirs(model_path_txt2vid_ze, exist_ok=True)

model_list_txt2vid_ze = [
#     "SG161222/Realistic_Vision_V3.0_VAE",
#     "SG161222/Paragon_V1.0",
#     "digiplay/majicMIX_realistic_v7",
#     "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
#     "sd-community/sdxl-flash",
#     "dataautogpt3/PrometheusV1",
#     "mann-e/Mann-E_Dreams",
#     "mann-e/Mann-E_Art",
#     "ehristoforu/Visionix-alpha",
#     "RunDiffusion/Juggernaut-X-Hyper",
#     "cutycat2000x/InterDiffusion-4.0",
#     "RunDiffusion/Juggernaut-XL-Lightning",
#     "fluently/Fluently-XL-v3-Lightning",
#     "Corcelio/mobius",
#     "fluently/Fluently-XL-Final",
#     "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
#     "recoilme/ColorfulXL-Lightning",
#     "playgroundai/playground-v2-512px-base",
#     "playgroundai/playground-v2-1024px-aesthetic",
# #    "playgroundai/playground-v2.5-1024px-aesthetic",
#     "stabilityai/sdxl-turbo",
#     "SG161222/RealVisXL_V4.0_Lightning",
#     "cagliostrolab/animagine-xl-3.1",
#     "aipicasso/emi-2",
#     "dataautogpt3/OpenDalleV1.1",
#     "dataautogpt3/ProteusV0.5",
#     "dataautogpt3/ProteusV0.4-Lightning",
#     "etri-vilab/koala-lightning-1b",
#     "etri-vilab/koala-lightning-700m",
#     "digiplay/AbsoluteReality_v1.8.1",
#     "segmind/Segmind-Vega",
#     "segmind/SSD-1B",
#     "gsdf/Counterfeit-V2.5",
# #    "ckpt/anything-v4.5-vae-swapped",
#     "runwayml/stable-diffusion-v1-5",
#     "nitrosocke/Ghibli-Diffusion",

    "-[ 👍 SD15 ]-",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "Yntec/VisionVision",
    "fluently/Fluently-epic",
    "SG161222/Paragon_V1.0",
    "digiplay/AbsoluteReality_v1.8.1",
    "digiplay/majicMIX_realistic_v7",
    "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
    "runwayml/stable-diffusion-v1-5",
    "-[ 👍 🇯🇵 Anime SD15 ]-",
    "gsdf/Counterfeit-V2.5",
    "fluently/Fluently-anime",
    "xyn-ai/anything-v4.0",
    "nitrosocke/Ghibli-Diffusion",
    "-[ 👌 🐢 SDXL ]-",
    "fluently/Fluently-XL-Final",
    "Corcelio/mobius",
    "misri/juggernautXL_juggXIByRundiffusion",
    "mann-e/Mann-E_Dreams",
    "mann-e/Mann-E_Art",
    "ehristoforu/Visionix-alpha",
    "cutycat2000x/InterDiffusion-4.0",
    "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
    "etri-vilab/koala-lightning-700m",
    "etri-vilab/koala-lightning-1b",
    "GraydientPlatformAPI/flashback-xl",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/PrometheusV1",
    "dataautogpt3/OpenDalleV1.1",
    "segmind/SSD-1B",
    "segmind/Segmind-Vega",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "-[ 👌 🚀 Fast SDXL ]-",
    "sd-community/sdxl-flash",
    "fluently/Fluently-XL-v3-Lightning",
    "GraydientPlatformAPI/epicrealism-lightning-xl",
    "Lykon/dreamshaper-xl-lightning",
    "RunDiffusion/Juggernaut-XL-Lightning",
    "RunDiffusion/Juggernaut-X-Hyper",
    "SG161222/RealVisXL_V5.0_Lightning",
    "dataautogpt3/ProteusV0.4-Lightning",
    "recoilme/ColorfulXL-Lightning",
    "GraydientPlatformAPI/lustify-lightning",
    "stabilityai/sdxl-turbo",
    "-[ 👌 🇯🇵 Anime SDXL ]-",
    "cagliostrolab/animagine-xl-3.1",
    "GraydientPlatformAPI/sanae-xl",
    "yodayo-ai/clandestine-xl-1.0",
    "stablediffusionapi/anime-journey-v2",
    "aipicasso/emi-2",
]

# Bouton Cancel
stop_txt2vid_ze = False

def initiate_stop_txt2vid_ze() :
    global stop_txt2vid_ze
    stop_txt2vid_ze = True

def check_txt2vid_ze(step, timestep, latents) : 
    global stop_txt2vid_ze
    if stop_txt2vid_ze == False :
        return
    elif stop_txt2vid_ze == True :
        print(">>>[Text2Video-Zero 📼 ]: generation canceled by user")
        stop_txt2vid_ze = False
        try:
            del ressources.txt2vid_ze.pipe_txt2vid_ze
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def video_txt2vid_ze(
    modelid_txt2vid_ze, 
    num_inference_step_txt2vid_ze, 
    sampler_txt2vid_ze, 
    guidance_scale_txt2vid_ze, 
    seed_txt2vid_ze, 
    num_frames_txt2vid_ze, 
    num_fps_txt2vid_ze, 
    height_txt2vid_ze, 
    width_txt2vid_ze, 
    num_videos_per_prompt_txt2vid_ze, 
    num_prompt_txt2vid_ze, 
    motion_field_strength_x_txt2vid_ze, 
    motion_field_strength_y_txt2vid_ze, 
    timestep_t0_txt2vid_ze :int, 
    timestep_t1_txt2vid_ze :int, 
    prompt_txt2vid_ze, 
    negative_prompt_txt2vid_ze, 
    output_type_txt2vid_ze,
    nsfw_filter, 
    num_chunks_txt2vid_ze :int, 
    use_gfpgan_txt2vid_ze,
    tkme_txt2vid_ze,
    progress_txt2vid_ze=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[Text2Video-Zero 📼 ]: starting module")

    modelid_txt2vid_ze = model_cleaner_sd(modelid_txt2vid_ze)

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2vid_ze, device_txt2vid_ze, nsfw_filter)

    if is_sdxl(modelid_txt2vid_ze):
        is_xl_txt2vid_ze: bool = True
    else :        
        is_xl_txt2vid_ze: bool = False

    if is_bin(modelid_txt2vid_ze):
        is_bin_txt2vid_ze: bool = True
    else :
        is_bin_txt2vid_ze: bool = False

    if (is_xl_txt2vid_ze == True):
        pipe_txt2vid_ze = TextToVideoZeroSDXLPipeline.from_pretrained(
            modelid_txt2vid_ze, 
            cache_dir=model_path_txt2vid_ze, 
            torch_dtype=model_arch, 
            use_safetensors=True if not is_bin_txt2vid_ze else False,
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    else:
        pipe_txt2vid_ze = TextToVideoZeroPipeline.from_pretrained(
            modelid_txt2vid_ze, 
            cache_dir=model_path_txt2vid_ze, 
            torch_dtype=model_arch, 
            use_safetensors=True if not is_bin_txt2vid_ze else False,
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )

    pipe_txt2vid_ze = schedulerer(pipe_txt2vid_ze, sampler_txt2vid_ze)
    tomesd.apply_patch(pipe_txt2vid_ze, ratio=tkme_txt2vid_ze)
    if device_label_txt2vid_ze == "cuda" :
        pipe_txt2vid_ze.enable_sequential_cpu_offload()
    else : 
        pipe_txt2vid_ze = pipe_txt2vid_ze.to(device_txt2vid_ze)
#    pipe_txt2vid_ze.enable_vae_slicing()
    
    if seed_txt2vid_ze == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2vid_ze
    generator = []
    for k in range(num_prompt_txt2vid_ze):
        generator.append(torch.Generator(device_txt2vid_ze).manual_seed(final_seed + k))

    if output_type_txt2vid_ze == "gif" :
        savename = []
    final_seed = []
    for j in range (num_prompt_txt2vid_ze):
        if num_chunks_txt2vid_ze != 1 :
            result = []
            chunk_ids = np.arange(0, num_frames_txt2vid_ze, num_chunks_txt2vid_ze)
#            generator = torch.Generator(device=device_txt2vid_ze)
            for i in range(len(chunk_ids)):
                print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
                ch_start = chunk_ids[i]
                ch_end = num_frames_txt2vid_ze if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                if i == 0 :
                    frame_ids = [0] + list(range(ch_start, ch_end))
                else :
                    frame_ids = [ch_start -1] + list(range(ch_start, ch_end))
#                generator = generator.manual_seed(seed_txt2vid_ze)
                output = pipe_txt2vid_ze(
                    prompt=prompt_txt2vid_ze,
                    negative_prompt=negative_prompt_txt2vid_ze,
                    height=height_txt2vid_ze,
                    width=width_txt2vid_ze,
                    num_inference_steps=num_inference_step_txt2vid_ze,
                    guidance_scale=guidance_scale_txt2vid_ze,
                    frame_ids=frame_ids,
                    video_length=len(frame_ids), 
                    num_videos_per_prompt=num_videos_per_prompt_txt2vid_ze,
                    motion_field_strength_x=motion_field_strength_x_txt2vid_ze,
                    motion_field_strength_y=motion_field_strength_y_txt2vid_ze,
                    t0=timestep_t0_txt2vid_ze,
                    t1=timestep_t1_txt2vid_ze,
                    generator = generator[j],
                    callback=check_txt2vid_ze, 
                )
                result.append(output.images[1:])
            result = np.concatenate(result)
        else :
            result = pipe_txt2vid_ze(
                prompt=prompt_txt2vid_ze,
                negative_prompt=negative_prompt_txt2vid_ze,
                height=height_txt2vid_ze,
                width=width_txt2vid_ze,
                num_inference_steps=num_inference_step_txt2vid_ze,
                guidance_scale=guidance_scale_txt2vid_ze,
                video_length=num_frames_txt2vid_ze,
                num_videos_per_prompt=num_videos_per_prompt_txt2vid_ze,
                motion_field_strength_x=motion_field_strength_x_txt2vid_ze,
                motion_field_strength_y=motion_field_strength_y_txt2vid_ze,
                t0=timestep_t0_txt2vid_ze,
                t1=timestep_t1_txt2vid_ze,
                generator = generator[j],
                callback=check_txt2vid_ze, 
            ).images
         
        result = [(r * 255).astype("uint8") for r in result]

        for n in range(len(result)):
            if use_gfpgan_txt2vid_ze == True :
                result[n] = image_gfpgan_mini(result[n])

        a = 1
        b = 0
        for o in range(len(result)):
            if (a < num_frames_txt2vid_ze):
                a += 1
            elif (a == num_frames_txt2vid_ze):
                seed_id = random_seed + j*num_videos_per_prompt_txt2vid_ze + b if (seed_txt2vid_ze == 0) else seed_txt2vid_ze + j*num_videos_per_prompt_txt2vid_ze + b
                if output_type_txt2vid_ze == "mp4" :
                    savename = name_seeded_video(seed_id)
                    imageio.mimsave(savename, result, fps=num_fps_txt2vid_ze)
                elif output_type_txt2vid_ze == "gif" :
                    savename_gif = name_seeded_gif(seed_id)
                    imageio.mimsave(savename_gif, result, format='GIF', loop=0, fps=num_fps_txt2vid_ze)
                    savename.append(savename_gif)
                final_seed.append(seed_id)
                a = 1
                b += 1

    print(f">>>[Text2Video-Zero 📼 ]: generated {num_prompt_txt2vid_ze} batch(es) of {num_videos_per_prompt_txt2vid_ze}")
    reporting_txt2vid_ze = f">>>[Text2Video-Zero 📼 ]: "+\
        f"Settings : Model={modelid_txt2vid_ze} | "+\
        f"Sampler={sampler_txt2vid_ze} | "+\
        f"Steps={num_inference_step_txt2vid_ze} | "+\
        f"CFG scale={guidance_scale_txt2vid_ze} | "+\
        f"Video length={num_frames_txt2vid_ze} frames | "+\
        f"FPS={num_fps_txt2vid_ze} frames | "+\
        f"Chunck size={num_chunks_txt2vid_ze} | "+\
        f"Size={width_txt2vid_ze}x{height_txt2vid_ze} | "+\
        f"Motion field strength x={motion_field_strength_x_txt2vid_ze} | "+\
        f"Motion field strength y={motion_field_strength_y_txt2vid_ze} | "+\
        f"Timestep t0={timestep_t0_txt2vid_ze} | "+\
        f"Timestep t1={timestep_t1_txt2vid_ze} | "+\
        f"GFPGAN={use_gfpgan_txt2vid_ze} | "+\
        f"Token merging={tkme_txt2vid_ze} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2vid_ze} | "+\
        f"Negative prompt={negative_prompt_txt2vid_ze} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2vid_ze) 

    if output_type_txt2vid_ze == "mp4":
        metadata_writer_mp4(reporting_txt2vid_ze, savename)
    elif output_type_txt2vid_ze == "gif":
        metadata_writer_gif(reporting_txt2vid_ze, savename, num_fps_txt2vid_ze)

    del nsfw_filter_final, feat_ex, pipe_txt2vid_ze, generator, result
    clean_ram()

    print(f">>>[Text2Video-Zero 📼 ]: leaving module")
    return savename
