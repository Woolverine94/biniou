# https://github.com/Woolverine94/biniou
# Webui.py
# import diffusers
# diffusers.utils.USE_PEFT_BACKEND = False
from llama_cpp import Llama
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings('ignore')
import os
import subprocess
import gradio as gr
import numpy as np
# import shutil
from PIL import Image
from ressources import *
import sys
import socket

def local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0)
    try:
        sock.connect(("10.0.0.1", 1))
        host_ip = sock.getsockname()[0]
    except Exception as e:
        host_ip = "127.0.0.1"
    finally:
        sock.close()
    return host_ip

tmp_biniou="./.tmp"
if os.path.exists(tmp_biniou):
#    shutil.rmtree(tmp_biniou)
     for tmpfile in os.listdir(tmp_biniou):
         os.remove(os.path.join(tmp_biniou, tmpfile))
os.makedirs(tmp_biniou, exist_ok=True)

blankfile_common = "./.tmp/blank.txt"
with open(blankfile_common, 'w') as savefile:
    savefile.write("")

ini_dir="./.ini"
os.makedirs(ini_dir, exist_ok=True)

log_dir="./.logs"
os.makedirs(log_dir, exist_ok=True)
logfile_biniou = f"{log_dir}/output.log"
sys.stdout = Logger(logfile_biniou)

get_window_url_params = """
    function(url_params) {

        function preventPageClose() {
            // Attacher un événement au clic sur le bouton fermer du navigateur
            window.addEventListener('beforeunload', function (event) {
                // Prévenir la fermeture par défaut
                event.preventDefault();

                // Afficher un message de confirmation personnalisé
                event.returnValue = 'Voulez-vous vraiment quitter cette page ?';
            });
        }
        preventPageClose();

        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
        }
    """

def split_url_params(url_params) :
    url_params = eval(url_params.replace("'", "\""))
    if "nsfw_filter" in url_params.keys():
        output_nsfw = url_params["nsfw_filter"]
        return output_nsfw, url_params, bool(int(output_nsfw))
    else :         
        return "1", url_params, "1"

biniou_global_lang_ui = "lang_en_US"
biniou_global_server_name = True
biniou_global_server_port = 7860
biniou_global_inbrowser = False
biniou_global_auth = False
biniou_global_auth_message = "Welcome to biniou !"
biniou_global_share = False
biniou_global_steps_max = 100
biniou_global_batch_size_max = 4
biniou_global_width_max_img_create = 4096
biniou_global_height_max_img_create = 4096
biniou_global_width_max_img_modify = 8192
biniou_global_height_max_img_modify = 8192
biniou_global_sd15_width = 512
biniou_global_sd15_height = 512
biniou_global_sdxl_width = 1024
biniou_global_sdxl_height = 1024
biniou_global_gfpgan = True
biniou_global_tkme = 0.6
biniou_global_clipskip = 0
biniou_global_ays = False
biniou_global_img_fmt = "png"
biniou_global_text_metadatas = True
biniou_global_img_exif = True
biniou_global_gif_exif = True
biniou_global_mp4_metadatas = True
biniou_global_audio_metadatas = True

if test_cfg_exist("settings") :
    with open(".ini/settings.cfg", "r", encoding="utf-8") as fichier:
        exec(fichier.read())

if not os.path.isfile(".ini/auth.cfg"):
    write_auth("biniou:biniou")

if biniou_global_auth == True:
    biniou_auth_values = read_auth()

if biniou_global_auth == False:
    biniou_global_share = False

with open("./version", "r", encoding="utf-8") as fichier:
    biniou_global_version = fichier.read()

biniou_global_version = biniou_global_version.replace("\n", "")

if biniou_global_version == "main":
    commit_ver = subprocess.getoutput("git rev-parse HEAD")
    biniou_global_version = f"dev {commit_ver[0:7]}"

with open(f"lang/lang_en_US.cfg", "r", encoding="utf-8") as fichier:
    exec(fichier.read())

if test_lang_exist(f"{biniou_global_lang_ui}.cfg") and biniou_global_lang_ui != "lang_en_US":
    with open(f"lang/{biniou_global_lang_ui}.cfg", "r", encoding="utf-8") as fichier:
        exec(fichier.read())

## Fonctions communes
def dummy():
    return

def in_and_out(input_value):
    return input_value

## fonctions Exports Outputs 
def send_to_module(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) # /!\ tabs_image = pas bon pour les autres modules

def send_to_module_inpaint(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)
    
def send_to_module_text(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], gr.Tabs.update(selected=numtab), tabs_text.update(selected=numtab_item)

def send_to_module_video(content, numtab, numtab_item):
	return content, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)

def send_image_to_module_video(content, index, numtab, numtab_item):
	index = int(index)
	return content[index], gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)

def send_to_module_3d(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], gr.Tabs.update(selected=numtab), tabs_3d.update(selected=numtab_item)

def send_text_to_module_image (prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def send_audio_to_module_text(audio, numtab, numtab_item):
    return audio, gr.Tabs.update(selected=numtab), tabs_text.update(selected=numtab_item)

def send_text_to_module_text(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_text.update(selected=numtab_item)

## fonctions Exports Inputs
def import_to_module(prompt, negative_prompt, numtab, numtab_item):
    return prompt, negative_prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def import_to_module_prompt_only(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def import_to_module_audio(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_audio.update(selected=numtab_item)

def import_to_module_video(prompt, negative_prompt, numtab, numtab_item):
    return prompt, negative_prompt, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)

def import_to_module_video_prompt_only(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item) 

def import_text_to_module_image(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def import_text_to_module_video(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)

## fonctions Exports Inputs + Outputs
def both_text_to_module_image (content, prompt, numtab, numtab_item):
    return content, prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) 

def both_text_to_module_inpaint_image (content, prompt, numtab, numtab_item):
    return content, content, prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def both_to_module(prompt, negative_prompt, content, index, numtab, numtab_item):
    index = int(index)
    return prompt, negative_prompt, content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def both_to_module_prompt_only(prompt, content, index, numtab, numtab_item):
    index = int(index)
    return prompt, content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def both_to_module_inpaint(prompt, negative_prompt, content, index, numtab, numtab_item):
    index = int(index)
    return prompt, negative_prompt, content[index], content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def both_to_module_inpaint_prompt_only(prompt, content, index, numtab, numtab_item):
    index = int(index)
    return prompt, content[index], content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def get_select_index(evt: gr.SelectData) :
    return evt.index

## Functions specific to llamacpp
def show_download_llamacpp():
    return btn_download_file_llamacpp.update(visible=False), download_file_llamacpp.update(visible=True)

def hide_download_llamacpp():
    return btn_download_file_llamacpp.update(visible=True), download_file_llamacpp.update(visible=False)

def change_model_type_llamacpp(model_llamacpp):
    model_llamacpp = model_cleaner_llamacpp(model_llamacpp)
    try:
        test_model = model_list_llamacpp[model_llamacpp]
    except KeyError as ke:
        test_model = None
    if (test_model != None):
        return prompt_template_llamacpp.update(value=model_list_llamacpp[model_llamacpp][1]), system_template_llamacpp.update(value=model_list_llamacpp[model_llamacpp][2]), quantization_llamacpp.update(value="")
    else:
        return prompt_template_llamacpp.update(value="{prompt}"), system_template_llamacpp.update(value=""), quantization_llamacpp.update(value="")

def change_prompt_template_llamacpp(prompt_template):
    return prompt_template_llamacpp.update(value=prompt_template_list_llamacpp[prompt_template][0]), system_template_llamacpp.update(value=prompt_template_list_llamacpp[prompt_template][1])

## Functions specific to llava
def show_download_llava():
    return btn_download_file_llava.update(visible=False), download_file_llava.update(visible=True)

def hide_download_llava():
    return btn_download_file_llava.update(visible=True), download_file_llava.update(visible=False)

# def change_model_type_llava(model_llava):
#     return prompt_template_llava.update(value=model_list_llava[model_llava][1])

def change_model_type_llava(model_llava):
    try:
        test_model = model_list_llava[model_llava]
    except KeyError as ke:
        test_model = None
    if (test_model != None):
        return prompt_template_llava.update(value=model_list_llava[model_llava][2]), system_template_llava.update(value=model_list_llava[model_llava][3])
    else:
        return prompt_template_llava.update(value="{prompt}"), system_template_llava.update(value="")

## Functions specific to img2txt_git

## Functions specific to whisper
def change_source_type_whisper(source_type_whisper):
    if source_type_whisper == "audio" :
        return source_audio_whisper.update(source="upload")
    elif source_type_whisper == "micro" :
        return source_audio_whisper.update(source="microphone")

def change_output_type_whisper(output_type_whisper):
    if output_type_whisper == "transcribe" :
        return output_language_whisper.update(visible=False)
    elif output_type_whisper == "translate" :
        return output_language_whisper.update(visible=True)

def stop_recording_whisper(source_audio_whisper):
    return source_audio_whisper.update(source="upload"), source_audio_whisper

## Functions specific to nllb

## Functions specific to txt2prompt
def change_output_type_txt2prompt(output_type_txt2prompt):
    if output_type_txt2prompt == "ChatGPT":
        return model_txt2prompt.update(value=model_list_txt2prompt[0]), max_tokens_txt2prompt.update(value=128)
    elif output_type_txt2prompt == "SD":
        return model_txt2prompt.update(value=model_list_txt2prompt[1]), max_tokens_txt2prompt.update(value=70)

## Functions specific to Stable Diffusion 
def zip_download_file_txt2img_sd(content):
    savename = zipper(content)
    return savename, download_file_txt2img_sd.update(visible=True)

def hide_download_file_txt2img_sd():
    return download_file_txt2img_sd.update(visible=False)

def change_ays_txt2img_sd(use_ays):
    if use_ays:
        return num_inference_step_txt2img_sd.update(interactive=False), sampler_txt2img_sd.update(interactive=False)
    else:
        return num_inference_step_txt2img_sd.update(interactive=True), sampler_txt2img_sd.update(interactive=True)

def change_model_type_txt2img_sd(model_txt2img_sd):
    model_txt2img_sd = model_cleaner_sd(model_txt2img_sd)
    if (model_txt2img_sd == "stabilityai/sdxl-turbo"):
        return sampler_txt2img_sd.update(value="Euler a"), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=1), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=False)
    elif (model_txt2img_sd == "thibaud/sdxl_dpo_turbo"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=2), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=False)
    elif ("IDKIRO/SDXS-512" in model_txt2img_sd.upper()):
        return sampler_txt2img_sd.update(value="Euler a"), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=1), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "stabilityai/sd-turbo"):
        return sampler_txt2img_sd.update(value="Euler a"), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=1), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=False)
    elif ("ETRI-VILAB/KOALA-" in model_txt2img_sd.upper()):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=3.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "GraydientPlatformAPI/lustify-lightning"):
        return sampler_txt2img_sd.update(value="DPM++ SDE Karras"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=1.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "SG161222/RealVisXL_V5.0_Lightning"):
        return sampler_txt2img_sd.update(value="DPM++ SDE Karras"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=5), guidance_scale_txt2img_sd.update(value=1.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "Chan-Y/Stable-Flash-Lightning"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif ("LIGHTNING" in model_txt2img_sd.upper()):
        return sampler_txt2img_sd.update(value="DPM++ SDE Karras"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=1.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "sd-community/sdxl-flash") or (model_txt2img_sd == "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl"):
        return sampler_txt2img_sd.update(value="DPM++ SDE"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=3.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "RunDiffusion/Juggernaut-X-Hyper"):
        return sampler_txt2img_sd.update(value="DPM++ SDE Karras"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=1.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "Corcelio/mobius"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=3.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "mann-e/Mann-E_Dreams") or (model_txt2img_sd == "mann-e/Mann-E_Art"):
        return sampler_txt2img_sd.update(value="DPM++ SDE Karras"), width_txt2img_sd.update(value=768), height_txt2img_sd.update(value=768), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=3.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "John6666/jib-mix-realistic-xl-v15-maximus-sdxl"):
        return sampler_txt2img_sd.update(value="DPM++ SDE"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=2.2), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "segmind/SSD-1B"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "segmind/Segmind-Vega"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=9.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "playgroundai/playground-v2-1024px-aesthetic"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=3.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "playgroundai/playground-v2.5-1024px-aesthetic"):
        return sampler_txt2img_sd.update(value="EDM DPM++ 2M"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=15), guidance_scale_txt2img_sd.update(value=3.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "playgroundai/playground-v2-512px-base"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=3.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif is_sdxl(model_txt2img_sd):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "ariG23498/sd-3.5-merged"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=1.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "adamo1139/stable-diffusion-3.5-large-turbo-ungated"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "aipicasso/emi-3"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=20), guidance_scale_txt2img_sd.update(value=4.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "tensorart/stable-diffusion-3.5-medium-turbo"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=8), guidance_scale_txt2img_sd.update(value=1.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=True)
    elif is_sd35m(model_txt2img_sd):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=20), guidance_scale_txt2img_sd.update(value=4.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=True)
    elif is_sd3(model_txt2img_sd):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=20), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "black-forest-labs/FLUX.1-schnell") or (model_txt2img_sd == "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16") or (model_txt2img_sd == "mikeyandfriends/PixelWave_FLUX.1-schnell_04") or (model_txt2img_sd == "minpeter/FLUX-Hyperscale-fused-fast") or (model_txt2img_sd == "city96/FLUX.1-schnell-gguf"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=False)
    elif (model_txt2img_sd == "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=2), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=False)
    elif (model_txt2img_sd == "sayakpaul/FLUX.1-merged") or (model_txt2img_sd == "shuttleai/shuttle-jaguar"):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=3.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=False)
    elif is_flux(model_txt2img_sd):
        return sampler_txt2img_sd.update(value="Flow Match Euler"), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=20), guidance_scale_txt2img_sd.update(value=3.5), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=False), negative_prompt_txt2img_sd.update(interactive=False)
    else:
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value="", interactive=True), negative_prompt_txt2img_sd.update(interactive=True)

def change_model_type_txt2img_sd_alternate2(model_txt2img_sd):
    if is_noloras(model_txt2img_sd):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model2_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd, True).keys()), value="", interactive=lora_interaction)

def change_model_type_txt2img_sd_alternate3(model_txt2img_sd):
    if is_noloras(model_txt2img_sd):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model3_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd, True).keys()), value="", interactive=lora_interaction)

def change_model_type_txt2img_sd_alternate4(model_txt2img_sd):
    if is_noloras(model_txt2img_sd):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model4_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd, True).keys()), value="", interactive=lora_interaction)

def change_model_type_txt2img_sd_alternate5(model_txt2img_sd):
    if is_noloras(model_txt2img_sd):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model5_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd, True).keys()), value="", interactive=lora_interaction)

biniou_internal_previous_model_txt2img_sd = ""
biniou_internal_previous_steps_txt2img_sd = ""
biniou_internal_previous_cfg_txt2img_sd = ""
biniou_internal_previous_trigger_txt2img_sd = ""
biniou_internal_previous_sampler_txt2img_sd = ""
def change_lora_model_txt2img_sd(model, lora_model, prompt, steps, cfg_scale, sampler):
    global biniou_internal_previous_model_txt2img_sd
    global biniou_internal_previous_steps_txt2img_sd
    global biniou_internal_previous_cfg_txt2img_sd
    global biniou_internal_previous_trigger_txt2img_sd
    global biniou_internal_previous_sampler_txt2img_sd
    lora_model = model_cleaner_lora(lora_model)
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_sd = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_sd = prompt
    else:
        lora_prompt_txt2img_sd = prompt

    if (biniou_internal_previous_trigger_txt2img_sd == ""):
        biniou_internal_previous_trigger_txt2img_sd = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger_txt2img_sd+ ", "
        lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_trigger, "")
        biniou_internal_previous_trigger_txt2img_sd = lora_keyword

    lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    if is_fast_lora(lora_model):
        biniou_internal_previous_model_txt2img_sd = model
        biniou_internal_previous_steps_txt2img_sd = steps
        biniou_internal_previous_cfg_txt2img_sd = cfg_scale
        biniou_internal_previous_sampler_txt2img_sd = sampler
        if (lora_model == "ByteDance/SDXL-Lightning") or (lora_model == "GraydientPlatformAPI/lightning-faster-lora"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="LCM")
        elif ((lora_model == "ByteDance/Hyper-SD") or ("H1T/TCD-SD" in lora_model.upper())) and not is_sd3(model) and not is_flux(model):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=2), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="TCD")
        elif (lora_model == "openskyml/lcm-lora-sdxl-turbo"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="LCM")
        elif (lora_model == "tianweiy/DMD2"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="LCM")
        elif (lora_model == "wangfuyun/PCM_Weights"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=2), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="LCM")
        elif (lora_model == "jasperai/flash-sdxl"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=2), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="LCM")
        elif (lora_model == "sd-community/sdxl-flash-lora"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=3.0), sampler_txt2img_sd.update(value="DPM++ SDE")
        elif (lora_model == "mann-e/Mann-E_Turbo"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=6), guidance_scale_txt2img_sd.update(value=3.0), sampler_txt2img_sd.update(value="DPM++ SDE Karras")
        elif (lora_model == "alimama-creative/slam-lora-sdxl"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=1.0), sampler_txt2img_sd.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd3"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "ByteDance/Hyper-SD") and (is_sd3(model)):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=3.0), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "sunhaha123/stable-diffusion-3.5-medium-turbo"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=8), guidance_scale_txt2img_sd.update(value=4.5), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "ByteDance/Hyper-SD") and (is_flux(model)):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=8), guidance_scale_txt2img_sd.update(value=3.5), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=3.5), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "RED-AIGC/TDD") and (is_flux(model)):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=2.0), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "alimama-creative/FLUX.1-Turbo-Alpha"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=8), guidance_scale_txt2img_sd.update(value=3.5), sampler_txt2img_sd.update(value="Flow Match Euler")
        elif (lora_model == "ostris/fluxdev2schnell-lora"):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=4), guidance_scale_txt2img_sd.update(value=0.0), sampler_txt2img_sd.update(value="Flow Match Euler")
    else:
        if ((biniou_internal_previous_model_txt2img_sd == "") and (biniou_internal_previous_steps_txt2img_sd == "") and (biniou_internal_previous_cfg_txt2img_sd == "") and (biniou_internal_previous_sampler_txt2img_sd == "")):
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(), guidance_scale_txt2img_sd.update(), sampler_txt2img_sd.update()
        elif (biniou_internal_previous_model_txt2img_sd != model):
            biniou_internal_previous_model_txt2img_sd = ""
            biniou_internal_previous_steps_txt2img_sd = ""
            biniou_internal_previous_cfg_txt2img_sd = ""
            biniou_internal_previous_sampler_txt2img_sd = ""
            return prompt_txt2img_sd.update(), num_inference_step_txt2img_sd.update(), guidance_scale_txt2img_sd.update(), sampler_txt2img_sd.update()
        else:
            var_steps = int(biniou_internal_previous_steps_txt2img_sd)
            var_cfg_scale = float(biniou_internal_previous_cfg_txt2img_sd)
            var_sampler = str(biniou_internal_previous_sampler_txt2img_sd)
            biniou_internal_previous_model_txt2img_sd = ""
            biniou_internal_previous_steps_txt2img_sd = ""
            biniou_internal_previous_cfg_txt2img_sd = ""
            biniou_internal_previous_sampler_txt2img_sd = ""
            return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd), num_inference_step_txt2img_sd.update(value=var_steps), guidance_scale_txt2img_sd.update(value=var_cfg_scale), sampler_txt2img_sd.update(value=var_sampler)

biniou_internal_previous_trigger2_txt2img_sd = ""
def change_lora_model2_txt2img_sd(model, lora_model, prompt):
    global biniou_internal_previous_trigger2_txt2img_sd
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_sd = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_sd = prompt
    else:
        lora_prompt_txt2img_sd = prompt

    if (biniou_internal_previous_trigger2_txt2img_sd == ""):
        biniou_internal_previous_trigger2_txt2img_sd = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger2_txt2img_sd+ ", "
        lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_trigger, "")
        biniou_internal_previous_trigger2_txt2img_sd = lora_keyword

    lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd)

biniou_internal_previous_trigger3_txt2img_sd = ""
def change_lora_model3_txt2img_sd(model, lora_model, prompt):
    global biniou_internal_previous_trigger3_txt2img_sd
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_sd = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_sd = prompt
    else:
        lora_prompt_txt2img_sd = prompt

    if (biniou_internal_previous_trigger3_txt2img_sd == ""):
        biniou_internal_previous_trigger3_txt2img_sd = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger3_txt2img_sd+ ", "
        lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_trigger, "")
        biniou_internal_previous_trigger3_txt2img_sd = lora_keyword

    lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd)

biniou_internal_previous_trigger4_txt2img_sd = ""
def change_lora_model4_txt2img_sd(model, lora_model, prompt):
    global biniou_internal_previous_trigger4_txt2img_sd
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_sd = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_sd = prompt
    else:
        lora_prompt_txt2img_sd = prompt

    if (biniou_internal_previous_trigger4_txt2img_sd == ""):
        biniou_internal_previous_trigger4_txt2img_sd = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger4_txt2img_sd+ ", "
        lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_trigger, "")
        biniou_internal_previous_trigger4_txt2img_sd = lora_keyword

    lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd)

biniou_internal_previous_trigger5_txt2img_sd = ""
def change_lora_model5_txt2img_sd(model, lora_model, prompt):
    global biniou_internal_previous_trigger5_txt2img_sd
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_sd = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_sd = prompt
    else:
        lora_prompt_txt2img_sd = prompt

    if (biniou_internal_previous_trigger5_txt2img_sd == ""):
        biniou_internal_previous_trigger5_txt2img_sd = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger5_txt2img_sd+ ", "
        lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_trigger, "")
        biniou_internal_previous_trigger5_txt2img_sd = lora_keyword

    lora_prompt_txt2img_sd = lora_prompt_txt2img_sd.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_sd.update(value=lora_prompt_txt2img_sd)


# def update_preview_txt2img_sd(preview):
#     return out_txt2img_sd.update(preview)

def change_txtinv_txt2img_sd(model, txtinv, prompt, negative_prompt):
    if txtinv != "":
        txtinv_keyword = txtinv_list(model)[txtinv][1]
        if txtinv_keyword != "" and txtinv_keyword != "EasyNegative":
            txtinv_prompt_txt2img_sd = txtinv_keyword+ ", "+ prompt
            txtinv_negative_prompt_txt2img_sd = negative_prompt
        elif txtinv_keyword != "" and txtinv_keyword == "EasyNegative":
            txtinv_prompt_txt2img_sd = prompt
            txtinv_negative_prompt_txt2img_sd = txtinv_keyword+ ", "+ negative_prompt
        else:
            txtinv_prompt_txt2img_sd = prompt
            txtinv_negative_prompt_txt2img_sd = negative_prompt
    else:
        txtinv_prompt_txt2img_sd = prompt
        txtinv_negative_prompt_txt2img_sd = negative_prompt
    return prompt_txt2img_sd.update(value=txtinv_prompt_txt2img_sd), negative_prompt_txt2img_sd.update(value=txtinv_negative_prompt_txt2img_sd)

## Functions specific to Kandinsky 
def zip_download_file_txt2img_kd(content):
    savename = zipper(content)
    return savename, download_file_txt2img_kd.update(visible=True)

def hide_download_file_txt2img_kd():
    return download_file_txt2img_kd.update(visible=False)

def change_model_type_txt2img_kd(model_txt2img_kd):
    if (model_txt2img_kd == "kandinsky-community/kandinsky-3"):
        return width_txt2img_kd.update(value=biniou_global_sdxl_width), height_txt2img_kd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_kd.update(value=15), sampler_txt2img_kd.update(value=list(SCHEDULER_MAPPING.keys())[1])
    else:
        return width_txt2img_kd.update(value=biniou_global_sd15_width), height_txt2img_kd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_kd.update(value=25), sampler_txt2img_kd.update(value=list(SCHEDULER_MAPPING.keys())[5])

## Functions specific to LCM
def zip_download_file_txt2img_lcm(content):
    savename = zipper(content)
    return savename, download_file_txt2img_lcm.update(visible=True)

def hide_download_file_txt2img_lcm():
    return download_file_txt2img_lcm.update(visible=False)

def update_preview_txt2img_lcm(preview):
    return out_txt2img_lcm.update(preview)

def change_model_type_txt2img_lcm(model_txt2img_lcm):
    if (model_txt2img_lcm == "latent-consistency/lcm-ssd-1b"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=False), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "latent-consistency/lcm-sdxl"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=8.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "latent-consistency/lcm-lora-sdxl"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "latent-consistency/lcm-lora-sdv1-5"):
        return width_txt2img_lcm.update(value=biniou_global_sd15_width), height_txt2img_lcm.update(value=biniou_global_sd15_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "segmind/Segmind-VegaRT"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=False), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    else:
        return width_txt2img_lcm.update(value=biniou_global_sd15_width), height_txt2img_lcm.update(value=biniou_global_sd15_height), guidance_scale_txt2img_lcm.update(value=8.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")

def change_model_type_txt2img_lcm_alternate2(model_txt2img_lcm):
    if is_noloras(model_txt2img_lcm):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model2_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm, True).keys()), value="", interactive=lora_interaction)

def change_model_type_txt2img_lcm_alternate3(model_txt2img_lcm):
    if is_noloras(model_txt2img_lcm):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model3_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm, True).keys()), value="", interactive=lora_interaction)

def change_model_type_txt2img_lcm_alternate4(model_txt2img_lcm):
    if is_noloras(model_txt2img_lcm):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model4_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm, True).keys()), value="", interactive=lora_interaction)

def change_model_type_txt2img_lcm_alternate5(model_txt2img_lcm):
    if is_noloras(model_txt2img_lcm):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model5_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm, True).keys()), value="", interactive=lora_interaction)

biniou_internal_previous_model_txt2img_lcm = ""
biniou_internal_previous_steps_txt2img_lcm = ""
biniou_internal_previous_cfg_txt2img_lcm = ""
biniou_internal_previous_trigger_txt2img_lcm = ""
biniou_internal_previous_sampler_txt2img_lcm = ""
def change_lora_model_txt2img_lcm(model, lora_model, prompt, steps, cfg_scale, sampler):
    global biniou_internal_previous_model_txt2img_lcm
    global biniou_internal_previous_steps_txt2img_lcm
    global biniou_internal_previous_cfg_txt2img_lcm
    global biniou_internal_previous_trigger_txt2img_lcm
    global biniou_internal_previous_sampler_txt2img_lcm
    lora_model = model_cleaner_lora(lora_model)
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_lcm = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_lcm = prompt
    else:
        lora_prompt_txt2img_lcm = prompt

    if (biniou_internal_previous_trigger_txt2img_lcm == ""):
        biniou_internal_previous_trigger_txt2img_lcm = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger_txt2img_lcm+ ", "
        lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_trigger, "")
        biniou_internal_previous_trigger_txt2img_lcm = lora_keyword

    lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    if is_fast_lora(lora_model):
        biniou_internal_previous_model_txt2img_lcm = model
        biniou_internal_previous_steps_txt2img_lcm = steps
        biniou_internal_previous_cfg_txt2img_lcm = cfg_scale
        biniou_internal_previous_sampler_txt2img_lcm = sampler
        if (lora_model == "ByteDance/SDXL-Lightning") or (lora_model == "GraydientPlatformAPI/lightning-faster-lora"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=4), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="LCM")
        elif (lora_model == "ByteDance/Hyper-SD") or ("H1T/TCD-SD" in lora_model.upper()):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=2), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="TCD")
        elif (lora_model == "openskyml/lcm-lora-sdxl-turbo"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=4), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="LCM")
        elif (lora_model == "tianweiy/DMD2"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=4), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="LCM")
        elif (lora_model == "wangfuyun/PCM_Weights"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=2), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="LCM")
        elif (lora_model == "jasperai/flash-sdxl"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=4), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=2), guidance_scale_txt2img_lcm.update(value=0.0), sampler_txt2img_lcm.update(value="LCM")
        elif (lora_model == "sd-community/sdxl-flash-lora"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=6), guidance_scale_txt2img_lcm.update(value=3.0), sampler_txt2img_lcm.update(value="DPM++ SDE")
        elif (lora_model == "mann-e/Mann-E_Turbo"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=6), guidance_scale_txt2img_lcm.update(value=3.0), sampler_txt2img_lcm.update(value="DPM++ SDE Karras")
        elif (lora_model == "alimama-creative/slam-lora-sdxl"):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=4), guidance_scale_txt2img_lcm.update(value=1.0), sampler_txt2img_lcm.update(value="LCM")
    else:
        if ((biniou_internal_previous_model_txt2img_lcm == "") and (biniou_internal_previous_steps_txt2img_lcm == "") and (biniou_internal_previous_cfg_txt2img_lcm == "") and (biniou_internal_previous_sampler_txt2img_lcm == "")):
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(), guidance_scale_txt2img_lcm.update(), sampler_txt2img_lcm.update()
        elif (biniou_internal_previous_model_txt2img_lcm != model):
            biniou_internal_previous_model_txt2img_lcm = ""
            biniou_internal_previous_steps_txt2img_lcm = ""
            biniou_internal_previous_cfg_txt2img_lcm = ""
            biniou_internal_previous_sampler_txt2img_lcm = ""
            return prompt_txt2img_lcm.update(), num_inference_step_txt2img_lcm.update(), guidance_scale_txt2img_lcm.update(), sampler_txt2img_lcm.update()
        else:
            var_steps = int(biniou_internal_previous_steps_txt2img_lcm)
            var_cfg_scale = float(biniou_internal_previous_cfg_txt2img_lcm)
            var_sampler = str(biniou_internal_previous_sampler_txt2img_lcm)
            biniou_internal_previous_model_txt2img_lcm = ""
            biniou_internal_previous_steps_txt2img_lcm = ""
            biniou_internal_previous_cfg_txt2img_lcm = ""
            biniou_internal_previous_sampler_txt2img_lcm = ""
            return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm), num_inference_step_txt2img_lcm.update(value=var_steps), guidance_scale_txt2img_lcm.update(value=var_cfg_scale), sampler_txt2img_lcm.update(value=var_sampler)

biniou_internal_previous_trigger2_txt2img_lcm = ""
def change_lora_model2_txt2img_lcm(model, lora_model, prompt):
    global biniou_internal_previous_trigger2_txt2img_lcm
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_lcm = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_lcm = prompt
    else:
        lora_prompt_txt2img_lcm = prompt

    if (biniou_internal_previous_trigger2_txt2img_lcm == ""):
        biniou_internal_previous_trigger2_txt2img_lcm = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger2_txt2img_lcm+ ", "
        lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_trigger, "")
        biniou_internal_previous_trigger2_txt2img_lcm = lora_keyword

    lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm)

biniou_internal_previous_trigger3_txt2img_lcm = ""
def change_lora_model3_txt2img_lcm(model, lora_model, prompt):
    global biniou_internal_previous_trigger3_txt2img_lcm
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_lcm = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_lcm = prompt
    else:
        lora_prompt_txt2img_lcm = prompt

    if (biniou_internal_previous_trigger3_txt2img_lcm == ""):
        biniou_internal_previous_trigger3_txt2img_lcm = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger3_txt2img_lcm+ ", "
        lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_trigger, "")
        biniou_internal_previous_trigger3_txt2img_lcm = lora_keyword

    lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm)

biniou_internal_previous_trigger4_txt2img_lcm = ""
def change_lora_model4_txt2img_lcm(model, lora_model, prompt):
    global biniou_internal_previous_trigger4_txt2img_lcm
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_lcm = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_lcm = prompt
    else:
        lora_prompt_txt2img_lcm = prompt

    if (biniou_internal_previous_trigger4_txt2img_lcm == ""):
        biniou_internal_previous_trigger4_txt2img_lcm = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger4_txt2img_lcm+ ", "
        lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_trigger, "")
        biniou_internal_previous_trigger4_txt2img_lcm = lora_keyword

    lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm)

biniou_internal_previous_trigger5_txt2img_lcm = ""
def change_lora_model5_txt2img_lcm(model, lora_model, prompt):
    global biniou_internal_previous_trigger5_txt2img_lcm
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_txt2img_lcm = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_lcm = prompt
    else:
        lora_prompt_txt2img_lcm = prompt

    if (biniou_internal_previous_trigger5_txt2img_lcm == ""):
        biniou_internal_previous_trigger5_txt2img_lcm = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger5_txt2img_lcm+ ", "
        lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_trigger, "")
        biniou_internal_previous_trigger5_txt2img_lcm = lora_keyword

    lora_prompt_txt2img_lcm = lora_prompt_txt2img_lcm.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_txt2img_lcm.update(value=lora_prompt_txt2img_lcm)

def change_txtinv_txt2img_lcm(model, txtinv, prompt):
    if txtinv != "":
        txtinv_keyword = txtinv_list(model)[txtinv][1]
        if txtinv_keyword != "" and txtinv_keyword != "EasyNegative":
            txtinv_prompt_txt2img_lcm = txtinv_keyword+ ", "+ prompt
        else:
            txtinv_prompt_txt2img_lcm = prompt
    else:
        txtinv_prompt_txt2img_lcm = prompt
    return prompt_txt2img_lcm.update(value=txtinv_prompt_txt2img_lcm)

## Functions specific to Midjourney mini
def zip_download_file_txt2img_mjm(content):
    savename = zipper(content)
    return savename, download_file_txt2img_mjm.update(visible=True)

def hide_download_file_txt2img_mjm():
    return download_file_txt2img_mjm.update(visible=False)

def change_ays_txt2img_mjm(use_ays):
    if use_ays:
        return num_inference_step_txt2img_mjm.update(interactive=False), sampler_txt2img_mjm.update(interactive=False)
    else:
        return num_inference_step_txt2img_mjm.update(interactive=True), sampler_txt2img_mjm.update(interactive=True)

## Functions specific to PixArt-Alpha
def zip_download_file_txt2img_paa(content):
    savename = zipper(content)
    return savename, download_file_txt2img_paa.update(visible=True)

def hide_download_file_txt2img_paa():
    return download_file_txt2img_paa.update(visible=False)

def change_model_type_txt2img_paa(model_txt2img_paa):
    if model_txt2img_paa == "PixArt-alpha/PixArt-XL-2-1024-MS":
        return sampler_txt2img_paa.update(value="UniPC", interactive=True), width_txt2img_paa.update(value=biniou_global_sdxl_width), height_txt2img_paa.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_paa.update(value=7.0), num_inference_step_txt2img_paa.update(value=15)
    elif model_txt2img_paa == "Luo-Yihong/yoso_pixart512":
        return sampler_txt2img_paa.update(value="LCM", interactive=False), width_txt2img_paa.update(value=biniou_global_sd15_width), height_txt2img_paa.update(value=biniou_global_sd15_width), guidance_scale_txt2img_paa.update(value=1.0), num_inference_step_txt2img_paa.update(value=1)
    elif model_txt2img_paa == "Luo-Yihong/yoso_pixart1024":
        return sampler_txt2img_paa.update(value="LCM", interactive=False), width_txt2img_paa.update(value=biniou_global_sdxl_width), height_txt2img_paa.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_paa.update(value=1.0), num_inference_step_txt2img_paa.update(value=1)
    elif model_txt2img_paa == "jasperai/flash-pixart":
        return sampler_txt2img_paa.update(value="LCM", interactive=False), width_txt2img_paa.update(value=biniou_global_sdxl_width), height_txt2img_paa.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_paa.update(value=0.0), num_inference_step_txt2img_paa.update(value=4)
    elif model_txt2img_paa == "PixArt-alpha/PixArt-LCM-XL-2-1024-MS":
        return sampler_txt2img_paa.update(value="LCM", interactive=False), width_txt2img_paa.update(value=biniou_global_sdxl_width), height_txt2img_paa.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_paa.update(value=0.0), num_inference_step_txt2img_paa.update(value=4)
    elif model_txt2img_paa == "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" or model_txt2img_paa == "dataautogpt3/PixArt-Sigma-900M":
        return sampler_txt2img_paa.update(value="UniPC", interactive=True), width_txt2img_paa.update(value=biniou_global_sdxl_width), height_txt2img_paa.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_paa.update(value=7.0), num_inference_step_txt2img_paa.update(value=15)
    elif model_txt2img_paa == "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS":
        return sampler_txt2img_paa.update(value="UniPC", interactive=True), width_txt2img_paa.update(value=2048), height_txt2img_paa.update(value=2048), guidance_scale_txt2img_paa.update(value=7.0), num_inference_step_txt2img_paa.update(value=15)
    else:
        return sampler_txt2img_paa.update(value="UniPC", interactive=True), width_txt2img_paa.update(value=biniou_global_sd15_width), height_txt2img_paa.update(value=biniou_global_sd15_height), guidance_scale_txt2img_paa.update(value=7.0), num_inference_step_txt2img_paa.update(value=15)

## Functions specific to img2img
def zip_download_file_img2img(content):
    savename = zipper(content)
    return savename, download_file_img2img.update(visible=True)

def hide_download_file_img2img():
    return download_file_img2img.update(visible=False)

def change_source_type_img2img(source_type_img2img):
    if source_type_img2img == "image" :
        return {"source": "upload", "tool": "", "width" : "", "value": None, "__type__": "update"}
    elif source_type_img2img == "sketch" :
        return {"source": "canvas", "tool": "color-sketch", "width" : 400, "height" : 400,  "__type__": "update"}

def change_ays_img2img(use_ays):
    if use_ays:
        return num_inference_step_img2img.update(interactive=False), sampler_img2img.update(interactive=False)
    else:
        return num_inference_step_img2img.update(interactive=True), sampler_img2img.update(interactive=True)

def change_model_type_img2img(model_img2img):
    model_img2img = model_cleaner_sd(model_img2img)
    if (model_img2img == "stabilityai/sdxl-turbo"):
        return sampler_img2img.update(value="Euler a"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=False)
    elif (model_img2img == "thibaud/sdxl_dpo_turbo"):
        return sampler_img2img.update(value="UniPC"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=False)
    elif ("IDKIRO/SDXS-512" in model_img2img.upper()):
        return sampler_img2img.update(value="Euler a"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "stabilityai/sd-turbo"):
        return sampler_img2img.update(value="Euler a"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=False)
    elif ("ETRI-VILAB/KOALA-" in model_img2img.upper()):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=3.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "GraydientPlatformAPI/lustify-lightning"):
        return sampler_img2img.update(value="DPM++ SDE Karras"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=1.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "SG161222/RealVisXL_V5.0_Lightning"):
        return sampler_img2img.update(value="DPM++ SDE Karras"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=5), guidance_scale_img2img.update(value=1.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "Chan-Y/Stable-Flash-Lightning"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif ("LIGHTNING" in model_img2img.upper()):
        return sampler_img2img.update(value="DPM++ SDE Karras"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=1.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "sd-community/sdxl-flash") or (model_img2img == "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl"):
        return sampler_img2img.update(value="DPM++ SDE"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=3.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "RunDiffusion/Juggernaut-X-Hyper"):
        return sampler_img2img.update(value="DPM++ SDE Karras"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=1.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "Corcelio/mobius"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=3.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "mann-e/Mann-E_Dreams") or (model_img2img == "mann-e/Mann-E_Art"):
        return sampler_img2img.update(value="DPM++ SDE Karras"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=3.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "John6666/jib-mix-realistic-xl-v15-maximus-sdxl"):
        return sampler_img2img.update(value="DPM++ SDE"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=2.2), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "segmind/SSD-1B"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "segmind/Segmind-Vega"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=9.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "playgroundai/playground-v2-1024px-aesthetic"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=3.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "playgroundai/playground-v2.5-1024px-aesthetic"):
        return sampler_img2img.update(value="EDM DPM++ 2M"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=15), guidance_scale_img2img.update(value=3.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "playgroundai/playground-v2-512px-base"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=3.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "ariG23498/sd-3.5-merged"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=1.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "adamo1139/stable-diffusion-3.5-large-turbo-ungated"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "aipicasso/emi-3"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=4.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "tensorart/stable-diffusion-3.5-medium-turbo"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=8), guidance_scale_img2img.update(value=1.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=True)
    elif is_sd35m(model_img2img):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=4.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=True)
    elif is_sd3(model_img2img):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "black-forest-labs/FLUX.1-schnell") or (model_img2img == "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16") or (model_img2img == "mikeyandfriends/PixelWave_FLUX.1-schnell_04") or (model_img2img == "minpeter/FLUX-Hyperscale-fused-fast"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=False)
    elif (model_img2img == "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=False)
    elif (model_img2img == "sayakpaul/FLUX.1-merged") or (model_img2img == "shuttleai/shuttle-jaguar"):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=3.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=False)
    elif is_flux(model_img2img):
        return sampler_img2img.update(value="Flow Match Euler"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=3.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=False), negative_prompt_img2img.update(interactive=False)
    elif is_sdxl(model_img2img):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)
    else:
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value="", interactive=True), negative_prompt_img2img.update(interactive=True)

def change_model_type_img2img_alternate2(model_img2img):
    if is_noloras(model_img2img):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model2_img2img.update(choices=list(lora_model_list(model_img2img, True).keys()), value="", interactive=lora_interaction)

def change_model_type_img2img_alternate3(model_img2img):
    if is_noloras(model_img2img):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model3_img2img.update(choices=list(lora_model_list(model_img2img, True).keys()), value="", interactive=lora_interaction)

def change_model_type_img2img_alternate4(model_img2img):
    if is_noloras(model_img2img):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model4_img2img.update(choices=list(lora_model_list(model_img2img, True).keys()), value="", interactive=lora_interaction)

def change_model_type_img2img_alternate5(model_img2img):
    if is_noloras(model_img2img):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model5_img2img.update(choices=list(lora_model_list(model_img2img, True).keys()), value="", interactive=lora_interaction)

biniou_internal_previous_model_img2img = ""
biniou_internal_previous_steps_img2img = ""
biniou_internal_previous_cfg_img2img = ""
biniou_internal_previous_trigger_img2img = ""
biniou_internal_previous_sampler_img2img = ""
def change_lora_model_img2img(model, lora_model, prompt, steps, cfg_scale, sampler):
    global biniou_internal_previous_model_img2img
    global biniou_internal_previous_steps_img2img
    global biniou_internal_previous_cfg_img2img
    global biniou_internal_previous_trigger_img2img
    global biniou_internal_previous_sampler_img2img
    lora_model = model_cleaner_lora(lora_model)
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img = prompt
    else:
        lora_prompt_img2img = prompt

    if (biniou_internal_previous_trigger_img2img == ""):
        biniou_internal_previous_trigger_img2img = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger_img2img+ ", "
        lora_prompt_img2img = lora_prompt_img2img.replace(lora_trigger, "")
        biniou_internal_previous_trigger_img2img = lora_keyword

    lora_prompt_img2img = lora_prompt_img2img.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    if is_fast_lora(lora_model):
        biniou_internal_previous_model_img2img = model
        biniou_internal_previous_steps_img2img = steps
        biniou_internal_previous_cfg_img2img = cfg_scale
        biniou_internal_previous_sampler_img2img = sampler
        if (lora_model == "ByteDance/SDXL-Lightning") or (lora_model == "GraydientPlatformAPI/lightning-faster-lora"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="LCM")
        elif ((lora_model == "ByteDance/Hyper-SD") or ("H1T/TCD-SD" in lora_model.upper())) and not is_sd3(model) and not is_flux(model):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="TCD")
        elif (lora_model == "openskyml/lcm-lora-sdxl-turbo"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="LCM")
        elif (lora_model == "tianweiy/DMD2"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="LCM")
        elif (lora_model == "wangfuyun/PCM_Weights"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="LCM")
        elif (lora_model == "jasperai/flash-sdxl"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="LCM")
        elif (lora_model == "sd-community/sdxl-flash-lora"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=3.0), sampler_img2img.update(value="DPM++ SDE")
        elif (lora_model == "mann-e/Mann-E_Turbo"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=6), guidance_scale_img2img.update(value=3.0), sampler_img2img.update(value="DPM++ SDE Karras")
        elif (lora_model == "alimama-creative/slam-lora-sdxl"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=1.0), sampler_img2img.update(value="LCM")
        elif (lora_model == "ByteDance/Hyper-SD") and (is_sd3(model)):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=3.0), sampler_img2img.update(value="Flow Match Euler")
        elif (lora_model == "sunhaha123/stable-diffusion-3.5-medium-turbo"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=8), guidance_scale_img2img.update(value=4.5), sampler_img2img.update(value="Flow Match Euler")
        elif (lora_model == "ByteDance/Hyper-SD") and (is_flux(model)):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=8), guidance_scale_img2img.update(value=3.5), sampler_img2img.update(value="Flow Match Euler")
        elif (lora_model == "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=3.5), sampler_img2img.update(value="Flow Match Euler")
        elif (lora_model == "RED-AIGC/TDD") and (is_flux(model)):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=2.0), sampler_img2img.update(value="Flow Match Euler")
        elif (lora_model == "alimama-creative/FLUX.1-Turbo-Alpha"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=8), guidance_scale_img2img.update(value=3.5), sampler_img2img.update(value="Flow Match Euler")
        elif (lora_model == "ostris/fluxdev2schnell-lora"):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=4), guidance_scale_img2img.update(value=0.0), sampler_img2img.update(value="Flow Match Euler")
    else:
        if ((biniou_internal_previous_model_img2img == "") and (biniou_internal_previous_steps_img2img == "") and (biniou_internal_previous_cfg_img2img == "") and (biniou_internal_previous_sampler_img2img == "")):
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(), guidance_scale_img2img.update(), sampler_img2img.update()
        elif (biniou_internal_previous_model_img2img != model):
            biniou_internal_previous_model_img2img = ""
            biniou_internal_previous_steps_img2img = ""
            biniou_internal_previous_cfg_img2img = ""
            biniou_internal_previous_sampler_img2img = ""
            return prompt_img2img.update(), num_inference_step_img2img.update(), guidance_scale_img2img.update(), sampler_img2img.update()
        else:
            var_steps = int(biniou_internal_previous_steps_img2img)
            var_cfg_scale = float(biniou_internal_previous_cfg_img2img)
            var_sampler = str(biniou_internal_previous_sampler_img2img)
            biniou_internal_previous_model_img2img = ""
            biniou_internal_previous_steps_img2img = ""
            biniou_internal_previous_cfg_img2img = ""
            biniou_internal_previous_sampler_img2img = ""
            return prompt_img2img.update(value=lora_prompt_img2img), num_inference_step_img2img.update(value=var_steps), guidance_scale_img2img.update(value=var_cfg_scale), sampler_img2img.update(value=var_sampler)

biniou_internal_previous_trigger2_img2img = ""
def change_lora_model2_img2img(model, lora_model, prompt):
    global biniou_internal_previous_trigger2_img2img
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img = prompt
    else:
        lora_prompt_img2img = prompt

    if (biniou_internal_previous_trigger2_img2img == ""):
        biniou_internal_previous_trigger2_img2img = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger2_img2img+ ", "
        lora_prompt_img2img = lora_prompt_img2img.replace(lora_trigger, "")
        biniou_internal_previous_trigger2_img2img = lora_keyword

    lora_prompt_img2img = lora_prompt_img2img.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img.update(value=lora_prompt_img2img)

biniou_internal_previous_trigger3_img2img = ""
def change_lora_model3_img2img(model, lora_model, prompt):
    global biniou_internal_previous_trigger3_img2img
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img = prompt
    else:
        lora_prompt_img2img = prompt

    if (biniou_internal_previous_trigger3_img2img == ""):
        biniou_internal_previous_trigger3_img2img = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger3_img2img+ ", "
        lora_prompt_img2img = lora_prompt_img2img.replace(lora_trigger, "")
        biniou_internal_previous_trigger3_img2img = lora_keyword

    lora_prompt_img2img = lora_prompt_img2img.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img.update(value=lora_prompt_img2img)

biniou_internal_previous_trigger4_img2img = ""
def change_lora_model4_img2img(model, lora_model, prompt):
    global biniou_internal_previous_trigger4_img2img
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img = prompt
    else:
        lora_prompt_img2img = prompt

    if (biniou_internal_previous_trigger4_img2img == ""):
        biniou_internal_previous_trigger4_img2img = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger4_img2img+ ", "
        lora_prompt_img2img = lora_prompt_img2img.replace(lora_trigger, "")
        biniou_internal_previous_trigger4_img2img = lora_keyword

    lora_prompt_img2img = lora_prompt_img2img.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img.update(value=lora_prompt_img2img)

biniou_internal_previous_trigger5_img2img = ""
def change_lora_model5_img2img(model, lora_model, prompt):
    global biniou_internal_previous_trigger5_img2img
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img = prompt
    else:
        lora_prompt_img2img = prompt

    if (biniou_internal_previous_trigger5_img2img == ""):
        biniou_internal_previous_trigger5_img2img = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger5_img2img+ ", "
        lora_prompt_img2img = lora_prompt_img2img.replace(lora_trigger, "")
        biniou_internal_previous_trigger5_img2img = lora_keyword

    lora_prompt_img2img = lora_prompt_img2img.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img.update(value=lora_prompt_img2img)

def change_txtinv_img2img(model, txtinv, prompt, negative_prompt):
    if txtinv != "":
        txtinv_keyword = txtinv_list(model)[txtinv][1]
        if txtinv_keyword != "" and txtinv_keyword != "EasyNegative":
            txtinv_prompt_img2img = txtinv_keyword+ ", "+ prompt
            txtinv_negative_prompt_img2img = negative_prompt
        elif txtinv_keyword != "" and txtinv_keyword == "EasyNegative":
            txtinv_prompt_img2img = prompt
            txtinv_negative_prompt_img2img = txtinv_keyword+ ", "+ negative_prompt
        else:
            txtinv_prompt_img2img = prompt
            txtinv_negative_prompt_img2img = negative_prompt
    else:
        txtinv_prompt_img2img = prompt
        txtinv_negative_prompt_img2img = negative_prompt
    return prompt_img2img.update(value=txtinv_prompt_img2img), negative_prompt_img2img.update(value=txtinv_negative_prompt_img2img)

## Functions specific to img2img_ip
def zip_download_file_img2img_ip(content):
    savename = zipper(content)
    return savename, download_file_img2img_ip.update(visible=True)

def hide_download_file_img2img_ip():
    return download_file_img2img_ip.update(visible=False)

def change_ays_img2img_ip(use_ays):
    if use_ays:
        return num_inference_step_img2img_ip.update(interactive=False), sampler_img2img_ip.update(interactive=False)
    else:
        return num_inference_step_img2img_ip.update(interactive=True), sampler_img2img_ip.update(interactive=True)

def change_model_type_img2img_ip(model_img2img_ip, source_type):
    model_img2img_ip = model_cleaner_sd(model_img2img_ip)
    if is_sdxl(model_img2img_ip):
        is_xl_size: bool = True
    else :
        is_xl_size: bool = False

    if (source_type == "composition") and not is_xl_size:
        width_value = biniou_global_sd15_width
        height_value = biniou_global_sd15_height
        interaction = True
    elif (source_type == "composition") and is_xl_size:
        width_value = biniou_global_sdxl_width
        height_value = biniou_global_sdxl_height
        interaction = True
    elif (source_type == "standard") and not is_xl_size:
        width_value = None
        height_value = None
        interaction = False
    elif (source_type == "standard") and is_xl_size:
        width_value = None
        height_value = None
        interaction = False

    if (model_img2img_ip == "stabilityai/sdxl-turbo"):
        return sampler_img2img_ip.update(value="Euler a"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=False), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
#    elif (model_img2img_ip == "thibaud/sdxl_dpo_turbo"):
#        return sampler_img2img_ip.update(value="UniPC"), width_img2img_ip.update(value=biniou_global_sd15_width), height_img2img_ip.update(value=biniou_global_sd15_height), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=False), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "stabilityai/sd-turbo"):
        return sampler_img2img_ip.update(value="Euler a"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=False), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif ("ETRI-VILAB/KOALA-" in model_img2img_ip.upper()):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 3.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "GraydientPlatformAPI/lustify-lightning"):
        return sampler_img2img_ip.update("DPM++ SDE Karras"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=6), guidance_scale_img2img_ip.update(value=1.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "SG161222/RealVisXL_V5.0_Lightning"):
        return sampler_img2img_ip.update("DPM++ SDE Karras"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=5), guidance_scale_img2img_ip.update(value=1.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "Chan-Y/Stable-Flash-Lightning"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif ("LIGHTNING" in model_img2img_ip.upper()):
        return sampler_img2img_ip.update("DPM++ SDE Karras"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=1.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "sd-community/sdxl-flash") or (model_img2img_ip == "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl"):
        return sampler_img2img_ip.update("DPM++ SDE"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=6), guidance_scale_img2img_ip.update(value=3.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "RunDiffusion/Juggernaut-X-Hyper"):
        return sampler_img2img_ip.update("DPM++ SDE Karras"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=6), guidance_scale_img2img_ip.update(value=1.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "Corcelio/mobius"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 3.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "mann-e/Mann-E_Dreams") or (model_img2img_ip == "mann-e/Mann-E_Art"):
        return sampler_img2img_ip.update(value="DPM++ SDE Karras"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=6), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 3.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "John6666/jib-mix-realistic-xl-v15-maximus-sdxl"):
        return sampler_img2img_ip.update(value="DPM++ SDE"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 2.2), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "segmind/SSD-1B"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "segmind/Segmind-Vega"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 9.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "playgroundai/playground-v2-1024px-aesthetic"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 3.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "playgroundai/playground-v2.5-1024px-aesthetic"):
        return sampler_img2img_ip.update(value="EDM DPM++ 2M"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=15), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 3.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "playgroundai/playground-v2-512px-base"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 3.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif is_sd3(model_img2img_ip):
        return sampler_img2img_ip.update("Flow Match Euler"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value="", interactive=False), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    elif (model_img2img_ip == "black-forest-labs/FLUX.1-schnell") or (model_img2img_ip == "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16") or (model_img2img_ip == "mikeyandfriends/PixelWave_FLUX.1-schnell_04") or (model_img2img_ip == "minpeter/FLUX-Hyperscale-fused-fast"):
        return sampler_img2img_ip.update("Flow Match Euler"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=0.0 if source_type == "composition" else 0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value="", interactive=False), negative_prompt_img2img_ip.update(interactive=False), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=False)
    elif (model_img2img_ip == "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers"):
        return sampler_img2img_ip.update("Flow Match Euler"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0 if source_type == "composition" else 0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value="", interactive=False), negative_prompt_img2img_ip.update(interactive=False), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=False)
    elif (model_img2img_ip == "sayakpaul/FLUX.1-merged") or (model_img2img_ip == "shuttleai/shuttle-jaguar"):
        return sampler_img2img_ip.update("Flow Match Euler"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=3.5 if source_type == "composition" else 3.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value="", interactive=False), negative_prompt_img2img_ip.update(interactive=False), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=False)
    elif is_flux(model_img2img_ip):
        return sampler_img2img_ip.update("Flow Match Euler"), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3.5 if source_type == "composition" else 3.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value="", interactive=False), negative_prompt_img2img_ip.update(interactive=False), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=False)
    elif is_sdxl(model_img2img_ip):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(width_value, interactive=interaction), height_img2img_ip.update(height_value, interactive=interaction), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True), source_type_img2img_ip.update(interactive=True), img_img2img_ip.update(visible=True)
    else:
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), \
            width_img2img_ip.update(width_value, interactive=interaction), \
            height_img2img_ip.update(height_value, interactive=interaction), \
            num_inference_step_img2img_ip.update(value=10), \
            guidance_scale_img2img_ip.update(value=3 if source_type == "composition" else 7.5), \
            lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), \
            txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), \
            negative_prompt_img2img_ip.update(interactive=True), \
            source_type_img2img_ip.update(interactive=True), \
            img_img2img_ip.update(visible=True)
def change_model_type_img2img_ip_alternate2(model_img2img_ip):
    if is_noloras(model_img2img_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model2_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip, True).keys()), value="", interactive=lora_interaction)

def change_model_type_img2img_ip_alternate3(model_img2img_ip):
    if is_noloras(model_img2img_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model3_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip, True).keys()), value="", interactive=lora_interaction)

def change_model_type_img2img_ip_alternate4(model_img2img_ip):
    if is_noloras(model_img2img_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model4_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip, True).keys()), value="", interactive=lora_interaction)

def change_model_type_img2img_ip_alternate5(model_img2img_ip):
    if is_noloras(model_img2img_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model5_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip, True).keys()), value="", interactive=lora_interaction)

biniou_internal_previous_model_img2img_ip = ""
biniou_internal_previous_steps_img2img_ip = ""
biniou_internal_previous_cfg_img2img_ip = ""
biniou_internal_previous_trigger_img2img_ip = ""
biniou_internal_previous_sampler_img2img_ip = ""
def change_lora_model_img2img_ip(model, lora_model, prompt, steps, cfg_scale, sampler):
    global biniou_internal_previous_model_img2img_ip
    global biniou_internal_previous_steps_img2img_ip
    global biniou_internal_previous_cfg_img2img_ip
    global biniou_internal_previous_trigger_img2img_ip
    global biniou_internal_previous_sampler_img2img_ip
    lora_model = model_cleaner_lora(lora_model)
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img_ip = prompt
    else:
        lora_prompt_img2img_ip = prompt

    if (biniou_internal_previous_trigger_img2img_ip == ""):
        biniou_internal_previous_trigger_img2img_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger_img2img_ip+ ", "
        lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger_img2img_ip = lora_keyword

    lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    if is_fast_lora(lora_model):
        biniou_internal_previous_model_img2img_ip = model
        biniou_internal_previous_steps_img2img_ip = steps
        biniou_internal_previous_cfg_img2img_ip = cfg_scale
        biniou_internal_previous_sampler_img2img_ip = sampler
        if (lora_model == "ByteDance/SDXL-Lightning") or (lora_model == "GraydientPlatformAPI/lightning-faster-lora"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="LCM")
        elif ((lora_model == "ByteDance/Hyper-SD") or ("H1T/TCD-SD" in lora_model.upper())) and not is_flux(model):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="TCD")
        elif (lora_model == "openskyml/lcm-lora-sdxl-turbo"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="LCM")
        elif (lora_model == "tianweiy/DMD2"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="LCM")
        elif (lora_model == "wangfuyun/PCM_Weights"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="LCM")
        elif (lora_model == "jasperai/flash-sdxl"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="LCM")
        elif (lora_model == "sd-community/sdxl-flash-lora"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=6), guidance_scale_img2img_ip.update(value=3.0), sampler_img2img_ip.update(value="DPM++ SDE")
        elif (lora_model == "mann-e/Mann-E_Turbo"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=6), guidance_scale_img2img_ip.update(value=3.0), sampler_img2img_ip.update(value="DPM++ SDE Karras")
        elif (lora_model == "alimama-creative/slam-lora-sdxl"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=1.0), sampler_img2img_ip.update(value="LCM")
        elif (lora_model == "ByteDance/Hyper-SD") and (is_flux(model)):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=8), guidance_scale_img2img_ip.update(value=3.5), sampler_img2img_ip.update(value="Flow Match Euler")
        elif (lora_model == "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=3.5), sampler_img2img_ip.update(value="Flow Match Euler")
        elif (lora_model == "RED-AIGC/TDD") and (is_flux(model)):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=2.0), sampler_img2img_ip.update(value="Flow Match Euler")
        elif (lora_model == "alimama-creative/FLUX.1-Turbo-Alpha"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=8), guidance_scale_img2img_ip.update(value=3.5), sampler_img2img_ip.update(value="Flow Match Euler")
        elif (lora_model == "ostris/fluxdev2schnell-lora"):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=4), guidance_scale_img2img_ip.update(value=0.0), sampler_img2img_ip.update(value="Flow Match Euler")
    else:
        if ((biniou_internal_previous_model_img2img_ip == "") and (biniou_internal_previous_steps_img2img_ip == "") and (biniou_internal_previous_cfg_img2img_ip == "") and (biniou_internal_previous_sampler_img2img_ip == "")):
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(), guidance_scale_img2img_ip.update(), sampler_img2img_ip.update()
        elif (biniou_internal_previous_model_img2img_ip != model):
            biniou_internal_previous_model_img2img_ip = ""
            biniou_internal_previous_steps_img2img_ip = ""
            biniou_internal_previous_cfg_img2img_ip = ""
            biniou_internal_previous_sampler_img2img_ip = ""
            return prompt_img2img_ip.update(), num_inference_step_img2img_ip.update(), guidance_scale_img2img_ip.update(), sampler_img2img_ip.update()
        else:
            var_steps = int(biniou_internal_previous_steps_img2img_ip)
            var_cfg_scale = float(biniou_internal_previous_cfg_img2img_ip)
            var_sampler = str(biniou_internal_previous_sampler_img2img_ip)
            biniou_internal_previous_model_img2img_ip = ""
            biniou_internal_previous_steps_img2img_ip = ""
            biniou_internal_previous_cfg_img2img_ip = ""
            biniou_internal_previous_sampler_img2img_ip = ""
            return prompt_img2img_ip.update(value=lora_prompt_img2img_ip), num_inference_step_img2img_ip.update(value=var_steps), guidance_scale_img2img_ip.update(value=var_cfg_scale), sampler_img2img_ip.update(value=var_sampler)

biniou_internal_previous_trigger2_img2img_ip = ""
def change_lora_model2_img2img_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger2_img2img_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img_ip = prompt
    else:
        lora_prompt_img2img_ip = prompt

    if (biniou_internal_previous_trigger2_img2img_ip == ""):
        biniou_internal_previous_trigger2_img2img_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger2_img2img_ip+ ", "
        lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger2_img2img_ip = lora_keyword

    lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img_ip.update(value=lora_prompt_img2img_ip)

biniou_internal_previous_trigger3_img2img_ip = ""
def change_lora_model3_img2img_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger3_img2img_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img_ip = prompt
    else:
        lora_prompt_img2img_ip = prompt

    if (biniou_internal_previous_trigger3_img2img_ip == ""):
        biniou_internal_previous_trigger3_img2img_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger3_img2img_ip+ ", "
        lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger3_img2img_ip = lora_keyword

    lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img_ip.update(value=lora_prompt_img2img_ip)

biniou_internal_previous_trigger4_img2img_ip = ""
def change_lora_model4_img2img_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger4_img2img_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img_ip = prompt
    else:
        lora_prompt_img2img_ip = prompt

    if (biniou_internal_previous_trigger4_img2img_ip == ""):
        biniou_internal_previous_trigger4_img2img_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger4_img2img_ip+ ", "
        lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger4_img2img_ip = lora_keyword

    lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img_ip.update(value=lora_prompt_img2img_ip)

biniou_internal_previous_trigger5_img2img_ip = ""
def change_lora_model5_img2img_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger5_img2img_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_img2img_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img_ip = prompt
    else:
        lora_prompt_img2img_ip = prompt

    if (biniou_internal_previous_trigger5_img2img_ip == ""):
        biniou_internal_previous_trigger5_img2img_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger5_img2img_ip+ ", "
        lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger5_img2img_ip = lora_keyword

    lora_prompt_img2img_ip = lora_prompt_img2img_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_img2img_ip.update(value=lora_prompt_img2img_ip)

def change_txtinv_img2img_ip(model, txtinv, prompt, negative_prompt):
    if txtinv != "":
        txtinv_keyword = txtinv_list(model)[txtinv][1]
        if txtinv_keyword != "" and txtinv_keyword != "EasyNegative":
            txtinv_prompt_img2img_ip = txtinv_keyword+ ", "+ prompt
            txtinv_negative_prompt_img2img_ip = negative_prompt
        elif txtinv_keyword != "" and txtinv_keyword == "EasyNegative":
            txtinv_prompt_img2img_ip = prompt
            txtinv_negative_prompt_img2img_ip = txtinv_keyword+ ", "+ negative_prompt
        else:
            txtinv_prompt_img2img_ip = prompt
            txtinv_negative_prompt_img2img_ip = negative_prompt
    else:
        txtinv_prompt_img2img_ip = prompt
        txtinv_negative_prompt_img2img_ip = negative_prompt
    return prompt_img2img_ip.update(value=txtinv_prompt_img2img_ip), negative_prompt_img2img_ip.update(value=txtinv_negative_prompt_img2img_ip)

def change_source_type_img2img_ip(source_type):
    if (source_type == "standard"):
        return img_img2img_ip.update(visible=True, interactive=True), denoising_strength_img2img_ip.update(value=0.6, interactive=True)
    elif (source_type == "composition"):
        return img_img2img_ip.update(value=None, visible=False, interactive=False), denoising_strength_img2img_ip.update(value=1, interactive=False)

## Functions specific to img2var
def zip_download_file_img2var(content):
    savename = zipper(content)
    return savename, download_file_img2var.update(visible=True)

def hide_download_file_img2var():
    return download_file_img2var.update(visible=False)

## Functions specific to pix2pix
def zip_download_file_pix2pix(content):
    savename = zipper(content)
    return savename, download_file_pix2pix.update(visible=True)

def change_model_type_pix2pix(model_pix2pix):
    if model_pix2pix == "diffusers/sdxl-instructpix2pix-768":
        return sampler_pix2pix.update(value=list(SCHEDULER_MAPPING.keys())[0], interactive=True), width_pix2pix.update(), height_pix2pix.update(), guidance_scale_pix2pix.update(value=3.0), image_guidance_scale_pix2pix.update(value=1.5), num_inference_step_pix2pix.update(value=10)
    else:
        return sampler_pix2pix.update(value=list(SCHEDULER_MAPPING.keys())[0], interactive=True), width_pix2pix.update(), height_pix2pix.update(), guidance_scale_pix2pix.update(value=7.0), image_guidance_scale_pix2pix.update(value=1.5), num_inference_step_pix2pix.update(value=10)

def hide_download_file_pix2pix():
    return download_file_pix2pix.update(visible=False)

## Functions specific to magicmix
def zip_download_file_magicmix(content):
    savename = zipper(content)
    return savename, download_file_magicmix.update(visible=True)

def hide_download_file_magicmix():
    return download_file_magicmix.update(visible=False)

## Functions specific to inpaint
def zip_download_file_inpaint(content):
    savename = zipper(content)
    return savename, download_file_inpaint.update(visible=True)

def hide_download_file_inpaint():
    return download_file_inpaint.update(visible=False)

def change_ays_inpaint(use_ays):
    if use_ays:
        return num_inference_step_inpaint.update(interactive=False), sampler_inpaint.update(interactive=False)
    else:
        return num_inference_step_inpaint.update(interactive=True), sampler_inpaint.update(interactive=True)

## Functions specific to paintbyex
def zip_download_file_paintbyex(content):
    savename = zipper(content)
    return savename, download_file_paintbyex.update(visible=True)

def hide_download_file_paintbyex():
    return download_file_paintbyex.update(visible=False)

## Functions specific to outpaint
def zip_download_file_outpaint(content):
    savename = zipper(content)
    return savename, download_file_outpaint.update(visible=True)

def hide_download_file_outpaint():
    return download_file_outpaint.update(visible=False)

def change_ays_outpaint(use_ays):
    if use_ays:
        return num_inference_step_outpaint.update(interactive=False), sampler_outpaint.update(interactive=False)
    else:
        return num_inference_step_outpaint.update(interactive=True), sampler_outpaint.update(interactive=True)

## Functions specific to controlnet
def zip_download_file_controlnet(content):
    savename = zipper(content)
    return savename, download_file_controlnet.update(visible=True)

def hide_download_file_controlnet():
    return download_file_controlnet.update(visible=False)

def change_ays_controlnet(use_ays):
    if use_ays:
        return num_inference_step_controlnet.update(interactive=False), sampler_controlnet.update(interactive=False)
    else:
        return num_inference_step_controlnet.update(interactive=True), sampler_controlnet.update(interactive=True)

def change_preview_controlnet(input_value):
    return gs_variant_controlnet.update(value=input_value)

def change_preview_gs_controlnet(input_value):
    return variant_controlnet.update(value=input_value)

def change_model_type_controlnet(model_controlnet):
    model_controlnet = model_cleaner_sd(model_controlnet)
    if (model_controlnet == "stabilityai/sdxl-turbo"):
        return sampler_controlnet.update(value="Euler a"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=1), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "thibaud/sdxl_dpo_turbo"):
        return sampler_controlnet.update(value="UniPC"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=2), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "stabilityai/sd-turbo"):
        return sampler_controlnet.update(value="Euler a"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=1), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif ("ETRI-VILAB/KOALA-" in model_controlnet.upper()):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=3.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "GraydientPlatformAPI/lustify-lightning"):
        return sampler_controlnet.update(value="DPM++ SDE Karras"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=6), guidance_scale_controlnet.update(value=1.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "SG161222/RealVisXL_V5.0_Lightning"):
        return sampler_controlnet.update(value="DPM++ SDE Karras"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=5), guidance_scale_controlnet.update(value=1.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "Chan-Y/Stable-Flash-Lightning"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif ("LIGHTNING" in model_controlnet.upper()):
        return sampler_controlnet.update(value="DPM++ SDE Karras"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=1.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "sd-community/sdxl-flash") or (model_controlnet == "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl"):
        return sampler_controlnet.update(value="DPM++ SDE"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=6), guidance_scale_controlnet.update(value=3.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "RunDiffusion/Juggernaut-X-Hyper"):
        return sampler_controlnet.update(value="DPM++ SDE Karras"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=6), guidance_scale_controlnet.update(value=1.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "Corcelio/mobius"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=3.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "mann-e/Mann-E_Dreams") or (model_controlnet == "mann-e/Mann-E_Art") :
        return sampler_controlnet.update(value="DPM++ SDE Karras"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=6), guidance_scale_controlnet.update(value=3.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "John6666/jib-mix-realistic-xl-v15-maximus-sdxl"):
        return sampler_controlnet.update(value="DPM++ SDE"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=2.2), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "segmind/SSD-1B"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "segmind/Segmind-Vega"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=9.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "playgroundai/playground-v2-1024px-aesthetic"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=3.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "playgroundai/playground-v2.5-1024px-aesthetic"):
        return sampler_controlnet.update(value="EDM DPM++ 2M"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=15), guidance_scale_controlnet.update(value=3.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "playgroundai/playground-v2-512px-base"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=3.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif is_sdxl(model_controlnet):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif is_sd3(model_controlnet):
        return sampler_controlnet.update(value="Flow Match Euler"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "black-forest-labs/FLUX.1-schnell") or (model_controlnet == "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16") or (model_controlnet == "mikeyandfriends/PixelWave_FLUX.1-schnell_04") or (model_controlnet == "minpeter/FLUX-Hyperscale-fused-fast"):
        return sampler_controlnet.update(value="Flow Match Euler"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers"):
        return sampler_controlnet.update(value="Flow Match Euler"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=2), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "sayakpaul/FLUX.1-merged") or (model_controlnet == "shuttleai/shuttle-jaguar"):
        return sampler_controlnet.update(value="Flow Match Euler"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=3.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif is_flux(model_controlnet):
        return sampler_controlnet.update(value="Flow Match Euler"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=3.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    else:
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)

def change_model_type_controlnet_alternate2(model_controlnet):
    if is_noloras(model_controlnet):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model2_controlnet.update(choices=list(lora_model_list(model_controlnet, True).keys()), value="", interactive=lora_interaction)

def change_model_type_controlnet_alternate3(model_controlnet):
    if is_noloras(model_controlnet):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model3_controlnet.update(choices=list(lora_model_list(model_controlnet, True).keys()), value="", interactive=lora_interaction)

def change_model_type_controlnet_alternate4(model_controlnet):
    if is_noloras(model_controlnet):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model4_controlnet.update(choices=list(lora_model_list(model_controlnet, True).keys()), value="", interactive=lora_interaction)

def change_model_type_controlnet_alternate5(model_controlnet):
    if is_noloras(model_controlnet):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model5_controlnet.update(choices=list(lora_model_list(model_controlnet, True).keys()), value="", interactive=lora_interaction)

biniou_internal_previous_model_controlnet = ""
biniou_internal_previous_steps_controlnet = ""
biniou_internal_previous_cfg_controlnet = ""
biniou_internal_previous_trigger_controlnet = ""
biniou_internal_previous_sampler_controlnet = ""
def change_lora_model_controlnet(model, lora_model, prompt, steps, cfg_scale, sampler):
    global biniou_internal_previous_model_controlnet
    global biniou_internal_previous_steps_controlnet
    global biniou_internal_previous_cfg_controlnet
    global biniou_internal_previous_trigger_controlnet
    global biniou_internal_previous_sampler_controlnet
    lora_model = model_cleaner_lora(lora_model)
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_controlnet = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_controlnet = prompt
    else:
        lora_prompt_controlnet = prompt

    if (biniou_internal_previous_trigger_controlnet == ""):
        biniou_internal_previous_trigger_controlnet = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger_controlnet+ ", "
        lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_trigger, "")
        biniou_internal_previous_trigger_controlnet = lora_keyword

    lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    if is_fast_lora(lora_model):
        biniou_internal_previous_model_controlnet = model
        biniou_internal_previous_steps_controlnet = steps
        biniou_internal_previous_cfg_controlnet = cfg_scale
        biniou_internal_previous_sampler_controlnet = sampler
        if (lora_model == "ByteDance/SDXL-Lightning") or (lora_model == "GraydientPlatformAPI/lightning-faster-lora"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="LCM")
        elif ((lora_model == "ByteDance/Hyper-SD") or ("H1T/TCD-SD" in lora_model.upper())) and not is_sd3(model) and not is_flux(model):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=2), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="TCD")
        elif (lora_model == "openskyml/lcm-lora-sdxl-turbo"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="LCM")
        elif (lora_model == "tianweiy/DMD2"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="LCM")
        elif (lora_model == "wangfuyun/PCM_Weights"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=2), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="LCM")
        elif (lora_model == "jasperai/flash-sdxl"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=2), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="LCM")
        elif (lora_model == "sd-community/sdxl-flash-lora"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=6), guidance_scale_controlnet.update(value=3.0), sampler_controlnet.update(value="DPM++ SDE")
        elif (lora_model == "mann-e/Mann-E_Turbo"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=6), guidance_scale_controlnet.update(value=3.0), sampler_controlnet.update(value="DPM++ SDE Karras")
        elif (lora_model == "alimama-creative/slam-lora-sdxl"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=1.0), sampler_controlnet.update(value="LCM")
        elif (lora_model == "ByteDance/Hyper-SD") and is_sd3(model):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=3.0), sampler_controlnet.update(value="Flow Match Euler")
        elif (lora_model == "ByteDance/Hyper-SD") and is_flux(model):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=8), guidance_scale_controlnet.update(value=3.5), sampler_controlnet.update(value="Flow Match Euler")
        elif (lora_model == "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=3.5), sampler_controlnet.update(value="Flow Match Euler")
        elif (lora_model == "RED-AIGC/TDD") and is_flux(model):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=2.0), sampler_controlnet.update(value="Flow Match Euler")
        elif (lora_model == "alimama-creative/FLUX.1-Turbo-Alpha"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=8), guidance_scale_controlnet.update(value=3.5), sampler_controlnet.update(value="Flow Match Euler")
        elif (lora_model == "ostris/fluxdev2schnell-lora"):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=4), guidance_scale_controlnet.update(value=0.0), sampler_controlnet.update(value="Flow Match Euler")
    else:
        if ((biniou_internal_previous_model_controlnet == "") and (biniou_internal_previous_steps_controlnet == "") and (biniou_internal_previous_cfg_controlnet == "") and (biniou_internal_previous_sampler_controlnet == "")):
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(), guidance_scale_controlnet.update(), sampler_controlnet.update()
        elif (biniou_internal_previous_model_controlnet != model):
            biniou_internal_previous_model_controlnet = ""
            biniou_internal_previous_steps_controlnet = ""
            biniou_internal_previous_cfg_controlnet = ""
            biniou_internal_previous_sampler_controlnet = ""
            return prompt_controlnet.update(), num_inference_step_controlnet.update(), guidance_scale_controlnet.update(), sampler_controlnet.update()
        else:
            var_steps = int(biniou_internal_previous_steps_controlnet)
            var_cfg_scale = float(biniou_internal_previous_cfg_controlnet)
            var_sampler = str(biniou_internal_previous_sampler_controlnet)
            biniou_internal_previous_model_controlnet = ""
            biniou_internal_previous_steps_controlnet = ""
            biniou_internal_previous_cfg_controlnet = ""
            biniou_internal_previous_sampler_controlnet = ""
            return prompt_controlnet.update(value=lora_prompt_controlnet), num_inference_step_controlnet.update(value=var_steps), guidance_scale_controlnet.update(value=var_cfg_scale), sampler_controlnet.update(value=var_sampler)

biniou_internal_previous_trigger2_controlnet = ""
def change_lora_model2_controlnet(model, lora_model, prompt):
    global biniou_internal_previous_trigger2_controlnet
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_controlnet = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_controlnet = prompt
    else:
        lora_prompt_controlnet = prompt

    if (biniou_internal_previous_trigger2_controlnet == ""):
        biniou_internal_previous_trigger2_controlnet = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger2_controlnet+ ", "
        lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_trigger, "")
        biniou_internal_previous_trigger2_controlnet = lora_keyword

    lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_controlnet.update(value=lora_prompt_controlnet)

biniou_internal_previous_trigger3_controlnet = ""
def change_lora_model3_controlnet(model, lora_model, prompt):
    global biniou_internal_previous_trigger3_controlnet
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_controlnet = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_controlnet = prompt
    else:
        lora_prompt_controlnet = prompt

    if (biniou_internal_previous_trigger3_controlnet == ""):
        biniou_internal_previous_trigger3_controlnet = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger3_controlnet+ ", "
        lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_trigger, "")
        biniou_internal_previous_trigger3_controlnet = lora_keyword

    lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_controlnet.update(value=lora_prompt_controlnet)

biniou_internal_previous_trigger4_controlnet = ""
def change_lora_model4_controlnet(model, lora_model, prompt):
    global biniou_internal_previous_trigger4_controlnet
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_controlnet = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_controlnet = prompt
    else:
        lora_prompt_controlnet = prompt

    if (biniou_internal_previous_trigger4_controlnet == ""):
        biniou_internal_previous_trigger4_controlnet = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger4_controlnet+ ", "
        lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_trigger, "")
        biniou_internal_previous_trigger4_controlnet = lora_keyword

    lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_controlnet.update(value=lora_prompt_controlnet)

biniou_internal_previous_trigger5_controlnet = ""
def change_lora_model5_controlnet(model, lora_model, prompt):
    global biniou_internal_previous_trigger5_controlnet
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_controlnet = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_controlnet = prompt
    else:
        lora_prompt_controlnet = prompt

    if (biniou_internal_previous_trigger5_controlnet == ""):
        biniou_internal_previous_trigger5_controlnet = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger5_controlnet+ ", "
        lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_trigger, "")
        biniou_internal_previous_trigger5_controlnet = lora_keyword

    lora_prompt_controlnet = lora_prompt_controlnet.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_controlnet.update(value=lora_prompt_controlnet)

def change_txtinv_controlnet(model, txtinv, prompt, negative_prompt):
    if txtinv != "":
        txtinv_keyword = txtinv_list(model)[txtinv][1]
        if txtinv_keyword != "" and txtinv_keyword != "EasyNegative":
            txtinv_prompt_controlnet = txtinv_keyword+ ", "+ prompt
            txtinv_negative_prompt_controlnet = negative_prompt
        elif txtinv_keyword != "" and txtinv_keyword == "EasyNegative":
            txtinv_prompt_controlnet = prompt
            txtinv_negative_prompt_controlnet = txtinv_keyword+ ", "+ negative_prompt
        else:
            txtinv_prompt_controlnet = prompt
            txtinv_negative_prompt_controlnet = negative_prompt
    else:
        txtinv_prompt_controlnet = prompt
        txtinv_negative_prompt_controlnet = negative_prompt
    return prompt_controlnet.update(value=txtinv_prompt_controlnet), negative_prompt_controlnet.update(value=txtinv_negative_prompt_controlnet)

## Functions specific to faceid_ip 
def zip_download_file_faceid_ip(content):
    savename = zipper(content)
    return savename, download_file_faceid_ip.update(visible=True)

def hide_download_file_faceid_ip():
    return download_file_faceid_ip.update(visible=False)

def change_model_type_faceid_ip(model_faceid_ip, prompt):
    model_faceid_ip = model_cleaner_sd(model_faceid_ip)
    if (model_faceid_ip == "stabilityai/sdxl-turbo"):
        return sampler_faceid_ip.update(value="Euler a"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=0.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=False), prompt_faceid_ip.update()
#    elif (model_faceid_ip == "thibaud/sdxl_dpo_turbo"):
#        return sampler_faceid_ip.update(value="UniPC"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=0.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=False), prompt_faceid_ip.update()
    elif (model_faceid_ip == "stabilityai/sd-turbo"):
        return sampler_faceid_ip.update(value="Euler a"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=0.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=False), prompt_faceid_ip.update()
    elif (model_faceid_ip == "Chan-Y/Stable-Flash-Lightning"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="UniPC"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=15), guidance_scale_faceid_ip.update(value=5.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif ("LIGHTNING" in model_faceid_ip.upper()):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="DPM++ SDE Karras"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=1.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip  == "sd-community/sdxl-flash") or (model_faceid_ip  == "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="DPM++ SDE"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=3.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip  == "RunDiffusion/Juggernaut-X-Hyper"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="DPM++ SDE Karras"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=1.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip  == "Corcelio/mobius"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="UniPC"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=15), guidance_scale_faceid_ip.update(value=3.5), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip  == "mann-e/Mann-E_Dreams") or (model_faceid_ip  == "mann-e/Mann-E_Art") :
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="DPM++ SDE Karras"), width_faceid_ip.update(value=768), height_faceid_ip.update(value=768), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=3.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip  == "John6666/jib-mix-realistic-xl-v15-maximus-sdxl"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="DPM++ SDE"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=2.2), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip == "segmind/SSD-1B"):
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=35), guidance_scale_faceid_ip.update(value=7.5), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update()
    elif (model_faceid_ip == "segmind/Segmind-Vega"):
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=35), guidance_scale_faceid_ip.update(value=9.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update()
    elif (model_faceid_ip == "playgroundai/playground-v2-1024px-aesthetic"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=15), guidance_scale_faceid_ip.update(value=3.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip == "playgroundai/playground-v2.5-1024px-aesthetic"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="EDM DPM++ 2M"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=20), guidance_scale_faceid_ip.update(value=3.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif (model_faceid_ip == "playgroundai/playground-v2-512px-base"):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_faceid_ip.update(value=biniou_global_sd15_width), height_faceid_ip.update(value=biniou_global_sd15_height), num_inference_step_faceid_ip.update(value=15), guidance_scale_faceid_ip.update(value=3.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    elif is_sdxl(model_faceid_ip):
        if not ((" img " in prompt) or (" img," in prompt) or (" img:" in prompt)):
            photomaker_prompt_faceid_ip = " img "+ prompt
        else:
            photomaker_prompt_faceid_ip = prompt
        return sampler_faceid_ip.update(value="UniPC"), width_faceid_ip.update(value=biniou_global_sdxl_width), height_faceid_ip.update(value=biniou_global_sdxl_height), num_inference_step_faceid_ip.update(value=15), guidance_scale_faceid_ip.update(value=5.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update(value=photomaker_prompt_faceid_ip)
    else:
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(value=biniou_global_sd15_width), height_faceid_ip.update(value=biniou_global_sd15_height), num_inference_step_faceid_ip.update(value=35), guidance_scale_faceid_ip.update(value=7.5), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True), prompt_faceid_ip.update()

def change_model_type_faceid_ip_alternate2(model_faceid_ip):
    if is_noloras(model_faceid_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model2_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip, True).keys()), value="", interactive=lora_interaction)

def change_model_type_faceid_ip_alternate3(model_faceid_ip):
    if is_noloras(model_faceid_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model3_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip, True).keys()), value="", interactive=lora_interaction)

def change_model_type_faceid_ip_alternate4(model_faceid_ip):
    if is_noloras(model_faceid_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model4_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip, True).keys()), value="", interactive=lora_interaction)

def change_model_type_faceid_ip_alternate5(model_faceid_ip):
    if is_noloras(model_faceid_ip):
         lora_interaction = False
    else:
         lora_interaction = True
    return lora_model5_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip, True).keys()), value="", interactive=lora_interaction)

biniou_internal_previous_model_faceid_ip = ""
biniou_internal_previous_steps_faceid_ip = ""
biniou_internal_previous_cfg_faceid_ip = ""
biniou_internal_previous_trigger_faceid_ip = ""
biniou_internal_previous_sampler_faceid_ip = ""
def change_lora_model_faceid_ip(model, lora_model, prompt, steps, cfg_scale, sampler):
    global biniou_internal_previous_model_faceid_ip
    global biniou_internal_previous_steps_faceid_ip
    global biniou_internal_previous_cfg_faceid_ip
    global biniou_internal_previous_trigger_faceid_ip
    global biniou_internal_previous_sampler_faceid_ip
    lora_model = model_cleaner_lora(lora_model)
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_faceid_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_faceid_ip = prompt
    else:
        lora_prompt_faceid_ip = prompt

    if (biniou_internal_previous_trigger_faceid_ip == ""):
        biniou_internal_previous_trigger_faceid_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger_faceid_ip+ ", "
        lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger_faceid_ip = lora_keyword

    lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    if is_fast_lora(lora_model):
        biniou_internal_previous_model_faceid_ip = model
        biniou_internal_previous_steps_faceid_ip = steps
        biniou_internal_previous_cfg_faceid_ip = cfg_scale
        biniou_internal_previous_sampler_faceid_ip = sampler
        if (lora_model == "ByteDance/SDXL-Lightning") or (lora_model == "GraydientPlatformAPI/lightning-faster-lora"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=4), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
        elif (lora_model == "ByteDance/Hyper-SD") or ("H1T/TCD-SD" in lora_model.upper()):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="TCD")
        elif (lora_model == "openskyml/lcm-lora-sdxl-turbo"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=4), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
        elif (lora_model == "tianweiy/DMD2"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=4), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
        elif (lora_model == "wangfuyun/PCM_Weights"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
        elif (lora_model == "jasperai/flash-sdxl"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=4), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
        elif (lora_model == "jasperai/flash-sd"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
        elif (lora_model == "sd-community/sdxl-flash-lora"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=3.0), sampler_faceid_ip.update(value="DPM++ SDE")
        elif (lora_model == "mann-e/Mann-E_Turbo"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=6), guidance_scale_faceid_ip.update(value=3.0), sampler_faceid_ip.update(value="DPM++ SDE Karras")
        elif (lora_model == "alimama-creative/slam-lora-sdxl"):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=4), guidance_scale_faceid_ip.update(value=1.0), sampler_faceid_ip.update(value="LCM")
    else:
        if ((biniou_internal_previous_model_faceid_ip == "") and (biniou_internal_previous_steps_faceid_ip == "") and (biniou_internal_previous_cfg_faceid_ip == "") and (biniou_internal_previous_sampler_faceid_ip == "")):
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(), guidance_scale_faceid_ip.update(), sampler_faceid_ip.update()
        elif (biniou_internal_previous_model_faceid_ip != model):
            biniou_internal_previous_model_faceid_ip = ""
            biniou_internal_previous_steps_faceid_ip = ""
            biniou_internal_previous_cfg_faceid_ip = ""
            biniou_internal_previous_sampler_faceid_ip = ""
            return prompt_faceid_ip.update(), num_inference_step_faceid_ip.update(), guidance_scale_faceid_ip.update(), sampler_faceid_ip.update()
        else:
            var_steps = int(biniou_internal_previous_steps_faceid_ip)
            var_cfg_scale = float(biniou_internal_previous_cfg_faceid_ip)
            var_sampler = str(biniou_internal_previous_sampler_faceid_ip)
            biniou_internal_previous_model_faceid_ip = ""
            biniou_internal_previous_steps_faceid_ip = ""
            biniou_internal_previous_cfg_faceid_ip = ""
            biniou_internal_previous_sampler_faceid_ip = ""
            return prompt_faceid_ip.update(value=lora_prompt_faceid_ip), num_inference_step_faceid_ip.update(value=var_steps), guidance_scale_faceid_ip.update(value=var_cfg_scale), sampler_faceid_ip.update(value=var_sampler)

biniou_internal_previous_trigger2_faceid_ip = ""
def change_lora_model2_faceid_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger2_faceid_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_faceid_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_faceid_ip = prompt
    else:
        lora_prompt_faceid_ip = prompt

    if (biniou_internal_previous_trigger2_faceid_ip == ""):
        biniou_internal_previous_trigger2_faceid_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger2_faceid_ip+ ", "
        lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger2_faceid_ip = lora_keyword

    lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_faceid_ip.update(value=lora_prompt_faceid_ip)

biniou_internal_previous_trigger3_faceid_ip = ""
def change_lora_model3_faceid_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger3_faceid_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_faceid_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_faceid_ip = prompt
    else:
        lora_prompt_faceid_ip = prompt

    if (biniou_internal_previous_trigger3_faceid_ip == ""):
        biniou_internal_previous_trigger3_faceid_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger3_faceid_ip+ ", "
        lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger3_faceid_ip = lora_keyword

    lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_faceid_ip.update(value=lora_prompt_faceid_ip)

biniou_internal_previous_trigger4_faceid_ip = ""
def change_lora_model4_faceid_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger4_faceid_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_faceid_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_faceid_ip = prompt
    else:
        lora_prompt_faceid_ip = prompt

    if (biniou_internal_previous_trigger4_faceid_ip == ""):
        biniou_internal_previous_trigger4_faceid_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger4_faceid_ip+ ", "
        lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger4_faceid_ip = lora_keyword

    lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_faceid_ip.update(value=lora_prompt_faceid_ip)

biniou_internal_previous_trigger5_faceid_ip = ""
def change_lora_model5_faceid_ip(model, lora_model, prompt):
    global biniou_internal_previous_trigger5_faceid_ip
    lora_keyword = lora_model_list(model)[lora_model][1]

    if lora_model != "":
        if lora_keyword != "":
            lora_prompt_faceid_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_faceid_ip = prompt
    else:
        lora_prompt_faceid_ip = prompt

    if (biniou_internal_previous_trigger5_faceid_ip == ""):
        biniou_internal_previous_trigger5_faceid_ip = lora_keyword
    else:
        lora_trigger = biniou_internal_previous_trigger5_faceid_ip+ ", "
        lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_trigger, "")
        biniou_internal_previous_trigger5_faceid_ip = lora_keyword

    lora_prompt_faceid_ip = lora_prompt_faceid_ip.replace(lora_keyword+ ", "+ lora_keyword+ ", ", lora_keyword+ ", ")

    return prompt_faceid_ip.update(value=lora_prompt_faceid_ip)

def change_txtinv_faceid_ip(model, txtinv, prompt, negative_prompt):
    if txtinv != "":
        txtinv_keyword = txtinv_list(model)[txtinv][1]
        if txtinv_keyword != "" and txtinv_keyword != "EasyNegative":
            txtinv_prompt_faceid_ip = txtinv_keyword+ ", "+ prompt
            txtinv_negative_prompt_faceid_ip = negative_prompt
        elif txtinv_keyword != "" and txtinv_keyword == "EasyNegative":
            txtinv_prompt_faceid_ip = prompt
            txtinv_negative_prompt_faceid_ip = txtinv_keyword+ ", "+ negative_prompt
        else:
            txtinv_prompt_faceid_ip = prompt
            txtinv_negative_prompt_faceid_ip = negative_prompt
    else:
        txtinv_prompt_faceid_ip = prompt
        txtinv_negative_prompt_faceid_ip = negative_prompt
    return prompt_faceid_ip.update(value=txtinv_prompt_faceid_ip), negative_prompt_faceid_ip.update(value=txtinv_negative_prompt_faceid_ip)

## Functions specific to faceswap 
def zip_download_file_faceswap(content):
    savename = zipper(content)
    return savename, download_file_faceswap.update(visible=True)

def hide_download_file_faceswap():
    return download_file_faceswap.update(visible=False)

## Functions specific to Real ESRGAN

## Functions specific to GFPGAN

## Functions specific to MusicGen

## Functions specific to MusicGen Melody
def change_source_type_musicgen_mel(source_type_musicgen_mel):
    if source_type_musicgen_mel == "audio" :
        return source_audio_musicgen_mel.update(source="upload")
    elif source_type_musicgen_mel == "micro" :
        return source_audio_musicgen_mel.update(source="microphone")

## Functions specific to MusicLDM

## Functions specific to AudioGen

## Functions specific to Harmonai

## Functions specific to Bark

## Functions specific to Modelscope

def change_output_type_txt2vid_ms(output_type_txt2vid_ms):
    if output_type_txt2vid_ms == "mp4" :
        return out_txt2vid_ms.update(visible=True), gif_out_txt2vid_ms.update(visible=False), btn_txt2vid_ms.update(visible=True), btn_txt2vid_ms_gif.update(visible=False)
    elif output_type_txt2vid_ms == "gif" :
        return out_txt2vid_ms.update(visible=False), gif_out_txt2vid_ms.update(visible=True), btn_txt2vid_ms.update(visible=False), btn_txt2vid_ms_gif.update(visible=True)

## Functions specific to Text2Video-Zero

def change_model_type_txt2vid_ze(model_txt2vid_ze):
    model_txt2vid_ze = model_cleaner_sd(model_txt2vid_ze)
    if (model_txt2vid_ze == "stabilityai/sdxl-turbo"):
        return sampler_txt2vid_ze.update(value="Euler a"), width_txt2vid_ze.update(value=biniou_global_sd15_width), height_txt2vid_ze.update(value=biniou_global_sd15_height), num_inference_step_txt2vid_ze.update(value=2), guidance_scale_txt2vid_ze.update(value=0.0), negative_prompt_txt2vid_ze.update(interactive=False)
    elif ("ETRI-VILAB/KOALA-" in model_txt2vid_ze.upper()):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=3.5), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "GraydientPlatformAPI/lustify-lightning"):
        return sampler_txt2vid_ze.update(value="DPM++ SDE Karras"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=6), guidance_scale_txt2vid_ze.update(value=1.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "SG161222/RealVisXL_V5.0_Lightning"):
        return sampler_txt2vid_ze.update(value="DPM++ SDE Karras"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=5), guidance_scale_txt2vid_ze.update(value=1.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "Chan-Y/Stable-Flash-Lightning"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=7.5), negative_prompt_txt2vid_ze.update(interactive=True)
    elif ("LIGHTNING" in model_txt2vid_ze.upper()):
        return sampler_txt2vid_ze.update(value="DPM++ SDE Karras"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=4), guidance_scale_txt2vid_ze.update(value=1.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "sd-community/sdxl-flash") or (model_txt2vid_ze == "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl"):
        return sampler_txt2vid_ze.update(value="DPM++ SDE"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=6), guidance_scale_txt2vid_ze.update(value=3.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "RunDiffusion/Juggernaut-X-Hyper"):
        return sampler_txt2vid_ze.update(value="DPM++ SDE Karras"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=6), guidance_scale_txt2vid_ze.update(value=1.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "Corcelio/mobius"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=3.5), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "mann-e/Mann-E_Dreams") or (model_txt2vid_ze == "mann-e/Mann-E_Art"):
        return sampler_txt2vid_ze.update(value="DPM++ SDE Karras"), width_txt2vid_ze.update(value=768), height_txt2vid_ze.update(value=768), num_inference_step_txt2vid_ze.update(value=6), guidance_scale_txt2vid_ze.update(value=3.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "John6666/jib-mix-realistic-xl-v15-maximus-sdxl"):
        return sampler_txt2vid_ze.update(value="DPM++ SDE"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=4), guidance_scale_txt2vid_ze.update(value=2.2), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "segmind/Segmind-Vega"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=9.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "playgroundai/playground-v2-1024px-aesthetic"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=3.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "playgroundai/playground-v2.5-1024px-aesthetic"):
        return sampler_txt2vid_ze.update(value="EDM DPM++ 2M"), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=15), guidance_scale_txt2vid_ze.update(value=3.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "playgroundai/playground-v2-512px-base"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sd15_width), height_txt2vid_ze.update(value=biniou_global_sd15_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=3.0), negative_prompt_txt2vid_ze.update(interactive=True)
    elif is_sdxl(model_txt2vid_ze):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sdxl_width), height_txt2vid_ze.update(value=biniou_global_sdxl_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=7.5), negative_prompt_txt2vid_ze.update(interactive=True)
    else:
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(value=biniou_global_sd15_width), height_txt2vid_ze.update(value=biniou_global_sd15_height), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=7.5), negative_prompt_txt2vid_ze.update(interactive=True)

def change_output_type_txt2vid_ze(output_type_txt2vid_ze):
    if output_type_txt2vid_ze == "mp4" :
        return out_txt2vid_ze.update(visible=True), gif_out_txt2vid_ze.update(visible=False), btn_txt2vid_ze.update(visible=True), btn_txt2vid_ze_gif.update(visible=False)
    elif output_type_txt2vid_ze == "gif" :
        return out_txt2vid_ze.update(visible=False), gif_out_txt2vid_ze.update(visible=True), btn_txt2vid_ze.update(visible=False), btn_txt2vid_ze_gif.update(visible=True)


## Functions specific to AnimateDiff
def change_model_type_animatediff_lcm(model_animatediff_lcm, model_adapters_animatediff_lcm):
    if (model_adapters_animatediff_lcm == "wangfuyun/AnimateLCM"):
        scheduler = "LCM"
        cfg_scale = 2.0
        steps = 4
    elif (model_adapters_animatediff_lcm == "ByteDance/AnimateDiff-Lightning"):
        scheduler = "Euler"
        cfg_scale = 1.0
        steps = 4
    if (model_animatediff_lcm == "stabilityai/sdxl-turbo"):
        return sampler_animatediff_lcm.update(value=scheduler), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=2), guidance_scale_animatediff_lcm.update(value=0.0), negative_prompt_animatediff_lcm.update(interactive=False)
    elif ("XL" in model_animatediff_lcm.upper()) or ("ETRI-VILAB/KOALA-" in model_animatediff_lcm.upper()) or (model_animatediff_lcm == "segmind/SSD-1B") or (model_animatediff_lcm == "dataautogpt3/OpenDalleV1.1") or (model_animatediff_lcm == "dataautogpt3/ProteusV0.4"):
        return sampler_animatediff_lcm.update(value=scheduler), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=steps), guidance_scale_animatediff_lcm.update(value=3.5), negative_prompt_animatediff_lcm.update(interactive=True)
    elif (model_animatediff_lcm == "segmind/Segmind-Vega"):
        return sampler_animatediff_lcm.update(value=scheduler), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=steps), guidance_scale_animatediff_lcm.update(value=9.0), negative_prompt_animatediff_lcm.update(interactive=True)
    else:
        return sampler_animatediff_lcm.update(value=scheduler), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=steps), guidance_scale_animatediff_lcm.update(value=cfg_scale), negative_prompt_animatediff_lcm.update(interactive=True)

def change_output_type_animatediff_lcm(output_type_animatediff_lcm):
    if output_type_animatediff_lcm == "mp4" :
        return out_animatediff_lcm.update(visible=True), gif_out_animatediff_lcm.update(visible=False), btn_animatediff_lcm.update(visible=True), btn_animatediff_lcm_gif.update(visible=False)
    elif output_type_animatediff_lcm == "gif" :
        return out_animatediff_lcm.update(visible=False), gif_out_animatediff_lcm.update(visible=True), btn_animatediff_lcm.update(visible=False), btn_animatediff_lcm_gif.update(visible=True)

## Functions specific to Stable Video Diffusion
def change_model_type_img2vid(model_img2vid):
    if (model_img2vid == "stabilityai/stable-video-diffusion-img2vid"):
        return num_frames_img2vid.update(value=14)
    else:
        return num_frames_img2vid.update(value=25)

def change_output_type_img2vid(output_type_img2vid):
    if output_type_img2vid == "mp4" :
        return out_img2vid.update(visible=True), gif_out_img2vid.update(visible=False), btn_img2vid.update(visible=True), btn_img2vid_gif.update(visible=False)
    elif output_type_img2vid == "gif" :
        return out_img2vid.update(visible=False), gif_out_img2vid.update(visible=True), btn_img2vid.update(visible=False), btn_img2vid_gif.update(visible=True)

## Functions specific to Video Instruct-Pix2Pix
def change_output_type_vid2vid_ze(output_type_vid2vid_ze):
    if output_type_vid2vid_ze == "mp4" :
        return out_vid2vid_ze.update(visible=True), gif_out_vid2vid_ze.update(visible=False), btn_vid2vid_ze.update(visible=True), btn_vid2vid_ze_gif.update(visible=False)
    elif output_type_vid2vid_ze == "gif" :
        return out_vid2vid_ze.update(visible=False), gif_out_vid2vid_ze.update(visible=True), btn_vid2vid_ze.update(visible=False), btn_vid2vid_ze_gif.update(visible=True)

## Functions specific to txt2shape
def zip_download_file_txt2shape(content):
    savename = zipper(content)
    return savename, download_file_txt2shape.update(visible=True)

def zip_mesh_txt2shape(content):
    savename = zipper_file(content)
    return savename, download_file_txt2shape.update(visible=True)

def hide_download_file_txt2shape():
    return download_file_txt2shape.update(visible=False)

def change_output_type_txt2shape(output_type_txt2shape, out_size_txt2shape, mesh_out_size_txt2shape):
    if output_type_txt2shape == "gif" :
        return out_txt2shape.update(visible=True), mesh_out_txt2shape.update(visible=False), True, btn_txt2shape_gif.update(visible=True), btn_txt2shape_mesh.update(visible=False), download_btn_txt2shape_gif.update(visible=True), download_btn_txt2shape_gif.update(visible=False), download_file_txt2shape.update(visible=False), frame_size_txt2shape.update(value=out_size_txt2shape)
    elif output_type_txt2shape == "mesh" :
        return out_txt2shape.update(visible=False), mesh_out_txt2shape.update(visible=True), False, btn_txt2shape_gif.update(visible=False), btn_txt2shape_mesh.update(visible=True), download_btn_txt2shape_gif.update(visible=False), download_btn_txt2shape_gif.update(visible=True), download_file_txt2shape.update(visible=False), frame_size_txt2shape.update(value=mesh_out_size_txt2shape)

## Functions specific to img2shape
def zip_download_file_img2shape(content):
    savename = zipper(content)
    return savename, download_file_img2shape.update(visible=True)

def zip_mesh_img2shape(content):
    savename = zipper_file(content)
    return savename, download_file_img2shape.update(visible=True)

def hide_download_file_img2shape():
    return download_file_img2shape.update(visible=False)

def change_output_type_img2shape(output_type_img2shape, out_size_img2shape, mesh_out_size_img2shape):
    if output_type_img2shape == "gif" :
        return out_img2shape.update(visible=True), mesh_out_img2shape.update(visible=False), True, btn_img2shape_gif.update(visible=True), btn_img2shape_mesh.update(visible=False), download_btn_img2shape_gif.update(visible=True), download_btn_img2shape_gif.update(visible=False), download_file_img2shape.update(visible=False), frame_size_img2shape.update(value=out_size_img2shape)
    elif output_type_img2shape == "mesh" :
        return out_img2shape.update(visible=False), mesh_out_img2shape.update(visible=True), False, btn_img2shape_gif.update(visible=False), btn_img2shape_mesh.update(visible=True), download_btn_img2shape_gif.update(visible=False), download_btn_img2shape_gif.update(visible=True), download_file_img2shape.update(visible=False), frame_size_img2shape.update(value=mesh_out_size_img2shape)

## Functions specific to Models cleaner
def refresh_models_cleaner_list():
    return gr.CheckboxGroup(choices=biniouModelsManager("./models").modelslister(), value=None, type="value", label=biniou_lang_tab_settings_list_label, info=biniou_lang_tab_cleaner_list_info)

## Functions specific to LoRA models manager
def refresh_lora_models_manager_list_sd():
    return gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SD").modelslister(), value=None, type="value", label=biniou_lang_tab_settings_list_label, info=biniou_lang_tab_lora_models_list_info)

def refresh_lora_models_manager_list_sdxl():
    return gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SDXL").modelslister(), value=None, type="value", label=biniou_lang_tab_settings_list_label, info=biniou_lang_tab_lora_models_list_info)

## Functions specific to Textual inversion manager
def refresh_textinv_manager_list_sd():
    return gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SD").modelslister(), value=None, type="value", label=biniou_lang_tab_textinv_models_label, info=biniou_lang_tab_textinv_models_info)

def refresh_textinv_manager_list_sdxl():
    return gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelslister(), value=None, type="value", label=biniou_lang_tab_textinv_models_label, info=biniou_lang_tab_textinv_models_info)

## Authentication functions

def biniou_settings_login(user, password):
    admin_user, admin_pass = biniouUIControl.check_login_reader()
    if (user == admin_user) and (password == admin_pass):
        return acc_webui.update(visible=True), acc_models_cleaner.update(visible=True), acc_lora_models_manager.update(visible=True), acc_textinv_manager.update(visible=True), acc_sd_models_downloader.update(visible=True), acc_gguf_models_downloader.update(visible=True), biniou_login_user.update(value=""), biniou_login_pass.update(value=""), biniou_login_test.update(value="True")
    else:
        return acc_webui.update(), acc_models_cleaner.update(), acc_lora_models_manager.update(), acc_textinv_manager.update(), acc_sd_models_downloader.update(), acc_gguf_models_downloader.update(), biniou_login_user.update(), biniou_login_pass.update(), biniou_login_test.update(value="False")

def biniou_settings_login_test(test):
    if test == "True":
        return gr.Info(biniou_lang_tab_login_test_success)
    elif test == "False":
        return gr.Info(biniou_lang_tab_login_test_fail)

def biniou_settings_login_test_clean():
    return biniou_login_test.update(value="")

def biniou_settings_logout():
    return acc_webui.update(visible=False), acc_models_cleaner.update(visible=False), acc_lora_models_manager.update(visible=False), acc_textinv_manager.update(visible=False), acc_sd_models_downloader.update(visible=False), acc_gguf_models_downloader.update(visible=False), biniou_login_user.update(""), biniou_login_pass.update("")

## Functions specific to Common settings
def biniou_global_settings_auth_switch(auth_value):
	if auth_value:
		return biniou_global_settings_auth_message.update(interactive=True), biniou_global_settings_share.update(interactive=True)
	else:
		return biniou_global_settings_auth_message.update(interactive=False), biniou_global_settings_share.update(value=False, interactive=False)

## Functions specific to console
def refresh_logfile():
    return logfile_biniou
        
def show_download_console():
    return btn_download_file_console.update(visible=False), download_file_console.update(visible=True)

def hide_download_console():
    return btn_download_file_console.update(visible=True), download_file_console.update(visible=False)

## Functions specific to banner 

def dict_to_url(url):
    url_final = "./?"
    for key, value in url.items():
        url_final += "&"+ key+ "="+ value
    return url_final.replace("?&", "?")

def url_params_theme(url):
    url = eval(url)
    if url.get('__theme') != None and url['__theme'] == "dark":
        del url['__theme']
        url_final = dict_to_url(url)
        return f"<a href='https://github.com/Woolverine94/biniou' target='_blank' style='text-decoration: none;'><p style='float:left;'><img src='file/images/biniou_64.png' width='48' height='48'/></p><span style='text-align: left; font-size: 32px; font-weight: bold; line-height:48px;'>biniou</span></a><span style='vertical-align: bottom; line-height:48px; font-size: 10px;'> ({biniou_global_version}) </span><span style='vertical-align: top; line-height:48px;'><button onclick=\"window.location.href='{url_final}';\" title='{biniou_lang_light_mode}'>☀️</button></span>", banner_biniou.update(visible=True)
    elif url.get('__theme') == None:
        url['__theme'] = "dark"
        url_final = dict_to_url(url)
        return f"<a href='https://github.com/Woolverine94/biniou' target='_blank' style='text-decoration: none;'><p style='float:left;'><img src='file/images/biniou_64.png' width='48' height='48'/></p><span style='text-align: left; font-size: 32px; font-weight: bold; line-height:48px;'>biniou</span></a><span style='vertical-align: bottom; line-height:48px; font-size: 10px;'> ({biniou_global_version}) </span><span style='vertical-align: top; line-height:48px;'><button onclick=\"window.location.href='{url_final}';\" title='{biniou_lang_dark_mode}'>🌘</button></span>", banner_biniou.update(visible=True)

color_label = "#7b43ee"
color_label_light = "#4361ee"
color_label_button = "#4361ee"
background_color_light = "#f3f3f3"
background_fill_color = "#fcfcfc"
color_grey="#aaa"
color_darkgrey="#666"
color_black="#000"
color_white="#fff"

theme_gradio = gr.themes.Base().set(
    body_background_fill=background_color_light,
    background_fill_primary=background_fill_color,
    body_text_color=color_black,
    body_text_color_subdued=color_darkgrey,
    border_color_primary=color_grey,
    button_secondary_border_color=color_grey,
    input_border_width="1px",
#
    block_label_background_fill=background_color_light,
    block_label_background_fill_dark=color_label,
    block_label_border_color=color_black,
    block_label_border_color_dark=color_black,
    block_label_text_color=color_black,
    block_label_text_color_dark=color_white,
    block_title_background_fill=background_color_light,
    block_title_background_fill_dark=color_label,
    block_title_text_color=color_black,
    block_title_text_color_dark=color_white,
    block_title_padding='5px',
    block_title_radius='*radius_lg',
    button_primary_background_fill=color_label_button,
    button_primary_background_fill_dark=color_label_button,
    button_primary_border_color=color_label_button,
    button_primary_border_color_dark=color_label_button,
    button_primary_text_color=color_white,
    button_primary_text_color_dark=color_white,
)

with gr.Blocks(theme=theme_gradio, title="biniou") as demo:
    nsfw_filter = gr.Textbox(value="1", visible=False)
    url_params_current = gr.Textbox(value="", visible=False)
    banner_biniou = gr.HTML("""""", visible=False)
    url_params_current.change(url_params_theme, url_params_current, [banner_biniou, banner_biniou], show_progress="hidden")
    with gr.Tabs() as tabs:
# Chat
        with gr.TabItem(f"{biniou_lang_tab_text} ✍️", id=1) as tab_text:
            with gr.Tabs() as tabs_text:
# llamacpp
                with gr.TabItem(f"{biniou_lang_tab_llamacpp} 📝", id=11) as tab_llamacpp:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_llamacpp}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_llamacpp_about_desc} <a href='https://github.com/abetlen/llama-cpp-python' target='_blank'>llama-cpp-python</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(list(model_list_llamacpp.keys()))}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_llamacpp_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_llamacpp_about_models_inst1}</br>
                                - {biniou_lang_tab_llamacpp_about_models_inst2}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_llamacpp = gr.Dropdown(choices=list(model_list_llamacpp.keys()), value=list(model_list_llamacpp.keys())[0], label=biniou_lang_model_label, allow_custom_value=True, info=biniou_lang_tab_llamacpp_model_info)
                            with gr.Column():
                                quantization_llamacpp = gr.Textbox(value="", label=biniou_lang_tab_llamacpp_quantization_label, info=biniou_lang_tab_llamacpp_quantization_info)
                            with gr.Column():
                                max_tokens_llamacpp = gr.Slider(0, 131072, step=16, value=0, label=biniou_lang_maxtoken_label, info=biniou_lang_maxtoken_info)
                            with gr.Column():
                                seed_llamacpp = gr.Slider(0, 10000000000, step=1, value=1337, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                stream_llamacpp = gr.Checkbox(value=False, label=biniou_lang_stream_label, info=biniou_lang_stream_info, interactive=False)
                            with gr.Column():
                                n_ctx_llamacpp = gr.Slider(0, 131072, step=128, value=8192, label=biniou_lang_ctx_label, info=biniou_lang_ctx_info)
                            with gr.Column():
                                repeat_penalty_llamacpp = gr.Slider(0.0, 10.0, step=0.1, value=1.1, label=biniou_lang_penalty_label, info=biniou_lang_penalty_info)
                        with gr.Row():
                            with gr.Column():
                                temperature_llamacpp = gr.Slider(0.0, 10.0, step=0.1, value=0.8, label=biniou_lang_temperature_label, info=biniou_lang_temperature_info)
                            with gr.Column():
                                top_p_llamacpp = gr.Slider(0.0, 10.0, step=0.05, value=0.95, label=biniou_lang_top_p_label, info=biniou_lang_top_p_info)
                            with gr.Column():
                                top_k_llamacpp = gr.Slider(0, 500, step=1, value=40, label=biniou_lang_top_k_label, info=biniou_lang_top_k_info)
                        with gr.Row():
                            with gr.Column():
                                force_prompt_template_llamacpp = gr.Dropdown(choices=list(prompt_template_list_llamacpp.keys()), value=list(prompt_template_list_llamacpp.keys())[0], label=biniou_lang_tab_llamacpp_force_prompt_label, info=biniou_lang_tab_llamacpp_force_prompt_info)
                            with gr.Column():
                                gr.Number(visible=False)
                            with gr.Column():
                                gr.Number(visible=False)
                        with gr.Row():
                            with gr.Column():
                                prompt_template_llamacpp = gr.Textbox(label=biniou_lang_prompt_template_label, value=model_list_llamacpp[model_llamacpp.value][1], lines=4, max_lines=4, show_copy_button=True, info=biniou_lang_prompt_template_info)
                        with gr.Row():
                            with gr.Column():
                                system_template_llamacpp = gr.Textbox(label=biniou_lang_system_template_label, value=model_list_llamacpp[model_llamacpp.value][2], lines=4, max_lines=4, show_copy_button=True, info=biniou_lang_system_template_info)
                                model_llamacpp.change(fn=change_model_type_llamacpp, inputs=model_llamacpp, outputs=[prompt_template_llamacpp, system_template_llamacpp, quantization_llamacpp])
                                force_prompt_template_llamacpp.change(fn=change_prompt_template_llamacpp, inputs=force_prompt_template_llamacpp, outputs=[prompt_template_llamacpp, system_template_llamacpp])
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_llamacpp = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_llamacpp = gr.Textbox(value="llamacpp", visible=False, interactive=False)
                                del_ini_btn_llamacpp = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_llamacpp.value) else False)
                                save_ini_btn_llamacpp.click(
                                    fn=write_ini_llamacpp,
                                    inputs=[
                                        module_name_llamacpp,
                                        model_llamacpp,
                                        quantization_llamacpp,
                                        max_tokens_llamacpp,
                                        seed_llamacpp,
                                        stream_llamacpp,
                                        n_ctx_llamacpp,
                                        repeat_penalty_llamacpp,
                                        temperature_llamacpp,
                                        top_p_llamacpp,
                                        top_k_llamacpp,
                                        force_prompt_template_llamacpp,
                                        prompt_template_llamacpp,
                                        system_template_llamacpp,
                                        ]
                                    )
                                save_ini_btn_llamacpp.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_llamacpp.click(fn=lambda: del_ini_btn_llamacpp.update(interactive=True), outputs=del_ini_btn_llamacpp)
                                del_ini_btn_llamacpp.click(fn=lambda: del_ini(module_name_llamacpp.value))
                                del_ini_btn_llamacpp.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_llamacpp.click(fn=lambda: del_ini_btn_llamacpp.update(interactive=False), outputs=del_ini_btn_llamacpp)
                        if test_ini_exist(module_name_llamacpp.value) :
                            with open(f".ini/{module_name_llamacpp.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        history_llamacpp = gr.Chatbot(
                            label=biniou_lang_chatbot_history,
                            height=400,
                            autoscroll=True, 
                            show_copy_button=True,
                            interactive=True,
                            bubble_full_width = False,
                            avatar_images = ("./images/avatar_cat_64.png", "./images/biniou_64.png"),
                        )
                        last_reply_llamacpp = gr.Textbox(value="", visible=False)
                    with gr.Row():
                            prompt_llamacpp = gr.Textbox(label=biniou_lang_chatbot_prompt_label, lines=1, max_lines=3, show_copy_button=True, placeholder=biniou_lang_chatbot_prompt_placeholder, autofocus=True)
                            hidden_prompt_llamacpp = gr.Textbox(value="", visible=False)
                            last_reply_llamacpp.change(fn=lambda x:x, inputs=hidden_prompt_llamacpp, outputs=prompt_llamacpp)
                    with gr.Row():
                        with gr.Column():
                            btn_llamacpp = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_llamacpp_continue = gr.Button(f"{biniou_lang_continue} ➕")
                        with gr.Column():
                            btn_llamacpp_clear_output = gr.ClearButton(components=[history_llamacpp], value=f"{biniou_lang_clear_outputs} 🧹")
                        with gr.Column():
                            btn_download_file_llamacpp = gr.ClearButton(value=f"{biniou_lang_download_chat} 💾", visible=True)
                            download_file_llamacpp = gr.File(label=f"{biniou_lang_download_chat}", value=blankfile_common, height=30, interactive=False, visible=False)
                            download_file_llamacpp_hidden = gr.Textbox(value=blankfile_common, interactive=False, visible=False)
                            btn_download_file_llamacpp.click(fn=show_download_llamacpp, outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                            download_file_llamacpp_hidden.change(fn=lambda x:x, inputs=download_file_llamacpp_hidden, outputs=download_file_llamacpp)
                        btn_llamacpp.click(
                            fn=text_llamacpp,
                            inputs=[
                                model_llamacpp,
                                quantization_llamacpp,
                                max_tokens_llamacpp,
                                seed_llamacpp,
                                stream_llamacpp,
                                n_ctx_llamacpp,
                                repeat_penalty_llamacpp,
                                temperature_llamacpp,
                                top_p_llamacpp,
                                top_k_llamacpp,
                                prompt_llamacpp,
                                history_llamacpp,
                                prompt_template_llamacpp,
                                system_template_llamacpp,
                            ],
                            outputs=[
                                history_llamacpp,
                                last_reply_llamacpp,
                                download_file_llamacpp_hidden,
                            ],
                            show_progress="full",
                        )
                        btn_llamacpp.click(fn=hide_download_llamacpp, outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                        prompt_llamacpp.submit(
                            fn=text_llamacpp,
                            inputs=[
                                model_llamacpp,
                                quantization_llamacpp,
                                max_tokens_llamacpp,
                                seed_llamacpp,
                                stream_llamacpp,
                                n_ctx_llamacpp,
                                repeat_penalty_llamacpp,
                                temperature_llamacpp,
                                top_p_llamacpp,
                                top_k_llamacpp,
                                prompt_llamacpp,
                                history_llamacpp,
                                prompt_template_llamacpp,
                                system_template_llamacpp,
                            ],
                            outputs=[
                                history_llamacpp,
                                last_reply_llamacpp,
                                download_file_llamacpp_hidden,
                            ],
                            show_progress="full",
                        )
                        prompt_llamacpp.submit(fn=hide_download_llamacpp, outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                        btn_llamacpp_continue.click(
                            fn=text_llamacpp_continue,
                            inputs=[
                                model_llamacpp,
                                quantization_llamacpp,
                                max_tokens_llamacpp,
                                seed_llamacpp,
                                stream_llamacpp,
                                n_ctx_llamacpp,
                                repeat_penalty_llamacpp,
                                temperature_llamacpp,
                                top_p_llamacpp,
                                top_k_llamacpp,
                                history_llamacpp,
                            ],
                            outputs=[
                                history_llamacpp,
                                last_reply_llamacpp,
                                download_file_llamacpp_hidden,
                            ],
                            show_progress="full",
                        )
                        btn_llamacpp_continue.click(fn=hide_download_llamacpp, outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_tab_text_chatbot_send_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        llamacpp_nllb = gr.Button(f"✍️ >> {biniou_lang_tab_nllb}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        llamacpp_txt2img_sd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        llamacpp_txt2img_kd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}") 
                                        llamacpp_txt2img_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}") 
                                        llamacpp_txt2img_mjm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        llamacpp_txt2img_paa = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        llamacpp_img2img = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        llamacpp_img2img_ip = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        llamacpp_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        llamacpp_inpaint = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        llamacpp_controlnet = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        llamacpp_faceid_ip = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        llamacpp_musicgen = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        llamacpp_audiogen = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                                        llamacpp_bark = gr.Button(f"✍️ >> {biniou_lang_tab_bark}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        llamacpp_txt2vid_ms = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        llamacpp_txt2vid_ze = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        llamacpp_animatediff_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# llava
                with gr.TabItem(f"{biniou_lang_tab_llava} 👁️", id=12) as tab_llava:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_llava}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_llava_about_desc}<a href='https://github.com/abetlen/llama-cpp-python' target='_blank'>llama-cpp-python</a>, <a href='https://llava-vl.github.io/' target='_blank'>Llava</a>, <a href='https://github.com/SkunkworksAI/BakLLaVA' target='_blank'>BakLLaVA</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_llava_about_input_value}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(list(model_list_llava.keys()))}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_llava_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_llamacpp_about_models_inst1}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_llava = gr.Dropdown(choices=list(model_list_llava.keys()), value=list(model_list_llava.keys())[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                max_tokens_llava = gr.Slider(0, 131072, step=16, value=512, label=biniou_lang_maxtoken_label, info=biniou_lang_maxtoken_info)
                            with gr.Column():
                                seed_llava = gr.Slider(0, 10000000000, step=1, value=1337, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                stream_llava = gr.Checkbox(value=False, label=biniou_lang_stream_label, info=biniou_lang_stream_info, interactive=False)
                            with gr.Column():
                                n_ctx_llava = gr.Slider(0, 131072, step=128, value=8192, label=biniou_lang_ctx_label, info=biniou_lang_ctx_info)
                            with gr.Column():
                                repeat_penalty_llava = gr.Slider(0.0, 10.0, step=0.1, value=1.1, label=biniou_lang_penalty_label, info=biniou_lang_penalty_info)
                        with gr.Row():
                            with gr.Column():
                                temperature_llava = gr.Slider(0.0, 10.0, step=0.1, value=0.8, label=biniou_lang_temperature_label, info=biniou_lang_temperature_info)
                            with gr.Column():
                                top_p_llava = gr.Slider(0.0, 10.0, step=0.05, value=0.95, label=biniou_lang_top_p_label, info=biniou_lang_top_p_info)
                            with gr.Column():
                                top_k_llava = gr.Slider(0, 500, step=1, value=40, label=biniou_lang_top_k_label, info=biniou_lang_top_k_info)
                        with gr.Row():
                            with gr.Column():
                                prompt_template_llava = gr.Textbox(label=biniou_lang_prompt_template_label, value=model_list_llava[model_llava.value][2], lines=4, max_lines=4, show_copy_button=True, info=biniou_lang_tab_llava_template_prompt_info)
                        with gr.Row():
                            with gr.Column():
                                system_template_llava = gr.Textbox(label=biniou_lang_system_template_label, value=model_list_llava[model_llava.value][3], lines=4, max_lines=4, show_copy_button=True, info=biniou_lang_system_template_info, interactive=True)
                                model_llava.change(fn=change_model_type_llava, inputs=model_llava, outputs=[prompt_template_llava, system_template_llava])
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_llava = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_llava = gr.Textbox(value="llava", visible=False, interactive=False)
                                del_ini_btn_llava = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_llava.value) else False)
                                save_ini_btn_llava.click(
                                    fn=write_ini_llava,
                                    inputs=[
                                        module_name_llava,
                                        model_llava,
                                        max_tokens_llava,
                                        seed_llava,
                                        stream_llava,
                                        n_ctx_llava,
                                        repeat_penalty_llava,
                                        temperature_llava,
                                        top_p_llava,
                                        top_k_llava,
                                        prompt_template_llava,
                                        system_template_llava,
                                        ]
                                    )
                                save_ini_btn_llava.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_llava.click(fn=lambda: del_ini_btn_llava.update(interactive=True), outputs=del_ini_btn_llava)
                                del_ini_btn_llava.click(fn=lambda: del_ini(module_name_llava.value))
                                del_ini_btn_llava.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_llava.click(fn=lambda: del_ini_btn_llava.update(interactive=False), outputs=del_ini_btn_llava)
                        if test_ini_exist(module_name_llava.value) :
                            with open(f".ini/{module_name_llava.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column(scale=1):
                            img_llava = gr.Image(label=biniou_lang_img_input_label, type="filepath", height=400)
                        with gr.Column(scale=3):
                            history_llava = gr.Chatbot(
                                label=biniou_lang_chatbot_history,
                                height=400,
                                autoscroll=True,
                                show_copy_button=True,
                                interactive=True,
                                bubble_full_width = False,
                                avatar_images = ("./images/avatar_cat_64.png", "./images/biniou_64.png"),
                            )
                            last_reply_llava = gr.Textbox(value="", visible=False)
                    with gr.Row():
                            prompt_llava = gr.Textbox(label=biniou_lang_chatbot_prompt_label, lines=1, max_lines=3, show_copy_button=True, placeholder=biniou_lang_chatbot_prompt_placeholder, autofocus=True)
                            hidden_prompt_llava = gr.Textbox(value="", visible=False)
                            last_reply_llava.change(fn=lambda x:x, inputs=hidden_prompt_llava, outputs=prompt_llava)
                    with gr.Row():
                        with gr.Column():
                            btn_llava = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_llava_clear_input = gr.ClearButton(components=[img_llava, prompt_llava], value=f"{biniou_lang_clear_inputs} 🧹")
                            btn_llava_continue = gr.Button(f"{biniou_lang_continue} ➕", visible=False)
                        with gr.Column():
                            btn_llava_clear_output = gr.ClearButton(components=[history_llava], value=f"{biniou_lang_clear_outputs} 🧹")
                        with gr.Column():
                            btn_download_file_llava = gr.ClearButton(value=f"{biniou_lang_download_chat} 💾", visible=True)
                            download_file_llava = gr.File(label=f"{biniou_lang_download_chat}", value=blankfile_common, height=30, interactive=False, visible=False)
                            download_file_llava_hidden = gr.Textbox(value=blankfile_common, interactive=False, visible=False)
                            btn_download_file_llava.click(fn=show_download_llava, outputs=[btn_download_file_llava, download_file_llava])
                            download_file_llava_hidden.change(fn=lambda x:x, inputs=download_file_llava_hidden, outputs=download_file_llava)
                        btn_llava.click(
                            fn=text_llava,
                            inputs=[
                                model_llava,
                                max_tokens_llava,
                                seed_llava,
                                stream_llava,
                                n_ctx_llava, 
                                repeat_penalty_llava,
                                temperature_llava,
                                top_p_llava,
                                top_k_llava,
                                img_llava,
                                prompt_llava,
                                history_llava,
                                prompt_template_llava,
                                system_template_llava,
                            ],
                            outputs=[
                                history_llava,
                                last_reply_llava,
                                download_file_llava_hidden,
                            ],
                            show_progress="full",
                        )
                        btn_llava.click(fn=hide_download_llava, outputs=[btn_download_file_llava, download_file_llava])
                        prompt_llava.submit(
                            fn=text_llava,
                            inputs=[
                                model_llava,
                                max_tokens_llava,
                                seed_llava,
                                stream_llava,
                                n_ctx_llava, 
                                repeat_penalty_llava,
                                temperature_llava,
                                top_p_llava,
                                top_k_llava,
                                img_llava,
                                prompt_llava,
                                history_llava,
                                prompt_template_llava,
                                system_template_llava,
                            ],
                            outputs=[
                                history_llava,
                                last_reply_llava,
                                download_file_llava_hidden,
                            ],
                            show_progress="full",
                        )
                        prompt_llava.submit(fn=hide_download_llava, outputs=[btn_download_file_llava, download_file_llava])
                        btn_llava_continue.click(
                            fn=text_llava_continue,
                            inputs=[
                                model_llava,
                                max_tokens_llava,
                                seed_llava,
                                stream_llava,
                                n_ctx_llava,
                                repeat_penalty_llava,
                                temperature_llava,
                                top_p_llava,
                                top_k_llava,
                                img_llava,
                                history_llava,
                            ],
                            outputs=[
                                history_llava,
                                last_reply_llava,
                                download_file_llava_hidden,
                            ],
                            show_progress="full",
                        )
                        btn_llava_continue.click(fn=hide_download_llava, outputs=[btn_download_file_llava, download_file_llava])
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_tab_text_chatbot_send_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        llava_nllb = gr.Button(f"✍️ >> {biniou_lang_tab_nllb}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        llava_txt2img_sd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        llava_txt2img_kd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}") 
                                        llava_txt2img_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        llava_txt2img_mjm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        llava_txt2img_paa = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        llava_img2img = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        llava_img2img_ip = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        llava_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        llava_inpaint = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        llava_controlnet = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        llava_faceid_ip = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        llava_musicgen = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        llava_audiogen = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                                        llava_bark = gr.Button(f"✍️ >> {biniou_lang_tab_bark}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        llava_txt2vid_ms = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        llava_txt2vid_ze = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        llava_animatediff_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Image captioning
                with gr.TabItem(f"{biniou_lang_tab_img2txt_git} 👁️", id=13) as tab_img2txt_git:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_img2txt_git}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_img2txt_about_desc}</br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_img2txt_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_img2txt_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_img2txt_git)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_img2txt_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2txt_git = gr.Dropdown(choices=model_list_img2txt_git, value=model_list_img2txt_git[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                min_tokens_img2txt_git = gr.Slider(0, 128, step=1, value=0, label=biniou_lang_tab_img2txt_min_tokens_label, info=biniou_lang_tab_img2txt_min_tokens_info)
                            with gr.Column():
                                max_tokens_img2txt_git = gr.Slider(0, 256, step=1, value=20, label=biniou_lang_maxtoken_label, info=biniou_lang_maxtoken_info)
                        with gr.Row():
                            with gr.Column():
                                num_beams_img2txt_git = gr.Slider(1, 16, step=1, value=1, label=biniou_lang_tab_img2txt_num_beams_label, info=biniou_lang_tab_img2txt_num_beams_info)
                            with gr.Column():
                                num_beam_groups_img2txt_git = gr.Slider(1, 8, step=1, value=1, label=biniou_lang_tab_img2txt_gr_beams_label, info=biniou_lang_tab_img2txt_gr_beams_info)
                                num_beams_img2txt_git.change(set_num_beam_groups_img2txt_git, inputs=[num_beams_img2txt_git, num_beam_groups_img2txt_git], outputs=num_beam_groups_img2txt_git)
                            with gr.Column():
                                diversity_penalty_img2txt_git = gr.Slider(0.0, 5.0, step=0.01, value=0.5, label=biniou_lang_tab_img2txt_penalty_label, info=biniou_lang_tab_img2txt_penalty_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2txt_git = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_img2txt_git = gr.Textbox(value="img2txt_git", visible=False, interactive=False)
                                del_ini_btn_img2txt_git = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_img2txt_git.value) else False)
                                save_ini_btn_img2txt_git.click(
                                    fn=write_ini_img2txt_git, 
                                    inputs=[
                                        module_name_img2txt_git,
                                        model_img2txt_git,
                                        min_tokens_img2txt_git,
                                        max_tokens_img2txt_git,
                                        num_beams_img2txt_git,
                                        num_beam_groups_img2txt_git,
                                        diversity_penalty_img2txt_git,
                                        ]
                                    )
                                save_ini_btn_img2txt_git.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_img2txt_git.click(fn=lambda: del_ini_btn_img2txt_git.update(interactive=True), outputs=del_ini_btn_img2txt_git)
                                del_ini_btn_img2txt_git.click(fn=lambda: del_ini(module_name_img2txt_git.value))
                                del_ini_btn_img2txt_git.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_img2txt_git.click(fn=lambda: del_ini_btn_img2txt_git.update(interactive=False), outputs=del_ini_btn_img2txt_git)
                        if test_ini_exist(module_name_img2txt_git.value) :
                            with open(f".ini/{module_name_img2txt_git.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            img_img2txt_git = gr.Image(label=biniou_lang_img_input_label, type="pil", height=400)
                        with gr.Column():
                            out_img2txt_git = gr.Textbox(label=biniou_lang_tab_img2txt_captions, lines=15, show_copy_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_img2txt_git = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_img2txt_git_clear_input = gr.ClearButton(components=[img_img2txt_git], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_img2txt_git_clear_output = gr.ClearButton(components=[out_img2txt_git], value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_img2txt_git.click(
                            fn=text_img2txt_git,
                            inputs=[
                                model_img2txt_git,
                                max_tokens_img2txt_git,
                                min_tokens_img2txt_git,
                                num_beams_img2txt_git,
                                num_beam_groups_img2txt_git,
                                diversity_penalty_img2txt_git,
                                img_img2txt_git,
                            ],
                            outputs=out_img2txt_git,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        img2txt_git_nllb = gr.Button(f"✍️ >> {biniou_lang_tab_nllb}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2txt_git_txt2img_sd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        img2txt_git_txt2img_kd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        img2txt_git_txt2img_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        img2txt_git_txt2img_mjm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        img2txt_git_txt2img_paa = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        img2txt_git_img2img = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        img2txt_git_img2img_ip = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        img2txt_git_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        img2txt_git_inpaint = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        img2txt_git_controlnet = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        img2txt_git_faceid_ip = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        img2txt_git_musicgen = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        img2txt_git_audiogen = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        img2txt_git_txt2vid_ms = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        img2txt_git_txt2vid_ze = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        img2txt_git_animatediff_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        img2txt_git_img2img_both = gr.Button(f"🖼️+✍️ >> {biniou_lang_tab_img2img}")
                                        img2txt_git_img2img_ip_both = gr.Button(f"🖼️+✍️ >> {biniou_lang_tab_img2img_ip}")
                                        img2txt_git_pix2pix_both = gr.Button(f"🖼️+✍️ >> {biniou_lang_tab_pix2pix}")
                                        img2txt_git_inpaint_both = gr.Button(f"🖼️+✍️ >> {biniou_lang_tab_inpaint}")
                                        img2txt_git_controlnet_both = gr.Button(f"🖼️+✍️ >> {biniou_lang_tab_controlnet}")
                                        img2txt_git_faceid_ip_both = gr.Button(f"🖼️+✍️ >> {biniou_lang_tab_faceid_ip}")

# Whisper 
                with gr.TabItem(f"{biniou_lang_tab_whisper} 👂", id=14) as tab_whisper:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_whisper}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_whisper_about_desc}<a href='https://openai.com/research/whisper' target='_blank'>whisper</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_whisper_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_whisper_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(list(model_list_whisper.keys()))}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_whisper_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_whisper = gr.Dropdown(choices=list(model_list_whisper.keys()), value=list(model_list_whisper.keys())[4], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                srt_output_whisper = gr.Checkbox(value=False, label=biniou_lang_tab_whisper_srt_output_label, info=biniou_lang_tab_whisper_srt_output_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_whisper = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_whisper = gr.Textbox(value="whisper", visible=False, interactive=False)
                                del_ini_btn_whisper = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_whisper.value) else False)
                                save_ini_btn_whisper.click(
                                    fn=write_ini_whisper,
                                    inputs=[
                                        module_name_whisper,
                                        model_whisper,
                                        srt_output_whisper,
                                        ]
                                    )
                                save_ini_btn_whisper.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_whisper.click(fn=lambda: del_ini_btn_whisper.update(interactive=True), outputs=del_ini_btn_whisper)
                                del_ini_btn_whisper.click(fn=lambda: del_ini(module_name_whisper.value))
                                del_ini_btn_whisper.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_whisper.click(fn=lambda: del_ini_btn_whisper.update(interactive=False), outputs=del_ini_btn_whisper)
                        if test_ini_exist(module_name_whisper.value) :
                            with open(f".ini/{module_name_whisper.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    source_type_whisper = gr.Radio(choices=["audio", "micro"], value="audio", label=biniou_lang_input_type_label, info=biniou_lang_input_type_info)
                                with gr.Column():
                                    source_language_whisper = gr.Dropdown(choices=language_list_whisper, value=language_list_whisper[14], label=biniou_lang_input_language_label, info=biniou_lang_input_language_info)
                            with gr.Row():
                                source_audio_whisper = gr.Audio(label=biniou_lang_tab_whisper_src_audio_label, source="upload", type="filepath")
                                source_type_whisper.change(fn=change_source_type_whisper, inputs=source_type_whisper, outputs=source_audio_whisper)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    output_type_whisper = gr.Radio(choices=["transcribe", "translate"], value="transcribe", label=biniou_lang_tab_whisper_output_type_label, info=biniou_lang_tab_whisper_output_type_info)
                                with gr.Column():
                                    output_language_whisper = gr.Dropdown(choices=language_list_whisper, value=language_list_whisper[14], label=biniou_lang_output_language_label, info=biniou_lang_output_language_info, visible=False, interactive=False)
                            with gr.Row():
                                out_whisper = gr.Textbox(label=biniou_lang_tab_whisper_output_text, lines=9, max_lines=9, show_copy_button=True, interactive=False)
                                output_type_whisper.change(fn=change_output_type_whisper, inputs=output_type_whisper, outputs=output_language_whisper)
                    with gr.Row():
                        with gr.Column():
                            btn_whisper = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_whisper_clear_input = gr.ClearButton(components=[source_audio_whisper], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_whisper_clear_output = gr.ClearButton(components=[out_whisper], value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_whisper.click(
                            fn=text_whisper,
                            inputs=[
                                model_whisper,
                                srt_output_whisper,
                                source_language_whisper,
                                source_audio_whisper,
                                output_type_whisper,
                                output_language_whisper,
                            ],
                            outputs=out_whisper,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        whisper_nllb = gr.Button(f"✍️ >> {biniou_lang_tab_nllb}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        whisper_txt2img_sd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        whisper_txt2img_kd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        whisper_txt2img_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        whisper_txt2img_mjm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        whisper_txt2img_paa = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        whisper_img2img = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        whisper_img2img_ip = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        whisper_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        whisper_inpaint = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        whisper_controlnet = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        whisper_faceid_ip = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        whisper_musicgen = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        whisper_audiogen = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                                        whisper_bark = gr.Button(f"✍️ >> {biniou_lang_tab_bark}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        whisper_txt2vid_ms = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        whisper_txt2vid_ze = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        whisper_animatediff_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# nllb 
                with gr.TabItem(f"{biniou_lang_tab_nllb} 👥", id=15) as tab_nllb:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_nllb}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_nllb_about_desc}<a href='https://ai.meta.com/research/no-language-left-behind/' target='_blank'>nllb</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_nllb_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_nllb)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_nllb_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_nllb = gr.Dropdown(choices=model_list_nllb, value=model_list_nllb[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                max_tokens_nllb = gr.Slider(0, 1024, step=1, value=1024, label=biniou_lang_maxtoken_label, info=biniou_lang_maxtoken_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_nllb = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_nllb = gr.Textbox(value="nllb", visible=False, interactive=False)
                                del_ini_btn_nllb = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_nllb.value) else False)
                                save_ini_btn_nllb.click(
                                    fn=write_ini_nllb, 
                                    inputs=[
                                        module_name_nllb, 
                                        model_nllb, 
                                        max_tokens_nllb,
                                        ]
                                    )
                                save_ini_btn_nllb.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_nllb.click(fn=lambda: del_ini_btn_nllb.update(interactive=True), outputs=del_ini_btn_nllb)
                                del_ini_btn_nllb.click(fn=lambda: del_ini(module_name_nllb.value))
                                del_ini_btn_nllb.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_nllb.click(fn=lambda: del_ini_btn_nllb.update(interactive=False), outputs=del_ini_btn_nllb)
                        if test_ini_exist(module_name_nllb.value) :
                            with open(f".ini/{module_name_nllb.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                source_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()), value=list(language_list_nllb.keys())[47], label=biniou_lang_input_language_label, info=biniou_lang_input_language_info)
                            with gr.Row():
                                prompt_nllb = gr.Textbox(label=biniou_lang_tab_nllb_src_text_label, lines=9, max_lines=9, show_copy_button=True, placeholder=biniou_lang_tab_nllb_src_text_placeholder)
                        with gr.Column():
                            with gr.Row():
                                output_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()), value=list(language_list_nllb.keys())[47], label=biniou_lang_output_language_label, info=biniou_lang_output_language_info)
                            with gr.Row():
                                out_nllb = gr.Textbox(label=biniou_lang_tab_nllb_output_text, lines=9, max_lines=9, show_copy_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_nllb = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_nllb_clear_input = gr.ClearButton(components=[prompt_nllb], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_nllb_clear_output = gr.ClearButton(components=[out_nllb], value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_nllb.click(
                            fn=text_nllb,
                            inputs=[
                                model_nllb,
                                max_tokens_nllb, 
                                source_language_nllb,
                                prompt_nllb, 
                                output_language_nllb,
                            ],
                            outputs=out_nllb,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_output_value)
                                        nllb_llamacpp = gr.Button(f"✍️ >> {biniou_lang_tab_llamacpp}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        nllb_txt2img_sd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        nllb_txt2img_kd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        nllb_txt2img_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        nllb_txt2img_mjm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        nllb_txt2img_paa = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        nllb_img2img = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        nllb_img2img_ip = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        nllb_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        nllb_inpaint = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        nllb_controlnet = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        nllb_faceid_ip = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        nllb_musicgen = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        nllb_audiogen = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                                        nllb_bark = gr.Button(f"✍️ >> {biniou_lang_tab_bark}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        nllb_txt2vid_ms = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        nllb_txt2vid_ze = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        nllb_animatediff_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# txt2prompt
                if ram_size() >= 16 :
                    titletab_txt2prompt = f"{biniou_lang_tab_txt2prompt} 📝"
                else :
                    titletab_txt2prompt = f"{biniou_lang_tab_txt2prompt} ⛔"

                with gr.TabItem(titletab_txt2prompt, id=16) as tab_txt2prompt:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2prompt}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_txt2prompt_about_desc}</br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_prompt_label}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_txt2prompt_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2prompt)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2prompt_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2prompt = gr.Dropdown(choices=model_list_txt2prompt, value=model_list_txt2prompt[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                max_tokens_txt2prompt = gr.Slider(0, 2048, step=1, value=128, label=biniou_lang_maxtoken_label, info=biniou_lang_maxtoken_info)
                            with gr.Column():
                                repetition_penalty_txt2prompt = gr.Slider(0.0, 10.0, step=0.01, value=1.05, label=biniou_lang_penalty_label, info=biniou_lang_penalty_info)
                        with gr.Row():
                            with gr.Column():
                                seed_txt2prompt = gr.Slider(0, 4294967295, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                            with gr.Column():
                                num_prompt_txt2prompt = gr.Slider(1, 64, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_txt2prompt_batch_size_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2prompt = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2prompt = gr.Textbox(value="txt2prompt", visible=False, interactive=False)
                                del_ini_btn_txt2prompt = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2prompt.value) else False)
                                save_ini_btn_txt2prompt.click(
                                    fn=write_ini_txt2prompt,
                                    inputs=[
                                        module_name_txt2prompt,
                                        model_txt2prompt,
                                        max_tokens_txt2prompt,
                                        repetition_penalty_txt2prompt,
                                        seed_txt2prompt,
                                        num_prompt_txt2prompt,
                                        ]
                                    )
                                save_ini_btn_txt2prompt.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2prompt.click(fn=lambda: del_ini_btn_txt2prompt.update(interactive=True), outputs=del_ini_btn_txt2prompt)
                                del_ini_btn_txt2prompt.click(fn=lambda: del_ini(module_name_txt2prompt.value))
                                del_ini_btn_txt2prompt.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2prompt.click(fn=lambda: del_ini_btn_txt2prompt.update(interactive=False), outputs=del_ini_btn_txt2prompt)
                        if test_ini_exist(module_name_txt2prompt.value) :
                            with open(f".ini/{module_name_txt2prompt.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                prompt_txt2prompt = gr.Textbox(label=biniou_lang_prompt_label, lines=9, max_lines=9, show_copy_button=True, placeholder=biniou_lang_tab_txt2prompt_prompt_placeholder)
                            with gr.Row():
                                output_type_txt2prompt = gr.Radio(choices=["ChatGPT", "SD"], value="ChatGPT", label=biniou_lang_output_type_label, info=biniou_lang_tab_txt2prompt_output_type_info)
                                output_type_txt2prompt.change(fn=change_output_type_txt2prompt, inputs=output_type_txt2prompt, outputs=[model_txt2prompt, max_tokens_txt2prompt])
                        with gr.Column():
                            with gr.Row():
                                out_txt2prompt = gr.Textbox(label=biniou_lang_tab_nllb_output_text, lines=16, max_lines=16, show_copy_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_txt2prompt = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_txt2prompt_clear_input = gr.ClearButton(components=[prompt_txt2prompt], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_txt2prompt_clear_output = gr.ClearButton(components=[out_txt2prompt], value=f"{biniou_lang_clear_outputs} 🧹") 
                        btn_txt2prompt.click(
                            fn=text_txt2prompt,
                            inputs=[
                                model_txt2prompt,
                                max_tokens_txt2prompt, 
                                repetition_penalty_txt2prompt,
                                seed_txt2prompt, 
                                num_prompt_txt2prompt,
                                prompt_txt2prompt, 
                                output_type_txt2prompt,
                            ],
                            outputs=out_txt2prompt,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value) 
                                        txt2prompt_nllb = gr.Button(f"✍️ >> {biniou_lang_tab_nllb}")
                                        txt2prompt_llamacpp = gr.Button(f"✍️ >> {biniou_lang_tab_llamacpp}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2prompt_txt2img_sd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2prompt_txt2img_kd = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2prompt_txt2img_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}") 
                                        txt2prompt_txt2img_mjm = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        txt2prompt_txt2img_paa = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        txt2prompt_img2img = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        txt2prompt_img2img_ip = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2prompt_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2prompt_inpaint = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2prompt_controlnet = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2prompt_faceid_ip = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2prompt_txt2vid_ms = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2prompt_txt2vid_ze = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2prompt_animatediff_lcm = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# Image
        with gr.TabItem(f"{biniou_lang_tab_image} 🖼️", id=2) as tab_image:
            with gr.Tabs() as tabs_image:
# Stable Diffusion
                with gr.TabItem(f"{biniou_lang_tab_txt2img_sd} 🖼️", id=21) as tab_txt2img_sd:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2img_sd}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_image_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2img_sd)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2img_sd_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}</br>
                                <b>{biniou_lang_about_lora}</b></br>
                                - {biniou_lang_tab_image_about_lora_inst1}</br>
                                </div>                                
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_sd = gr.Dropdown(choices=model_list_txt2img_sd, value=model_list_txt2img_sd[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2img_sd = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_txt2img_sd = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_sd = gr.Slider(0.0, 20.0, step=0.1, value=7.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_txt2img_sd = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_txt2img_sd = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_sd = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2img_sd = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                seed_txt2img_sd = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_txt2img_sd = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_txt2img_sd = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_txt2img_sd = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_txt2img_sd = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_sd = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2img_sd = gr.Textbox(value="txt2img_sd", visible=False, interactive=False)
                                del_ini_btn_txt2img_sd = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2img_sd.value) else False)
                                save_ini_btn_txt2img_sd.click(
                                    fn=write_ini_txt2img_sd,
                                    inputs=[
                                        module_name_txt2img_sd,
                                        model_txt2img_sd,
                                        num_inference_step_txt2img_sd,
                                        sampler_txt2img_sd,
                                        guidance_scale_txt2img_sd,
                                        num_images_per_prompt_txt2img_sd,
                                        num_prompt_txt2img_sd,
                                        width_txt2img_sd,
                                        height_txt2img_sd,
                                        seed_txt2img_sd,
                                        use_gfpgan_txt2img_sd,
                                        tkme_txt2img_sd,
                                        clipskip_txt2img_sd,
                                        use_ays_txt2img_sd,
                                        ]
                                    )
                                save_ini_btn_txt2img_sd.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2img_sd.click(fn=lambda: del_ini_btn_txt2img_sd.update(interactive=True), outputs=del_ini_btn_txt2img_sd)
                                del_ini_btn_txt2img_sd.click(fn=lambda: del_ini(module_name_txt2img_sd.value))
                                del_ini_btn_txt2img_sd.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2img_sd.click(fn=lambda: del_ini_btn_txt2img_sd.update(interactive=False), outputs=del_ini_btn_txt2img_sd)
                        if test_ini_exist(module_name_txt2img_sd.value) :
                            with open(f".ini/{module_name_txt2img_sd.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                        with gr.Accordion(biniou_lang_lora_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_txt2img_sd = gr.Dropdown(choices=list(lora_model_list(model_txt2img_sd.value).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info)
                                with gr.Column():
                                    lora_weight_txt2img_sd = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info)
                            with gr.Row():
                                with gr.Column():
                                    lora_model2_txt2img_sd = gr.Dropdown(choices=list(lora_model_list(model_txt2img_sd.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight2_txt2img_sd = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model3_txt2img_sd = gr.Dropdown(choices=list(lora_model_list(model_txt2img_sd.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight3_txt2img_sd = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    lora_model4_txt2img_sd = gr.Dropdown(choices=list(lora_model_list(model_txt2img_sd.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight4_txt2img_sd = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model5_txt2img_sd = gr.Dropdown(choices=list(lora_model_list(model_txt2img_sd.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight5_txt2img_sd = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                        with gr.Accordion(biniou_lang_textinv_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_txt2img_sd = gr.Dropdown(choices=list(txtinv_list(model_txt2img_sd.value).keys()), value="", label=biniou_lang_textinv_label, info=biniou_lang_textinv_info)
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2img_sd = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2img_sd = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                        model_txt2img_sd.change(
                            fn=change_model_type_txt2img_sd,
                            inputs=[model_txt2img_sd],
                            outputs=[
                                sampler_txt2img_sd,
                                width_txt2img_sd,
                                height_txt2img_sd,
                                num_inference_step_txt2img_sd,
                                guidance_scale_txt2img_sd,
                                lora_model_txt2img_sd,
                                txtinv_txt2img_sd,
                                negative_prompt_txt2img_sd
                            ]
                        )
                        model_txt2img_sd.change(fn=change_model_type_txt2img_sd_alternate2, inputs=[model_txt2img_sd],outputs=[lora_model2_txt2img_sd])
                        model_txt2img_sd.change(fn=change_model_type_txt2img_sd_alternate3, inputs=[model_txt2img_sd],outputs=[lora_model3_txt2img_sd])
                        model_txt2img_sd.change(fn=change_model_type_txt2img_sd_alternate4, inputs=[model_txt2img_sd],outputs=[lora_model4_txt2img_sd])
                        model_txt2img_sd.change(fn=change_model_type_txt2img_sd_alternate5, inputs=[model_txt2img_sd],outputs=[lora_model5_txt2img_sd])

                        lora_model_txt2img_sd.change(fn=change_lora_model_txt2img_sd, inputs=[model_txt2img_sd, lora_model_txt2img_sd, prompt_txt2img_sd, num_inference_step_txt2img_sd, guidance_scale_txt2img_sd, sampler_txt2img_sd], outputs=[prompt_txt2img_sd, num_inference_step_txt2img_sd, guidance_scale_txt2img_sd, sampler_txt2img_sd])

                        lora_model2_txt2img_sd.change(fn=change_lora_model2_txt2img_sd, inputs=[model_txt2img_sd, lora_model2_txt2img_sd, prompt_txt2img_sd], outputs=[prompt_txt2img_sd])
                        lora_model3_txt2img_sd.change(fn=change_lora_model3_txt2img_sd, inputs=[model_txt2img_sd, lora_model3_txt2img_sd, prompt_txt2img_sd], outputs=[prompt_txt2img_sd])
                        lora_model4_txt2img_sd.change(fn=change_lora_model4_txt2img_sd, inputs=[model_txt2img_sd, lora_model4_txt2img_sd, prompt_txt2img_sd], outputs=[prompt_txt2img_sd])
                        lora_model5_txt2img_sd.change(fn=change_lora_model5_txt2img_sd, inputs=[model_txt2img_sd, lora_model5_txt2img_sd, prompt_txt2img_sd], outputs=[prompt_txt2img_sd])

                        txtinv_txt2img_sd.change(fn=change_txtinv_txt2img_sd, inputs=[model_txt2img_sd, txtinv_txt2img_sd, prompt_txt2img_sd, negative_prompt_txt2img_sd], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd])
                        use_ays_txt2img_sd.change(fn=change_ays_txt2img_sd, inputs=use_ays_txt2img_sd, outputs=[num_inference_step_txt2img_sd, sampler_txt2img_sd])
                        with gr.Column(scale=2):
                            out_txt2img_sd = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                            )
                            gs_out_txt2img_sd = gr.State()
                            sel_out_txt2img_sd = gr.Number(precision=0, visible=False)
                            out_txt2img_sd.select(get_select_index, None, sel_out_txt2img_sd)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_txt2img_sd = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_txt2img_sd = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_txt2img_sd.click(fn=zip_download_file_txt2img_sd, inputs=out_txt2img_sd, outputs=[download_file_txt2img_sd, download_file_txt2img_sd])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_sd = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_sd_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2img_sd_cancel.click(fn=initiate_stop_txt2img_sd, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2img_sd_clear_input = gr.ClearButton(components=[prompt_txt2img_sd, negative_prompt_txt2img_sd], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_txt2img_sd_clear_output = gr.ClearButton(components=[out_txt2img_sd, gs_out_txt2img_sd], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2img_sd.click(fn=hide_download_file_txt2img_sd, inputs=None, outputs=download_file_txt2img_sd)
                            btn_txt2img_sd.click(
                            fn=image_txt2img_sd,
                            inputs=[
                                model_txt2img_sd,
                                sampler_txt2img_sd,
                                prompt_txt2img_sd,
                                negative_prompt_txt2img_sd,
                                num_images_per_prompt_txt2img_sd,
                                num_prompt_txt2img_sd,
                                guidance_scale_txt2img_sd,
                                num_inference_step_txt2img_sd,
                                height_txt2img_sd,
                                width_txt2img_sd,
                                seed_txt2img_sd,
                                use_gfpgan_txt2img_sd,
                                nsfw_filter,
                                tkme_txt2img_sd,
                                clipskip_txt2img_sd,
                                use_ays_txt2img_sd,
                                lora_model_txt2img_sd,
                                lora_weight_txt2img_sd,
                                lora_model2_txt2img_sd,
                                lora_weight2_txt2img_sd,
                                lora_model3_txt2img_sd,
                                lora_weight3_txt2img_sd,
                                lora_model4_txt2img_sd,
                                lora_weight4_txt2img_sd,
                                lora_model5_txt2img_sd,
                                lora_weight5_txt2img_sd,
                                txtinv_txt2img_sd,
                            ],
                                outputs=[out_txt2img_sd, gs_out_txt2img_sd],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        txt2img_sd_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        txt2img_sd_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_sd_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        txt2img_sd_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_sd_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        txt2img_sd_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_sd_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        txt2img_sd_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_sd_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        txt2img_sd_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        txt2img_sd_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_sd_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        txt2img_sd_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        txt2img_sd_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        txt2img_sd_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_sd_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        txt2img_sd_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_sd_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2img_sd_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        txt2img_sd_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        txt2img_sd_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        txt2img_sd_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_sd_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_sd_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_sd_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_sd_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_sd_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_sd_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2img_sd_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2img_sd_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_sd_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_sd_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_sd_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_sd_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_sd_controlnet_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_sd_faceid_ip_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_faceid_ip}")

# Kandinsky
                if ram_size() >= 16 :
                    titletab_txt2img_kd = f"{biniou_lang_tab_txt2img_kd} 🖼️"
                else :
                    titletab_txt2img_kd = f"{biniou_lang_tab_txt2img_kd} ⛔"

                with gr.TabItem(titletab_txt2img_kd, id=22) as tab_txt2img_kd:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2img_kd}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_image_about_desc}<a href='https://github.com/ai-forever/Kandinsky-2' target='_blank'>Kandinsky</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2img_kd)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2img_kd_about_instruct}</br>
                                </div>
                                """
                            )                                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_kd = gr.Dropdown(choices=model_list_txt2img_kd, value=model_list_txt2img_kd[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2img_kd = gr.Slider(1, biniou_global_steps_max, step=1, value=25, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_txt2img_kd = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[5], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_kd = gr.Slider(0.1, 20.0, step=0.1, value=4.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_txt2img_kd = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_txt2img_kd = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_kd = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2img_kd = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                seed_txt2img_kd = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_txt2img_kd = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_kd = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2img_kd = gr.Textbox(value="txt2img_kd", visible=False, interactive=False)
                                del_ini_btn_txt2img_kd = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2img_kd.value) else False)
                                save_ini_btn_txt2img_kd.click(
                                    fn=write_ini_txt2img_kd, 
                                    inputs=[
                                        module_name_txt2img_kd,
                                        model_txt2img_kd,
                                        num_inference_step_txt2img_kd,
                                        sampler_txt2img_kd,
                                        guidance_scale_txt2img_kd,
                                        num_images_per_prompt_txt2img_kd,
                                        num_prompt_txt2img_kd,
                                        width_txt2img_kd,
                                        height_txt2img_kd,
                                        seed_txt2img_kd,
                                        use_gfpgan_txt2img_kd,
                                        ]
                                    )
                                save_ini_btn_txt2img_kd.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2img_kd.click(fn=lambda: del_ini_btn_txt2img_kd.update(interactive=True), outputs=del_ini_btn_txt2img_kd)
                                del_ini_btn_txt2img_kd.click(fn=lambda: del_ini(module_name_txt2img_kd.value))
                                del_ini_btn_txt2img_kd.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2img_kd.click(fn=lambda: del_ini_btn_txt2img_kd.update(interactive=False), outputs=del_ini_btn_txt2img_kd)
                        if test_ini_exist(module_name_txt2img_kd.value) :
                            with open(f".ini/{module_name_txt2img_kd.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2img_kd = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_tab_txt2img_kd_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2img_kd = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_tab_txt2img_kd_negprompt_placeholder)
                        model_txt2img_kd.change(
                            fn=change_model_type_txt2img_kd,
                            inputs=[model_txt2img_kd],
                            outputs=[
                                width_txt2img_kd,
                                height_txt2img_kd,
                                num_inference_step_txt2img_kd,
                                sampler_txt2img_kd,
                            ]
                        )
                        with gr.Column(scale=2):
                            out_txt2img_kd = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery_k",
                                columns=3,
                                height=400,
                            )
                            gs_out_txt2img_kd = gr.State()
                            sel_out_txt2img_kd = gr.Number(precision=0, visible=False)
                            out_txt2img_kd.select(get_select_index, None, sel_out_txt2img_kd)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_txt2img_kd = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_txt2img_kd = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_txt2img_kd.click(fn=zip_download_file_txt2img_kd, inputs=out_txt2img_kd, outputs=[download_file_txt2img_kd, download_file_txt2img_kd])                            
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_kd = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_kd_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2img_kd_cancel.click(fn=initiate_stop_txt2img_kd, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2img_kd_clear_input = gr.ClearButton(components=[prompt_txt2img_kd, negative_prompt_txt2img_kd], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_txt2img_kd_clear_output = gr.ClearButton(components=[out_txt2img_kd, gs_out_txt2img_kd], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2img_kd.click(fn=hide_download_file_txt2img_kd, inputs=None, outputs=download_file_txt2img_kd)
                            btn_txt2img_kd.click(
                                fn=image_txt2img_kd,
                                inputs=[
                                    model_txt2img_kd,
                                    sampler_txt2img_kd,
                                    prompt_txt2img_kd,
                                    negative_prompt_txt2img_kd,
                                    num_images_per_prompt_txt2img_kd,
                                    num_prompt_txt2img_kd,
                                    guidance_scale_txt2img_kd,
                                    num_inference_step_txt2img_kd,
                                    height_txt2img_kd,
                                    width_txt2img_kd,
                                    seed_txt2img_kd,
                                    use_gfpgan_txt2img_kd,
                                ],
                                outputs=[out_txt2img_kd, gs_out_txt2img_kd],
                                show_progress="full",
                            )                    
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        txt2img_kd_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        txt2img_kd_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_kd_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        txt2img_kd_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_kd_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        txt2img_kd_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_kd_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        txt2img_kd_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_kd_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        txt2img_kd_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        txt2img_kd_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_kd_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        txt2img_kd_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        txt2img_kd_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        txt2img_kd_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_kd_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        txt2img_kd_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_kd_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2img_kd_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        txt2img_kd_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        txt2img_kd_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        txt2img_kd_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_kd_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_kd_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_kd_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_kd_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_kd_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_kd_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2img_kd_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2img_kd_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_kd_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_kd_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_kd_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_kd_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_kd_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_kd_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")

# LCM
                with gr.TabItem(f"{biniou_lang_tab_txt2img_lcm} 🖼️", id=23) as tab_txt2img_lcm:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2img_lcm}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_txt2img_lcm_about_desc}<a href='https://github.com/luosiallen/latent-consistency-model' target='_blank'>LCM (Latent Consistency Model)</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2img_lcm)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2img_lcm_about_instruct}
                                <b>{biniou_lang_about_lora}</b></br>
                                - {biniou_lang_tab_image_about_lora_inst1}</br>
                                """
                            )                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_lcm = gr.Dropdown(choices=model_list_txt2img_lcm, value=model_list_txt2img_lcm[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2img_lcm = gr.Slider(1, biniou_global_steps_max, step=1, value=4, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_txt2img_lcm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[13], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_lcm = gr.Slider(0.1, 20.0, step=0.1, value=8.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                lcm_origin_steps_txt2img_lcm = gr.Slider(1, biniou_global_steps_max, step=1, value=50, label=biniou_lang_tab_txt2img_lcm_origin_label, info=biniou_lang_tab_txt2img_lcm_origin_info)
                            with gr.Column():
                                num_images_per_prompt_txt2img_lcm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_txt2img_lcm = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_lcm = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2img_lcm = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                seed_txt2img_lcm = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info) 
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_lcm = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_txt2img_lcm = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_lcm = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2img_lcm = gr.Textbox(value="txt2img_lcm", visible=False, interactive=False)
                                del_ini_btn_txt2img_lcm = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2img_lcm.value) else False)
                                save_ini_btn_txt2img_lcm.click(
                                    fn=write_ini_txt2img_lcm, 
                                    inputs=[
                                        module_name_txt2img_lcm, 
                                        model_txt2img_lcm, 
                                        num_inference_step_txt2img_lcm,
                                        sampler_txt2img_lcm,
                                        guidance_scale_txt2img_lcm,
                                        lcm_origin_steps_txt2img_lcm,
                                        num_images_per_prompt_txt2img_lcm,
                                        num_prompt_txt2img_lcm,
                                        width_txt2img_lcm,
                                        height_txt2img_lcm,
                                        seed_txt2img_lcm,
                                        use_gfpgan_txt2img_lcm,
                                        tkme_txt2img_lcm,
                                        ]
                                    )
                                save_ini_btn_txt2img_lcm.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2img_lcm.click(fn=lambda: del_ini_btn_txt2img_lcm.update(interactive=True), outputs=del_ini_btn_txt2img_lcm)
                                del_ini_btn_txt2img_lcm.click(fn=lambda: del_ini(module_name_txt2img_lcm.value))
                                del_ini_btn_txt2img_lcm.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2img_lcm.click(fn=lambda: del_ini_btn_txt2img_lcm.update(interactive=False), outputs=del_ini_btn_txt2img_lcm)
                        if test_ini_exist(module_name_txt2img_lcm.value) :
                            with open(f".ini/{module_name_txt2img_lcm.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                        with gr.Accordion(biniou_lang_lora_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_txt2img_lcm = gr.Dropdown(choices=list(lora_model_list(model_txt2img_lcm.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info)
                                with gr.Column():
                                    lora_weight_txt2img_lcm = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info)
                            with gr.Row():
                                with gr.Column():
                                    lora_model2_txt2img_lcm = gr.Dropdown(choices=list(lora_model_list(model_txt2img_lcm.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight2_txt2img_lcm = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model3_txt2img_lcm = gr.Dropdown(choices=list(lora_model_list(model_txt2img_lcm.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight3_txt2img_lcm = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    lora_model4_txt2img_lcm = gr.Dropdown(choices=list(lora_model_list(model_txt2img_lcm.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight4_txt2img_lcm = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model5_txt2img_lcm = gr.Dropdown(choices=list(lora_model_list(model_txt2img_lcm.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight5_txt2img_lcm = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                        with gr.Accordion(biniou_lang_textinv_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_txt2img_lcm = gr.Dropdown(choices=list(txtinv_list(model_txt2img_lcm.value).keys()), value="", label=biniou_lang_textinv_label, info=biniou_lang_textinv_info)
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2img_lcm = gr.Textbox(lines=18, max_lines=18, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_tab_txt2img_lcm_prompt_placeholder)
                        model_txt2img_lcm.change(fn=change_model_type_txt2img_lcm,
                            inputs=model_txt2img_lcm, 
                            outputs=[
                                width_txt2img_lcm, 
                                height_txt2img_lcm, 
                                guidance_scale_txt2img_lcm, 
                                num_inference_step_txt2img_lcm,
                                lora_model_txt2img_lcm,
                                txtinv_txt2img_lcm,
                            ]
                        )
                        model_txt2img_lcm.change(fn=change_model_type_txt2img_lcm_alternate2, inputs=[model_txt2img_lcm],outputs=[lora_model2_txt2img_lcm])
                        model_txt2img_lcm.change(fn=change_model_type_txt2img_lcm_alternate3, inputs=[model_txt2img_lcm],outputs=[lora_model3_txt2img_lcm])
                        model_txt2img_lcm.change(fn=change_model_type_txt2img_lcm_alternate4, inputs=[model_txt2img_lcm],outputs=[lora_model4_txt2img_lcm])
                        model_txt2img_lcm.change(fn=change_model_type_txt2img_lcm_alternate5, inputs=[model_txt2img_lcm],outputs=[lora_model5_txt2img_lcm])
                        lora_model_txt2img_lcm.change(fn=change_lora_model_txt2img_lcm, inputs=[model_txt2img_lcm, lora_model_txt2img_lcm, prompt_txt2img_lcm, num_inference_step_txt2img_lcm, guidance_scale_txt2img_lcm, sampler_txt2img_lcm], outputs=[prompt_txt2img_lcm, num_inference_step_txt2img_lcm, guidance_scale_txt2img_lcm, sampler_txt2img_lcm])
                        lora_model2_txt2img_lcm.change(fn=change_lora_model2_txt2img_lcm, inputs=[model_txt2img_lcm, lora_model2_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        lora_model3_txt2img_lcm.change(fn=change_lora_model3_txt2img_lcm, inputs=[model_txt2img_lcm, lora_model3_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        lora_model4_txt2img_lcm.change(fn=change_lora_model4_txt2img_lcm, inputs=[model_txt2img_lcm, lora_model4_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        lora_model5_txt2img_lcm.change(fn=change_lora_model5_txt2img_lcm, inputs=[model_txt2img_lcm, lora_model5_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        txtinv_txt2img_lcm.change(fn=change_txtinv_txt2img_lcm, inputs=[model_txt2img_lcm, txtinv_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        with gr.Column(scale=2):
                            out_txt2img_lcm = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                            )    
                            gs_out_txt2img_lcm = gr.State()
                            sel_out_txt2img_lcm = gr.Number(precision=0, visible=False)
                            out_txt2img_lcm.select(get_select_index, None, sel_out_txt2img_lcm)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_txt2img_lcm = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_txt2img_lcm = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_txt2img_lcm.click(fn=zip_download_file_txt2img_lcm, inputs=out_txt2img_lcm, outputs=[download_file_txt2img_lcm, download_file_txt2img_lcm])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_lcm = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_txt2img_lcm_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2img_lcm_cancel.click(fn=initiate_stop_txt2img_lcm, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2img_lcm_clear_input = gr.ClearButton(components=[prompt_txt2img_lcm], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_txt2img_lcm_clear_output = gr.ClearButton(components=[out_txt2img_lcm, gs_out_txt2img_lcm], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2img_lcm.click(fn=hide_download_file_txt2img_lcm, inputs=None, outputs=download_file_txt2img_lcm)
                            btn_txt2img_lcm.click(
                            fn=image_txt2img_lcm,
                            inputs=[
                                model_txt2img_lcm,
                                sampler_txt2img_lcm,
                                prompt_txt2img_lcm,
                                num_images_per_prompt_txt2img_lcm,
                                num_prompt_txt2img_lcm,
                                guidance_scale_txt2img_lcm,
                                lcm_origin_steps_txt2img_lcm,
                                num_inference_step_txt2img_lcm,
                                height_txt2img_lcm,
                                width_txt2img_lcm,
                                seed_txt2img_lcm,
                                use_gfpgan_txt2img_lcm,
                                nsfw_filter,
                                tkme_txt2img_lcm,
                                lora_model_txt2img_lcm,
                                lora_weight_txt2img_lcm,
                                lora_model2_txt2img_lcm,
                                lora_weight2_txt2img_lcm,
                                lora_model3_txt2img_lcm,
                                lora_weight3_txt2img_lcm,
                                lora_model4_txt2img_lcm,
                                lora_weight4_txt2img_lcm,
                                lora_model5_txt2img_lcm,
                                lora_weight5_txt2img_lcm,
                                txtinv_txt2img_lcm,
                            ],
                                outputs=[out_txt2img_lcm, gs_out_txt2img_lcm],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        txt2img_lcm_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        txt2img_lcm_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_lcm_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        txt2img_lcm_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_lcm_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        txt2img_lcm_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_lcm_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        txt2img_lcm_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_lcm_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        txt2img_lcm_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        txt2img_lcm_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_lcm_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        txt2img_lcm_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        txt2img_lcm_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        txt2img_lcm_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_lcm_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        txt2img_lcm_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_lcm_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2img_lcm_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2img_lcm_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        txt2img_lcm_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        txt2img_lcm_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_lcm_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_lcm_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_lcm_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_lcm_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_lcm_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_lcm_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2img_lcm_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2img_lcm_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_lcm_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_lcm_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_lcm_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_lcm_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_lcm_controlnet_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_lcm_faceid_ip_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_faceid_ip}")

# txt2img_mjm
                with gr.TabItem(f"{biniou_lang_tab_txt2img_mjm} 🖼️", id=24) as tab_txt2img_mjm:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2img_mjm}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_image_about_desc}<a href='https://huggingface.co/openskyml/midjourney-mini' target='_blank'>Midjourney-mini</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2img_mjm)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2img_mjm_about_instruct}
                                </br>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_mjm = gr.Dropdown(choices=model_list_txt2img_mjm, value=model_list_txt2img_mjm[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2img_mjm = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_txt2img_mjm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[4], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_mjm = gr.Slider(0.1, 20.0, step=0.1, value=7.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_txt2img_mjm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_txt2img_mjm = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_mjm = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2img_mjm = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                seed_txt2img_mjm = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_txt2img_mjm = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_txt2img_mjm = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_txt2img_mjm = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_txt2img_mjm = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_mjm = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2img_mjm = gr.Textbox(value="txt2img_mjm", visible=False, interactive=False)
                                del_ini_btn_txt2img_mjm = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2img_mjm.value) else False)
                                save_ini_btn_txt2img_mjm.click(
                                    fn=write_ini_txt2img_mjm,
                                    inputs=[
                                        module_name_txt2img_mjm,
                                        model_txt2img_mjm,
                                        num_inference_step_txt2img_mjm,
                                        sampler_txt2img_mjm,
                                        guidance_scale_txt2img_mjm,
                                        num_images_per_prompt_txt2img_mjm,
                                        num_prompt_txt2img_mjm,
                                        width_txt2img_mjm,
                                        height_txt2img_mjm,
                                        seed_txt2img_mjm,
                                        use_gfpgan_txt2img_mjm,
                                        tkme_txt2img_mjm,
                                        clipskip_txt2img_mjm,
                                        use_ays_txt2img_mjm,
                                        ]
                                    )
                                save_ini_btn_txt2img_mjm.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2img_mjm.click(fn=lambda: del_ini_btn_txt2img_mjm.update(interactive=True), outputs=del_ini_btn_txt2img_mjm)
                                del_ini_btn_txt2img_mjm.click(fn=lambda: del_ini(module_name_txt2img_mjm.value))
                                del_ini_btn_txt2img_mjm.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2img_mjm.click(fn=lambda: del_ini_btn_txt2img_mjm.update(interactive=False), outputs=del_ini_btn_txt2img_mjm)
                        if test_ini_exist(module_name_txt2img_mjm.value) :
                            with open(f".ini/{module_name_txt2img_mjm.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2img_mjm = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2img_mjm = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                        use_ays_txt2img_mjm.change(fn=change_ays_txt2img_mjm, inputs=use_ays_txt2img_mjm, outputs=[num_inference_step_txt2img_mjm, sampler_txt2img_mjm])
                        with gr.Column(scale=2):
                            out_txt2img_mjm = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                            )
                            gs_out_txt2img_mjm = gr.State()
                            sel_out_txt2img_mjm = gr.Number(precision=0, visible=False)
                            out_txt2img_mjm.select(get_select_index, None, sel_out_txt2img_mjm)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_txt2img_mjm = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_txt2img_mjm = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_txt2img_mjm.click(fn=zip_download_file_txt2img_mjm, inputs=out_txt2img_mjm, outputs=[download_file_txt2img_mjm, download_file_txt2img_mjm])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_mjm = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_txt2img_mjm_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2img_mjm_cancel.click(fn=initiate_stop_txt2img_mjm, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2img_mjm_clear_input = gr.ClearButton(components=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_txt2img_mjm_clear_output = gr.ClearButton(components=[out_txt2img_mjm, gs_out_txt2img_mjm], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2img_mjm.click(fn=hide_download_file_txt2img_mjm, inputs=None, outputs=download_file_txt2img_mjm)
                            btn_txt2img_mjm.click(
                            fn=image_txt2img_mjm,
                            inputs=[
                                model_txt2img_mjm,
                                sampler_txt2img_mjm,
                                prompt_txt2img_mjm,
                                negative_prompt_txt2img_mjm,
                                num_images_per_prompt_txt2img_mjm,
                                num_prompt_txt2img_mjm,
                                guidance_scale_txt2img_mjm,
                                num_inference_step_txt2img_mjm,
                                height_txt2img_mjm,
                                width_txt2img_mjm,
                                seed_txt2img_mjm,
                                use_gfpgan_txt2img_mjm,
                                nsfw_filter,
                                tkme_txt2img_mjm,
                                clipskip_txt2img_mjm,
                                use_ays_txt2img_mjm,
                            ],
                                outputs=[out_txt2img_mjm, gs_out_txt2img_mjm],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        txt2img_mjm_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        txt2img_mjm_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_mjm_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        txt2img_mjm_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_mjm_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        txt2img_mjm_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_mjm_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        txt2img_mjm_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_mjm_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        txt2img_mjm_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        txt2img_mjm_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_mjm_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        txt2img_mjm_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        txt2img_mjm_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        txt2img_mjm_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_mjm_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        txt2img_mjm_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_mjm_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2img_mjm_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2img_mjm_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        txt2img_mjm_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        txt2img_mjm_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_mjm_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_mjm_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_mjm_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_mjm_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_mjm_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_mjm_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2img_mjm_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2img_mjm_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_mjm_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_mjm_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_mjm_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_mjm_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_mjm_controlnet_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_mjm_faceid_ip_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_faceid_ip}")

# txt2img_paa
                with gr.TabItem(f"{biniou_lang_tab_txt2img_paa} 🖼️", id=25) as tab_txt2img_paa:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2img_paa}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_image_about_desc}<a href='https://pixart-alpha.github.io/' target='_blank'>PixArt-Alpha</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2img_paa)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2img_paa_about_instruct}
                                </br>
                                """
                            )                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_paa = gr.Dropdown(choices=model_list_txt2img_paa, value=model_list_txt2img_paa[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2img_paa = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_txt2img_paa = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_paa = gr.Slider(0.1, 20.0, step=0.1, value=7.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_txt2img_paa = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_txt2img_paa = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_paa = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2img_paa = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                seed_txt2img_paa = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)    
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_paa = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_txt2img_paa = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info, visible=False, interactive=False)
                        model_txt2img_paa.change(fn=change_model_type_txt2img_paa, inputs=model_txt2img_paa, outputs=[sampler_txt2img_paa, width_txt2img_paa, height_txt2img_paa, guidance_scale_txt2img_paa, num_inference_step_txt2img_paa])
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_paa = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2img_paa = gr.Textbox(value="txt2img_paa", visible=False, interactive=False)
                                del_ini_btn_txt2img_paa = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2img_paa.value) else False)
                                save_ini_btn_txt2img_paa.click(
                                    fn=write_ini_txt2img_paa, 
                                    inputs=[
                                        module_name_txt2img_paa, 
                                        model_txt2img_paa, 
                                        num_inference_step_txt2img_paa,
                                        sampler_txt2img_paa,
                                        guidance_scale_txt2img_paa,
                                        num_images_per_prompt_txt2img_paa,
                                        num_prompt_txt2img_paa,
                                        width_txt2img_paa,
                                        height_txt2img_paa,
                                        seed_txt2img_paa,
                                        use_gfpgan_txt2img_paa,
                                        tkme_txt2img_paa,
                                        ]
                                    )
                                save_ini_btn_txt2img_paa.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2img_paa.click(fn=lambda: del_ini_btn_txt2img_paa.update(interactive=True), outputs=del_ini_btn_txt2img_paa)
                                del_ini_btn_txt2img_paa.click(fn=lambda: del_ini(module_name_txt2img_paa.value))
                                del_ini_btn_txt2img_paa.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2img_paa.click(fn=lambda: del_ini_btn_txt2img_paa.update(interactive=False), outputs=del_ini_btn_txt2img_paa)
                        if test_ini_exist(module_name_txt2img_paa.value) :
                            with open(f".ini/{module_name_txt2img_paa.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2img_paa = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_tab_txt2img_paa_prompt_placeholder)
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_txt2img_paa = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                        with gr.Column(scale=2):
                            out_txt2img_paa = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                            )    
                            gs_out_txt2img_paa = gr.State()
                            sel_out_txt2img_paa = gr.Number(precision=0, visible=False)
                            out_txt2img_paa.select(get_select_index, None, sel_out_txt2img_paa)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_txt2img_paa = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_txt2img_paa = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_txt2img_paa.click(fn=zip_download_file_txt2img_paa, inputs=out_txt2img_paa, outputs=[download_file_txt2img_paa, download_file_txt2img_paa])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_paa = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_paa_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2img_paa_cancel.click(fn=initiate_stop_txt2img_paa, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2img_paa_clear_input = gr.ClearButton(components=[prompt_txt2img_paa, negative_prompt_txt2img_paa], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_txt2img_paa_clear_output = gr.ClearButton(components=[out_txt2img_paa, gs_out_txt2img_paa], value=f"{biniou_lang_clear_outputs} 🧹")   
                            btn_txt2img_paa.click(fn=hide_download_file_txt2img_paa, inputs=None, outputs=download_file_txt2img_paa)   
                            btn_txt2img_paa.click(
                            fn=image_txt2img_paa, 
                            inputs=[
                                model_txt2img_paa,
                                sampler_txt2img_paa,
                                prompt_txt2img_paa,
                                negative_prompt_txt2img_paa,
                                num_images_per_prompt_txt2img_paa,
                                num_prompt_txt2img_paa,
                                guidance_scale_txt2img_paa,
                                num_inference_step_txt2img_paa,
                                height_txt2img_paa,
                                width_txt2img_paa,
                                seed_txt2img_paa,
                                use_gfpgan_txt2img_paa,
                                nsfw_filter,
                                tkme_txt2img_paa,
                            ],
                                outputs=[out_txt2img_paa, gs_out_txt2img_paa],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        txt2img_paa_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        txt2img_paa_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_paa_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        txt2img_paa_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_paa_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        txt2img_paa_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_paa_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        txt2img_paa_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_paa_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        txt2img_paa_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        txt2img_paa_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_paa_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        txt2img_paa_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        txt2img_paa_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        txt2img_paa_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_paa_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        txt2img_paa_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2img_paa_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2img_paa_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2img_paa_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        txt2img_paa_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        txt2img_paa_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_paa_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_paa_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_paa_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_paa_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        txt2img_paa_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2img_paa_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2img_paa_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2img_paa_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value) 
                                        txt2img_paa_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        txt2img_paa_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        txt2img_paa_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        txt2img_paa_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        txt2img_paa_controlnet_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_faceid_ip}") 
                                        txt2img_paa_faceid_ip_both = gr.Button(f"🖼️ + ✍️️ >> {biniou_lang_tab_controlnet}") 
# img2img
                with gr.TabItem(f"{biniou_lang_tab_img2img} 🖌️", id=26) as tab_img2img:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_img2img}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_img2img_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                {biniou_lang_tab_img2img_about_desc_com}</br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_img_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_img2img)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_img2img_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}</br>
                                <b>{biniou_lang_about_lora}</b></br>
                                - {biniou_lang_tab_image_about_lora_inst1}</br>
                                </div>
                                """
                            )               
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2img = gr.Dropdown(choices=model_list_img2img, value=model_list_img2img[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_img2img = gr.Slider(2, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_img2img = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2img = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_img2img = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_img2img = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_img2img = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_img2img = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_img2img = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_img2img = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_img2img = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)    
                            with gr.Column():
                                clipskip_img2img = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_img2img = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2img = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_img2img = gr.Textbox(value="img2img", visible=False, interactive=False)
                                del_ini_btn_img2img = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_img2img.value) else False)
                                save_ini_btn_img2img.click(
                                    fn=write_ini_img2img,
                                    inputs=[
                                        module_name_img2img,
                                        model_img2img, 
                                        num_inference_step_img2img,
                                        sampler_img2img,
                                        guidance_scale_img2img,
                                        num_images_per_prompt_img2img,
                                        num_prompt_img2img,
                                        width_img2img,
                                        height_img2img,
                                        seed_img2img,
                                        use_gfpgan_img2img,
                                        tkme_img2img,
                                        clipskip_img2img,
                                        use_ays_img2img,
                                        ]
                                    )
                                save_ini_btn_img2img.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_img2img.click(fn=lambda: del_ini_btn_img2img.update(interactive=True), outputs=del_ini_btn_img2img)
                                del_ini_btn_img2img.click(fn=lambda: del_ini(module_name_img2img.value))
                                del_ini_btn_img2img.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_img2img.click(fn=lambda: del_ini_btn_img2img.update(interactive=False), outputs=del_ini_btn_img2img)
                        if test_ini_exist(module_name_img2img.value) :
                            with open(f".ini/{module_name_img2img.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                        with gr.Accordion(biniou_lang_lora_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_img2img = gr.Dropdown(choices=list(lora_model_list(model_img2img.value).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info)
                                with gr.Column():
                                    lora_weight_img2img = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info)
                            with gr.Row():
                                with gr.Column():
                                    lora_model2_img2img = gr.Dropdown(choices=list(lora_model_list(model_img2img.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight2_img2img = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model3_img2img = gr.Dropdown(choices=list(lora_model_list(model_img2img.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight3_img2img = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    lora_model4_img2img = gr.Dropdown(choices=list(lora_model_list(model_img2img.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight4_img2img = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model5_img2img = gr.Dropdown(choices=list(lora_model_list(model_img2img.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight5_img2img = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                        with gr.Accordion(biniou_lang_textinv_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_img2img = gr.Dropdown(choices=list(txtinv_list(model_img2img.value).keys()), value="", label=biniou_lang_textinv_label, info=biniou_lang_textinv_info)
                    with gr.Row():
                        with gr.Column():
                            img_img2img = gr.Image(label=biniou_lang_img_input_label, height=400, type="filepath")
                            with gr.Row():
                                source_type_img2img = gr.Radio(choices=["image", "sketch"], value="image", label=biniou_lang_input_type_label, info=biniou_lang_input_type_info)
                                img_img2img.change(image_upload_event, inputs=img_img2img, outputs=[width_img2img, height_img2img])
                                source_type_img2img.change(fn=change_source_type_img2img, inputs=source_type_img2img, outputs=img_img2img)
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_img2img = gr.Slider(0.01, 1.0, step=0.01, value=0.75, label=biniou_lang_image_denoising_label, info=biniou_lang_image_denoising_info)
                            with gr.Row():
                                with gr.Column():
                                    prompt_img2img = gr.Textbox(lines=5, max_lines=5, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_img2img = gr.Textbox(lines=5, max_lines=5, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                        model_img2img.change(
                            fn=change_model_type_img2img, 
                            inputs=[model_img2img],
                            outputs=[
                                sampler_img2img,
                                width_img2img,
                                height_img2img,
                                num_inference_step_img2img,
                                guidance_scale_img2img,
                                lora_model_img2img,
                                txtinv_img2img,
                                negative_prompt_img2img,
                            ]
                        )
                        model_img2img.change(fn=change_model_type_img2img_alternate2, inputs=[model_img2img],outputs=[lora_model2_img2img])
                        model_img2img.change(fn=change_model_type_img2img_alternate3, inputs=[model_img2img],outputs=[lora_model3_img2img])
                        model_img2img.change(fn=change_model_type_img2img_alternate4, inputs=[model_img2img],outputs=[lora_model4_img2img])
                        model_img2img.change(fn=change_model_type_img2img_alternate5, inputs=[model_img2img],outputs=[lora_model5_img2img])
                        lora_model_img2img.change(fn=change_lora_model_img2img, inputs=[model_img2img, lora_model_img2img, prompt_img2img, num_inference_step_img2img, guidance_scale_img2img, sampler_img2img], outputs=[prompt_img2img, num_inference_step_img2img, guidance_scale_img2img, sampler_img2img])
                        lora_model2_img2img.change(fn=change_lora_model2_img2img, inputs=[model_img2img, lora_model2_img2img, prompt_img2img], outputs=[prompt_img2img])
                        lora_model3_img2img.change(fn=change_lora_model3_img2img, inputs=[model_img2img, lora_model3_img2img, prompt_img2img], outputs=[prompt_img2img])
                        lora_model4_img2img.change(fn=change_lora_model4_img2img, inputs=[model_img2img, lora_model4_img2img, prompt_img2img], outputs=[prompt_img2img])
                        lora_model5_img2img.change(fn=change_lora_model5_img2img, inputs=[model_img2img, lora_model5_img2img, prompt_img2img], outputs=[prompt_img2img])
                        txtinv_img2img.change(fn=change_txtinv_img2img, inputs=[model_img2img, txtinv_img2img, prompt_img2img, negative_prompt_img2img], outputs=[prompt_img2img, negative_prompt_img2img])
                        denoising_strength_img2img.change(check_steps_strength, [num_inference_step_img2img, denoising_strength_img2img, model_img2img, lora_model_img2img], [num_inference_step_img2img])
                        use_ays_img2img.change(fn=change_ays_img2img, inputs=use_ays_img2img, outputs=[num_inference_step_img2img, sampler_img2img])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                            
                                    out_img2img = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_i2i",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                )
                                gs_out_img2img = gr.State()
                                sel_out_img2img = gr.Number(precision=0, visible=False)
                                out_img2img.select(get_select_index, None, sel_out_img2img)
                                with gr.Row():
                                    with gr.Column():
                                        download_btn_img2img = gr.Button(f"{biniou_lang_image_zip} 💾")
                                    with gr.Column():
                                        download_file_img2img = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                        download_btn_img2img.click(fn=zip_download_file_img2img, inputs=out_img2img, outputs=[download_file_img2img, download_file_img2img])
                    with gr.Row():
                        with gr.Column():
                            btn_img2img = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_img2img_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_img2img_cancel.click(fn=initiate_stop_img2img, inputs=None, outputs=None)
                        with gr.Column():
                            btn_img2img_clear_input = gr.ClearButton(components=[img_img2img, prompt_img2img, negative_prompt_img2img], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_img2img_clear_output = gr.ClearButton(components=[out_img2img, gs_out_img2img], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_img2img.click(fn=hide_download_file_img2img, inputs=None, outputs=download_file_img2img)
                            btn_img2img.click(
                                fn=image_img2img,
                                inputs=[
                                    model_img2img,
                                    sampler_img2img,
                                    img_img2img,
                                    prompt_img2img,
                                    negative_prompt_img2img,
                                    num_images_per_prompt_img2img,
                                    num_prompt_img2img,
                                    guidance_scale_img2img,
                                    denoising_strength_img2img,
                                    num_inference_step_img2img,
                                    height_img2img,
                                    width_img2img,
                                    seed_img2img,
                                    source_type_img2img,
                                    use_gfpgan_img2img,
                                    nsfw_filter,
                                    tkme_img2img,
                                    clipskip_img2img,
                                    use_ays_img2img,
                                    lora_model_img2img,
                                    lora_weight_img2img,
                                    lora_model2_img2img,
                                    lora_weight2_img2img,
                                    lora_model3_img2img,
                                    lora_weight3_img2img,
                                    lora_model4_img2img,
                                    lora_weight4_img2img,
                                    lora_model5_img2img,
                                    lora_weight5_img2img,
                                    txtinv_img2img,
                                ],
                                outputs=[out_img2img, gs_out_img2img], 
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        img2img_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        img2img_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2img_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        img2img_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        img2img_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        img2img_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        img2img_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        img2img_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        img2img_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        img2img_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        img2img_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        img2img_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        img2img_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        img2img_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        img2img_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        img2img_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        img2img_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2img_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        img2img_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        img2img_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        img2img_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        img2img_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        img2img_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        img2img_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        img2img_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        img2img_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2img_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        img2img_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        img2img_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")
                                        img2img_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")

# img2img_ip
                with gr.TabItem(f"{biniou_lang_tab_img2img_ip} 🖌️", id=27) as tab_img2img_ip:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_img2img_ip}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_img2img_ip_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a>, <a href='https://ip-adapter.github.io/' target='_blank'>IP-Adapter</a>, <a href='https://huggingface.co/ostris/ip-composition-adapter' target='_blank'>ostris/ip-composition-adapter</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_img2img_ip_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_img2img_ip)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_img2img_ip_about_instruct}
                                <b>{biniou_lang_about_lora}</b></br>
                                - {biniou_lang_tab_image_about_lora_inst1}</br>
                                </br>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2img_ip = gr.Dropdown(choices=model_list_img2img_ip, value=model_list_img2img_ip[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_img2img_ip = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_img2img_ip = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2img_ip = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_img2img_ip = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_img2img_ip = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_img2img_ip = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_img2img_ip = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_img2img_ip = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_img2img_ip = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_img2img_ip = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_img2img_ip = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_img2img_ip = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2img_ip = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_img2img_ip = gr.Textbox(value="img2img_ip", visible=False, interactive=False)
                                del_ini_btn_img2img_ip = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_img2img_ip.value) else False)
                                save_ini_btn_img2img_ip.click(
                                    fn=write_ini_img2img_ip,
                                    inputs=[
                                        module_name_img2img_ip,
                                        model_img2img_ip,
                                        num_inference_step_img2img_ip,
                                        sampler_img2img_ip,
                                        guidance_scale_img2img_ip,
                                        num_images_per_prompt_img2img_ip,
                                        num_prompt_img2img_ip,
                                        width_img2img_ip,
                                        height_img2img_ip,
                                        seed_img2img_ip,
                                        use_gfpgan_img2img_ip,
                                        tkme_img2img_ip,
                                        clipskip_img2img_ip,
                                        use_ays_img2img_ip,
                                        ]
                                    )
                                save_ini_btn_img2img_ip.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_img2img_ip.click(fn=lambda: del_ini_btn_img2img_ip.update(interactive=True), outputs=del_ini_btn_img2img_ip)
                                del_ini_btn_img2img_ip.click(fn=lambda: del_ini(module_name_img2img_ip.value))
                                del_ini_btn_img2img_ip.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_img2img_ip.click(fn=lambda: del_ini_btn_img2img_ip.update(interactive=False), outputs=del_ini_btn_img2img_ip)
                        if test_ini_exist(module_name_img2img_ip.value) :
                            with open(f".ini/{module_name_img2img_ip.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                        with gr.Accordion(biniou_lang_lora_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_img2img_ip = gr.Dropdown(choices=list(lora_model_list(model_img2img_ip.value).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info)
                                with gr.Column():
                                    lora_weight_img2img_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info)
                            with gr.Row():
                                with gr.Column():
                                    lora_model2_img2img_ip = gr.Dropdown(choices=list(lora_model_list(model_img2img_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight2_img2img_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model3_img2img_ip = gr.Dropdown(choices=list(lora_model_list(model_img2img_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight3_img2img_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    lora_model4_img2img_ip = gr.Dropdown(choices=list(lora_model_list(model_img2img_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight4_img2img_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model5_img2img_ip = gr.Dropdown(choices=list(lora_model_list(model_img2img_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight5_img2img_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                        with gr.Accordion(biniou_lang_textinv_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_img2img_ip = gr.Dropdown(choices=list(txtinv_list(model_img2img_ip.value).keys()), value="", label=biniou_lang_textinv_label, info=biniou_lang_textinv_info)
                    with gr.Row():
                        with gr.Column():
                            img_img2img_ip = gr.Image(label=biniou_lang_img_input_label, height=400, type="filepath", visible=True)
                            img_img2img_ip.change(image_upload_event, inputs=img_img2img_ip, outputs=[width_img2img_ip, height_img2img_ip])
                            source_type_img2img_ip = gr.Radio(choices=["standard", "composition"], value="standard", label=biniou_lang_tab_img2img_ip_src_type_label, info=biniou_lang_tab_img2img_ip_src_type_info)
                        with gr.Column():
                            img_ipa_img2img_ip = gr.Image(label=biniou_lang_tab_img2img_ip_img_ipa, height=400, type="filepath")
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_img2img_ip = gr.Slider(0.01, 1.0, step=0.01, value=0.6, label=biniou_lang_image_denoising_label, info=biniou_lang_image_denoising_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    prompt_img2img_ip = gr.Textbox(lines=2, max_lines=2, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_tab_img2img_ip_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_img2img_ip = gr.Textbox(lines=2, max_lines=2, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_tab_img2img_ip_negprompt_placeholder)
                        denoising_strength_img2img_ip.change(check_steps_strength, [num_inference_step_img2img_ip, denoising_strength_img2img_ip, model_img2img_ip, lora_model_img2img_ip], [num_inference_step_img2img_ip])
                        model_img2img_ip.change(
                            fn=change_model_type_img2img_ip,
                            inputs=[
                                model_img2img_ip,
                                source_type_img2img_ip
                            ],
                            outputs=[
                                sampler_img2img_ip,
                                width_img2img_ip,
                                height_img2img_ip,
                                num_inference_step_img2img_ip,
                                guidance_scale_img2img_ip,
                                lora_model_img2img_ip,
                                txtinv_img2img_ip,
                                negative_prompt_img2img_ip,
                                source_type_img2img_ip,
                                img_img2img_ip,
                            ]
                        )
                        model_img2img_ip.change(fn=change_model_type_img2img_ip_alternate2, inputs=[model_img2img_ip],outputs=[lora_model2_img2img_ip])
                        model_img2img_ip.change(fn=change_model_type_img2img_ip_alternate3, inputs=[model_img2img_ip],outputs=[lora_model3_img2img_ip])
                        model_img2img_ip.change(fn=change_model_type_img2img_ip_alternate4, inputs=[model_img2img_ip],outputs=[lora_model4_img2img_ip])
                        model_img2img_ip.change(fn=change_model_type_img2img_ip_alternate5, inputs=[model_img2img_ip],outputs=[lora_model5_img2img_ip])
                        model_img2img_ip.change(image_upload_event, inputs=img_img2img_ip, outputs=[width_img2img_ip, height_img2img_ip])
                        lora_model_img2img_ip.change(fn=change_lora_model_img2img_ip, inputs=[model_img2img_ip, lora_model_img2img_ip, prompt_img2img_ip, num_inference_step_img2img_ip, guidance_scale_img2img_ip, sampler_img2img_ip], outputs=[prompt_img2img_ip, num_inference_step_img2img_ip, guidance_scale_img2img_ip, sampler_img2img_ip])
                        lora_model2_img2img_ip.change(fn=change_lora_model2_img2img_ip, inputs=[model_img2img_ip, lora_model2_img2img_ip, prompt_img2img_ip], outputs=[prompt_img2img_ip])
                        lora_model3_img2img_ip.change(fn=change_lora_model3_img2img_ip, inputs=[model_img2img_ip, lora_model3_img2img_ip, prompt_img2img_ip], outputs=[prompt_img2img_ip])
                        lora_model4_img2img_ip.change(fn=change_lora_model4_img2img_ip, inputs=[model_img2img_ip, lora_model4_img2img_ip, prompt_img2img_ip], outputs=[prompt_img2img_ip])
                        lora_model5_img2img_ip.change(fn=change_lora_model5_img2img_ip, inputs=[model_img2img_ip, lora_model5_img2img_ip, prompt_img2img_ip], outputs=[prompt_img2img_ip])
                        txtinv_img2img_ip.change(fn=change_txtinv_img2img_ip, inputs=[model_img2img_ip, txtinv_img2img_ip, prompt_img2img_ip, negative_prompt_img2img_ip], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip])
                        use_ays_img2img_ip.change(fn=change_ays_img2img_ip, inputs=use_ays_img2img_ip, outputs=[num_inference_step_img2img_ip, sampler_img2img_ip])
                        source_type_img2img_ip.change(change_source_type_img2img_ip, source_type_img2img_ip, [img_img2img_ip, denoising_strength_img2img_ip])
                        source_type_img2img_ip.change(
                            fn=change_model_type_img2img_ip,
                            inputs=[
                                model_img2img_ip,
                                source_type_img2img_ip
                            ],
                            outputs=[
                                sampler_img2img_ip,
                                width_img2img_ip,
                                height_img2img_ip,
                                num_inference_step_img2img_ip,
                                guidance_scale_img2img_ip,
                                lora_model_img2img_ip,
                                txtinv_img2img_ip,
                                negative_prompt_img2img_ip,
                                source_type_img2img_ip
                            ]
                        )
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_img2img_ip = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_ipa",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                )
                                gs_out_img2img_ip = gr.State()
                                sel_out_img2img_ip = gr.Number(precision=0, visible=False)
                                out_img2img_ip.select(get_select_index, None, sel_out_img2img_ip)
                                with gr.Row():
                                    with gr.Column():
                                        download_btn_img2img_ip = gr.Button(f"{biniou_lang_image_zip} 💾")
                                    with gr.Column():
                                        download_file_img2img_ip = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                        download_btn_img2img_ip.click(fn=zip_download_file_img2img_ip, inputs=out_img2img_ip, outputs=[download_file_img2img_ip, download_file_img2img_ip])
                    with gr.Row():
                        with gr.Column():
                            btn_img2img_ip = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_img2img_ip_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_img2img_ip_cancel.click(fn=initiate_stop_img2img_ip, inputs=None, outputs=None)
                        with gr.Column():
                            btn_img2img_ip_clear_input = gr.ClearButton(components=[img_img2img_ip, img_ipa_img2img_ip, prompt_img2img_ip, negative_prompt_img2img_ip], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_img2img_ip_clear_output = gr.ClearButton(components=[out_img2img_ip, gs_out_img2img_ip], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_img2img_ip.click(fn=hide_download_file_img2img_ip, inputs=None, outputs=download_file_img2img_ip)
                            btn_img2img_ip.click(
                                fn=image_img2img_ip,
                                inputs=[
                                    model_img2img_ip,
                                    sampler_img2img_ip,
                                    img_img2img_ip,
                                    source_type_img2img_ip,
                                    img_ipa_img2img_ip,
                                    prompt_img2img_ip,
                                    negative_prompt_img2img_ip,
                                    num_images_per_prompt_img2img_ip,
                                    num_prompt_img2img_ip,
                                    guidance_scale_img2img_ip,
                                    denoising_strength_img2img_ip,
                                    num_inference_step_img2img_ip,
                                    height_img2img_ip,
                                    width_img2img_ip,
                                    seed_img2img_ip,
                                    use_gfpgan_img2img_ip,
                                    nsfw_filter,
                                    tkme_img2img_ip,
                                    clipskip_img2img_ip,
                                    use_ays_img2img_ip,
                                    lora_model_img2img_ip,
                                    lora_weight_img2img_ip,
                                    lora_model2_img2img_ip,
                                    lora_weight2_img2img_ip,
                                    lora_model3_img2img_ip,
                                    lora_weight3_img2img_ip,
                                    lora_model4_img2img_ip,
                                    lora_weight4_img2img_ip,
                                    lora_model5_img2img_ip,
                                    lora_weight5_img2img_ip,
                                    txtinv_img2img_ip,
                                ],
                                outputs=[out_img2img_ip, gs_out_img2img_ip],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        img2img_ip_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        img2img_ip_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2img_ip_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        img2img_ip_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        img2img_ip_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        img2img_ip_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        img2img_ip_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        img2img_ip_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        img2img_ip_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        img2img_ip_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        img2img_ip_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        img2img_ip_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        img2img_ip_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        img2img_ip_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        img2img_ip_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        img2img_ip_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        img2img_ip_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2img_ip_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        img2img_ip_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        img2img_ip_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        img2img_ip_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        img2img_ip_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        img2img_ip_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        img2img_ip_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        img2img_ip_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        img2img_ip_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2img_ip_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        img2img_ip_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        img2img_ip_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")
                                        img2img_ip_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")

# img2var
                if ram_size() >= 16 :
                    titletab_img2var = f"{biniou_lang_tab_img2var} 🖼️"
                else :
                    titletab_img2var = f"{biniou_lang_tab_img2var} ⛔"

                with gr.TabItem(titletab_img2var, id=28) as tab_img2var: 
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_img2var}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_img2var_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_image}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_img2var)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_img2var_about_instruct}
                                </br>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2var = gr.Dropdown(choices=model_list_img2var, value=model_list_img2var[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_img2var = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_img2var = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2var = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_img2var = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_img2var = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_img2var = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_img2var = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_img2var = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_img2var = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_img2var = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2var = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_img2var = gr.Textbox(value="img2var", visible=False, interactive=False)
                                del_ini_btn_img2var = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_img2var.value) else False)
                                save_ini_btn_img2var.click(
                                    fn=write_ini_img2var,
                                    inputs=[
                                        module_name_img2var,
                                        model_img2var,
                                        num_inference_step_img2var,
                                        sampler_img2var,
                                        guidance_scale_img2var,
                                        num_images_per_prompt_img2var,
                                        num_prompt_img2var,
                                        width_img2var,
                                        height_img2var,
                                        seed_img2var,
                                        use_gfpgan_img2var,
                                        tkme_img2var,
                                        ]
                                    )
                                save_ini_btn_img2var.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_img2var.click(fn=lambda: del_ini_btn_img2var.update(interactive=True), outputs=del_ini_btn_img2var)
                                del_ini_btn_img2var.click(fn=lambda: del_ini(module_name_img2var.value))
                                del_ini_btn_img2var.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_img2var.click(fn=lambda: del_ini_btn_img2var.update(interactive=False), outputs=del_ini_btn_img2var)
                        if test_ini_exist(module_name_img2var.value) :
                            with open(f".ini/{module_name_img2var.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            img_img2var = gr.Image(label=biniou_lang_img_input_label, height=400, type="filepath")
                            img_img2var.change(image_upload_event, inputs=img_img2var, outputs=[width_img2var, height_img2var])
                        with gr.Column():
                            with gr.Row():
                                out_img2var = gr.Gallery(
                                    label=biniou_lang_image_gallery_label,
                                    show_label=True,
                                    elem_id="gallery_i2v",
                                    columns=2,
                                    height=400,
                                    preview=True,
                                )
                                gs_out_img2var = gr.State()
                                sel_out_img2var = gr.Number(precision=0, visible=False)
                                out_img2var.select(get_select_index, None, sel_out_img2var)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_img2var = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_img2var = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_img2var.click(fn=zip_download_file_img2var, inputs=out_img2var, outputs=[download_file_img2var, download_file_img2var])
                    with gr.Row():
                        with gr.Column():
                            btn_img2var = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_img2var_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_img2var_cancel.click(fn=initiate_stop_img2var, inputs=None, outputs=None)
                        with gr.Column():
                            btn_img2var_clear_input = gr.ClearButton(components=[img_img2var], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_img2var_clear_output = gr.ClearButton(components=[out_img2var, gs_out_img2var], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_img2var.click(fn=hide_download_file_img2var, inputs=None, outputs=download_file_img2var)
                            btn_img2var.click(
                                fn=image_img2var,
                                inputs=[
                                    model_img2var,
                                    sampler_img2var,
                                    img_img2var,
                                    num_images_per_prompt_img2var,
                                    num_prompt_img2var,
                                    guidance_scale_img2var,
                                    num_inference_step_img2var,
                                    height_img2var,
                                    width_img2var,
                                    seed_img2var,
                                    use_gfpgan_img2var,
                                    nsfw_filter,
                                    tkme_img2var,
                                ],
                                outputs=[out_img2var, gs_out_img2var], 
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        img2var_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        img2var_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        img2var_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        img2var_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        img2var_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        img2var_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        img2var_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        img2var_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        img2var_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        img2var_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        img2var_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        img2var_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        img2var_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        img2var_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        img2var_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        img2var_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        img2var_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                       

# pix2pix    
                with gr.TabItem(f"{biniou_lang_tab_pix2pix} 🖌️", id=29) as tab_pix2pix:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_pix2pix}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_pix2pix_about_desc}<a href='https://github.com/timothybrooks/instruct-pix2pix' target='_blank'>Instructpix2pix</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_img_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_pix2pix)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_pix2pix_about_instruct}
                                <b>{biniou_lang_about_examples}</b><a href='https://www.timothybrooks.com/instruct-pix2pix/' target='_blank'>InstructPix2Pix : Learning to Follow Image Editing Instructions</a>
                                </div>
                                """
                            )                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_pix2pix = gr.Dropdown(choices=model_list_pix2pix, value=model_list_pix2pix[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_pix2pix = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_pix2pix = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_pix2pix = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                image_guidance_scale_pix2pix = gr.Slider(0.0, 10.0, step=0.1, value=1.5, label=biniou_lang_imgcfg_label, info=biniou_lang_tab_pix2pix_imgcfg_info)
                            with gr.Column():
                                num_images_per_prompt_pix2pix = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_pix2pix = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_pix2pix = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_pix2pix = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_pix2pix = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_pix2pix = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_pix2pix = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        model_pix2pix.change(fn=change_model_type_pix2pix, inputs=model_pix2pix, outputs=[sampler_pix2pix, width_pix2pix, height_pix2pix, guidance_scale_pix2pix, image_guidance_scale_pix2pix, num_inference_step_pix2pix])
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_pix2pix = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_pix2pix = gr.Textbox(value="pix2pix", visible=False, interactive=False)
                                del_ini_btn_pix2pix = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_pix2pix.value) else False)
                                save_ini_btn_pix2pix.click(
                                    fn=write_ini_pix2pix, 
                                    inputs=[
                                        module_name_pix2pix, 
                                        model_pix2pix, 
                                        num_inference_step_pix2pix,
                                        sampler_pix2pix,
                                        guidance_scale_pix2pix,
                                        image_guidance_scale_pix2pix, 
                                        num_images_per_prompt_pix2pix,
                                        num_prompt_pix2pix,
                                        width_pix2pix,
                                        height_pix2pix,
                                        seed_pix2pix,
                                        use_gfpgan_pix2pix,
                                        tkme_pix2pix,
                                        ]
                                    )
                                save_ini_btn_pix2pix.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_pix2pix.click(fn=lambda: del_ini_btn_pix2pix.update(interactive=True), outputs=del_ini_btn_pix2pix)
                                del_ini_btn_pix2pix.click(fn=lambda: del_ini(module_name_pix2pix.value))
                                del_ini_btn_pix2pix.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_pix2pix.click(fn=lambda: del_ini_btn_pix2pix.update(interactive=False), outputs=del_ini_btn_pix2pix)
                        if test_ini_exist(module_name_pix2pix.value) :
                            with open(f".ini/{module_name_pix2pix.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                             img_pix2pix = gr.Image(label=biniou_lang_img_input_label, height=400, type="filepath")
                             img_pix2pix.change(image_upload_event, inputs=img_pix2pix, outputs=[width_pix2pix, height_pix2pix])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_pix2pix = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_tab_pix2pix_prompt_info, placeholder=biniou_lang_tab_pix2pix_prompt_placeholder)
                                with gr.Column():
                                    negative_prompt_pix2pix = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_tab_pix2pix_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_pix2pix = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_p2p",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                    )
                                    gs_out_pix2pix = gr.State()
                                    sel_out_pix2pix = gr.Number(precision=0, visible=False)
                                    out_pix2pix.select(get_select_index, None, sel_out_pix2pix)
                                    with gr.Row():
                                        with gr.Column():
                                            download_btn_pix2pix = gr.Button(f"{biniou_lang_image_zip} 💾")
                                        with gr.Column():
                                            download_file_pix2pix = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                            download_btn_pix2pix.click(fn=zip_download_file_pix2pix, inputs=out_pix2pix, outputs=[download_file_pix2pix, download_file_pix2pix])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_pix2pix = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_pix2pix_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_pix2pix_cancel.click(fn=initiate_stop_pix2pix, inputs=None, outputs=None)
                        with gr.Column():
                            btn_pix2pix_clear_input = gr.ClearButton(components=[img_pix2pix, prompt_pix2pix, negative_prompt_pix2pix], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_pix2pix_clear_output = gr.ClearButton(components=[out_pix2pix, gs_out_pix2pix], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_pix2pix.click(fn=hide_download_file_pix2pix, inputs=None, outputs=download_file_pix2pix)
                            btn_pix2pix.click(
                                fn=image_pix2pix,
                                inputs=[
                                    model_pix2pix,
                                    sampler_pix2pix,
                                    img_pix2pix,
                                    prompt_pix2pix,
                                    negative_prompt_pix2pix,
                                    num_images_per_prompt_pix2pix,
                                    num_prompt_pix2pix,
                                    guidance_scale_pix2pix,
                                    image_guidance_scale_pix2pix,
                                    num_inference_step_pix2pix,
                                    height_pix2pix,
                                    width_pix2pix,
                                    seed_pix2pix,
                                    use_gfpgan_pix2pix,
                                    nsfw_filter,
                                    tkme_pix2pix,
                                ],
                                outputs=[out_pix2pix, gs_out_pix2pix],
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        pix2pix_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        pix2pix_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        pix2pix_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        pix2pix_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        pix2pix_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        pix2pix_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        pix2pix_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        pix2pix_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        pix2pix_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        pix2pix_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        pix2pix_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        pix2pix_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        pix2pix_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        pix2pix_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        pix2pix_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        pix2pix_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        pix2pix_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        pix2pix_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        pix2pix_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        pix2pix_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        pix2pix_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        pix2pix_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        pix2pix_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        pix2pix_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        pix2pix_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        pix2pix_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        pix2pix_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        pix2pix_vid2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_vid2vid_ze}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        pix2pix_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        pix2pix_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        pix2pix_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        pix2pix_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")
                                        pix2pix_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")
# magicmix
                with gr.TabItem(f"{biniou_lang_tab_magicmix} 🖌️", id=291) as tab_magicmix:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_magicmix}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_magicmix_about_desc}<a href='https://magicmix.github.io/' target='_blank'>MagicMix</a>, <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_magicmix_about_input_img_prompt}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_magicmix)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_magicmix_about_instruct}
                                <b>{biniou_lang_about_examples}</b><a href ='https://magicmix.github.io/' target='_blank'>MagicMix: Semantic Mixing with Diffusion Models</a>
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_magicmix = gr.Dropdown(choices=model_list_magicmix, value=model_list_magicmix[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_magicmix = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_magicmix = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[1], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_magicmix = gr.Slider(0.0, 20.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                kmin_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=0.3, label=biniou_lang_tab_magicmix_kmin_label, info=biniou_lang_tab_magicmix_kmin_info)
                            with gr.Column():
                                kmax_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label=biniou_lang_tab_magicmix_kmax_label, info=biniou_lang_tab_magicmix_kmax_info)
                        with gr.Row():
                            with gr.Column():
                                num_prompt_magicmix = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                            with gr.Column():
                                seed_magicmix = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_magicmix = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_magicmix = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_magicmix = gr.Textbox(value="magicmix", visible=False, interactive=False)
                                del_ini_btn_magicmix = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_magicmix.value) else False)
                                save_ini_btn_magicmix.click(
                                    fn=write_ini_magicmix, 
                                    inputs=[
                                        module_name_magicmix, 
                                        model_magicmix, 
                                        num_inference_step_magicmix,
                                        sampler_magicmix,
                                        guidance_scale_magicmix,
                                        kmin_magicmix,
                                        kmax_magicmix,
                                        num_prompt_magicmix,
                                        seed_magicmix,
                                        use_gfpgan_magicmix,
                                        tkme_magicmix,
                                        ]
                                    )
                                save_ini_btn_magicmix.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_magicmix.click(fn=lambda: del_ini_btn_magicmix.update(interactive=True), outputs=del_ini_btn_magicmix)
                                del_ini_btn_magicmix.click(fn=lambda: del_ini(module_name_magicmix.value))
                                del_ini_btn_magicmix.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_magicmix.click(fn=lambda: del_ini_btn_magicmix.update(interactive=False), outputs=del_ini_btn_magicmix)
                        if test_ini_exist(module_name_magicmix.value):
                            with open(f".ini/{module_name_magicmix.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                             img_magicmix = gr.Image(label=biniou_lang_img_input_label, height=400, type="filepath")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    mix_factor_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=0.5, label=biniou_lang_tab_magicmix_factor_label, info=biniou_lang_tab_magicmix_factor_info)
                            with gr.Row(): 
                                with gr.Column():
                                    prompt_magicmix = gr.Textbox(lines=9, max_lines=9, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_tab_magicmix_prompt_info, placeholder=biniou_lang_tab_magicmix_prompt_placeholder)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_magicmix = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_p2p",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                    )
                                    gs_out_magicmix = gr.State()
                                    sel_out_magicmix = gr.Number(precision=0, visible=False)
                                    out_magicmix.select(get_select_index, None, sel_out_magicmix)
                                    with gr.Row():
                                        with gr.Column():
                                            download_btn_magicmix = gr.Button(f"{biniou_lang_image_zip} 💾")
                                        with gr.Column():
                                            download_file_magicmix = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                            download_btn_magicmix.click(fn=zip_download_file_magicmix, inputs=out_magicmix, outputs=[download_file_magicmix, download_file_magicmix])
                    with gr.Row():
                        with gr.Column():
                            btn_magicmix = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_magicmix_clear_input = gr.ClearButton(components=[img_magicmix, prompt_magicmix], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_magicmix_clear_output = gr.ClearButton(components=[out_magicmix, gs_out_magicmix], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_magicmix.click(fn=hide_download_file_magicmix, inputs=None, outputs=download_file_magicmix)
                            btn_magicmix.click(
                                fn=image_magicmix,
                                inputs=[
                                    model_magicmix,
                                    sampler_magicmix,
                                    num_inference_step_magicmix,
                                    guidance_scale_magicmix,
                                    kmin_magicmix,
                                    kmax_magicmix,
                                    num_prompt_magicmix,
                                    seed_magicmix,
                                    img_magicmix,
                                    prompt_magicmix,
                                    mix_factor_magicmix,
                                    use_gfpgan_magicmix, 
                                    nsfw_filter, 
                                    tkme_magicmix,
                                ],
                                outputs=[out_magicmix, gs_out_magicmix],
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        magicmix_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        magicmix_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        magicmix_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        magicmix_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        magicmix_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        magicmix_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        magicmix_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        magicmix_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        magicmix_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        magicmix_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        magicmix_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        magicmix_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        magicmix_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        magicmix_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        magicmix_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        magicmix_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        magicmix_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# inpaint
                with gr.TabItem(f"{biniou_lang_tab_inpaint} 🖌️", id=292) as tab_inpaint:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_inpaint}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_inpaint_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_inpaint_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_inpaint)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_inpaint_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}
                                </div>
                                """
                            )                   
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_inpaint = gr.Dropdown(choices=model_list_inpaint, value=model_list_inpaint[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_inpaint = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_inpaint = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_inpaint = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_inpaint= gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_inpaint = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_inpaint = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_inpaint = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_inpaint = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_inpaint = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_inpaint = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_inpaint = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_inpaint = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_inpaint = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_inpaint = gr.Textbox(value="inpaint", visible=False, interactive=False)
                                del_ini_btn_inpaint = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_inpaint.value) else False)
                                save_ini_btn_inpaint.click(
                                    fn=write_ini_inpaint, 
                                    inputs=[
                                        module_name_inpaint, 
                                        model_inpaint, 
                                        num_inference_step_inpaint,
                                        sampler_inpaint,
                                        guidance_scale_inpaint,
                                        num_images_per_prompt_inpaint,
                                        num_prompt_inpaint,
                                        width_inpaint,
                                        height_inpaint,
                                        seed_inpaint,
                                        use_gfpgan_inpaint,
                                        tkme_inpaint,
                                        clipskip_inpaint,
                                        use_ays_inpaint,
                                        ]
                                    )
                                save_ini_btn_inpaint.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_inpaint.click(fn=lambda: del_ini_btn_inpaint.update(interactive=True), outputs=del_ini_btn_inpaint)
                                del_ini_btn_inpaint.click(fn=lambda: del_ini(module_name_inpaint.value))
                                del_ini_btn_inpaint.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_inpaint.click(fn=lambda: del_ini_btn_inpaint.update(interactive=False), outputs=del_ini_btn_inpaint)
                        if test_ini_exist(module_name_inpaint.value) :
                            with open(f".ini/{module_name_inpaint.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column(scale=2):
                             rotation_img_inpaint = gr.Number(value=0, visible=False)
                             img_inpaint = gr.Image(label=biniou_lang_img_input_label, type="pil", height=400, tool="sketch")
                             img_inpaint.upload(image_upload_event_inpaint_c, inputs=[img_inpaint, model_inpaint], outputs=[width_inpaint, height_inpaint, img_inpaint, rotation_img_inpaint], preprocess=False)
                             gs_img_inpaint = gr.Image(type="pil", visible=False)
                             gs_img_inpaint.change(image_upload_event_inpaint_b, inputs=gs_img_inpaint, outputs=[width_inpaint, height_inpaint], preprocess=False)
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_inpaint = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label=biniou_lang_image_denoising_label, info=biniou_lang_image_denoising_info)                                
                            with gr.Row():
                                with gr.Column():
                                    prompt_inpaint = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                                with gr.Column():
                                    negative_prompt_inpaint = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                            use_ays_inpaint.change(fn=change_ays_inpaint, inputs=use_ays_inpaint, outputs=[num_inference_step_inpaint, sampler_inpaint])
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    out_inpaint = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_inpaint",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                    )
                                    gs_out_inpaint = gr.State()
                                    sel_out_inpaint = gr.Number(precision=0, visible=False)
                                    out_inpaint.select(get_select_index, None, sel_out_inpaint)
                                    with gr.Row():
                                        with gr.Column():
                                            download_btn_inpaint = gr.Button(f"{biniou_lang_image_zip} 💾")
                                        with gr.Column():
                                            download_file_inpaint = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                            download_btn_inpaint.click(fn=zip_download_file_inpaint, inputs=out_inpaint, outputs=[download_file_inpaint, download_file_inpaint])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_inpaint = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_inpaint_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_inpaint_cancel.click(fn=initiate_stop_inpaint, inputs=None, outputs=None)
                        with gr.Column():
                            btn_inpaint_clear_input = gr.ClearButton(components=[img_inpaint, gs_img_inpaint, prompt_inpaint, negative_prompt_inpaint], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_inpaint_clear_output = gr.ClearButton(components=[out_inpaint, gs_out_inpaint], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_inpaint.click(fn=hide_download_file_inpaint, inputs=None, outputs=download_file_inpaint)
                            btn_inpaint.click(
                                fn=image_inpaint,
                                inputs=[
                                    model_inpaint,
                                    sampler_inpaint,
                                    img_inpaint,
                                    rotation_img_inpaint,
                                    prompt_inpaint,
                                    negative_prompt_inpaint,
                                    num_images_per_prompt_inpaint,
                                    num_prompt_inpaint,
                                    guidance_scale_inpaint,
                                    denoising_strength_inpaint,
                                    num_inference_step_inpaint,
                                    height_inpaint,
                                    width_inpaint,
                                    seed_inpaint,
                                    use_gfpgan_inpaint,
                                    nsfw_filter,
                                    tkme_inpaint,
                                    clipskip_inpaint,
                                    use_ays_inpaint,
                                ],
                                outputs=[out_inpaint, gs_out_inpaint], 
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        inpaint_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        inpaint_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        inpaint_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        inpaint_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        inpaint_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        inpaint_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        inpaint_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        inpaint_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        inpaint_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        inpaint_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        inpaint_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        inpaint_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        inpaint_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        inpaint_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        inpaint_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        inpaint_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        inpaint_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        inpaint_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        inpaint_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        inpaint_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        inpaint_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        inpaint_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        inpaint_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        inpaint_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        inpaint_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        inpaint_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        inpaint_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        inpaint_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        inpaint_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        inpaint_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        inpaint_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")
                                        inpaint_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")

# paintbyex
                if ram_size() >= 16 :
                    titletab_paintbyex = f"{biniou_lang_tab_paintbyex} 🖌️"
                else :
                    titletab_paintbyex = f"{biniou_lang_tab_paintbyex} ⛔"

                with gr.TabItem(titletab_paintbyex, id=293) as tab_paintbyex:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_paintbyex}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_paintbyex_about_desc}<a href='https://github.com/Fantasy-Studio/Paint-by-Example' target='_blank'>Paint by example</a>, <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_paintbyex_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_paintbyex)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_paintbyex_about_instruct}
                                </br>
                                </div>
                                """
                            )                   
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_paintbyex = gr.Dropdown(choices=model_list_paintbyex, value=model_list_paintbyex[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_paintbyex = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_paintbyex = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_paintbyex = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_paintbyex= gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_paintbyex = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_paintbyex = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_paintbyex = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_paintbyex = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_paintbyex = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_paintbyex = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_paintbyex = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_paintbyex = gr.Textbox(value="paintbyex", visible=False, interactive=False)
                                del_ini_btn_paintbyex = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_paintbyex.value) else False)
                                save_ini_btn_paintbyex.click(
                                    fn=write_ini_paintbyex,
                                    inputs=[
                                        module_name_paintbyex, 
                                        model_paintbyex, 
                                        num_inference_step_paintbyex,
                                        sampler_paintbyex,
                                        guidance_scale_paintbyex,
                                        num_images_per_prompt_paintbyex,
                                        num_prompt_paintbyex,
                                        width_paintbyex,
                                        height_paintbyex,
                                        seed_paintbyex,
                                        use_gfpgan_paintbyex,
                                        tkme_paintbyex,
                                        ]
                                    )
                                save_ini_btn_paintbyex.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_paintbyex.click(fn=lambda: del_ini_btn_paintbyex.update(interactive=True), outputs=del_ini_btn_paintbyex)
                                del_ini_btn_paintbyex.click(fn=lambda: del_ini(module_name_paintbyex.value))
                                del_ini_btn_paintbyex.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_paintbyex.click(fn=lambda: del_ini_btn_paintbyex.update(interactive=False), outputs=del_ini_btn_paintbyex)
                        if test_ini_exist(module_name_paintbyex.value) :
                            with open(f".ini/{module_name_paintbyex.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column(scale=2):
                             rotation_img_paintbyex = gr.Number(value=0, visible=False)
                             img_paintbyex = gr.Image(label=biniou_lang_img_input_label, type="pil", height=400, tool="sketch")
                             img_paintbyex.upload(image_upload_event_inpaint, inputs=img_paintbyex, outputs=[width_paintbyex, height_paintbyex, img_paintbyex, rotation_img_paintbyex], preprocess=False)
                             gs_img_paintbyex = gr.Image(type="pil", visible=False)
                             gs_img_paintbyex.change(image_upload_event_inpaint_b, inputs=gs_img_paintbyex, outputs=[width_paintbyex, height_paintbyex], preprocess=False)
                        with gr.Column():
                             example_img_paintbyex = gr.Image(label=biniou_lang_tab_paintbyex_imgex_label, type="pil", height=400)
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    out_paintbyex = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_paintbyex",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                    )
                                    gs_out_paintbyex = gr.State()
                                    sel_out_paintbyex = gr.Number(precision=0, visible=False)
                                    out_paintbyex.select(get_select_index, None, sel_out_paintbyex)
                                    with gr.Row():
                                        with gr.Column():
                                            download_btn_paintbyex = gr.Button(f"{biniou_lang_image_zip} 💾")
                                        with gr.Column():
                                            download_file_paintbyex = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                            download_btn_paintbyex.click(fn=zip_download_file_paintbyex, inputs=out_paintbyex, outputs=[download_file_paintbyex, download_file_paintbyex])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_paintbyex = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_paintbyex_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_paintbyex_cancel.click(fn=initiate_stop_paintbyex, inputs=None, outputs=None)
                        with gr.Column():
                            btn_paintbyex_clear_input = gr.ClearButton(components=[img_paintbyex, gs_img_paintbyex, example_img_paintbyex], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_paintbyex_clear_output = gr.ClearButton(components=[out_paintbyex, gs_out_paintbyex], value=f"{biniou_lang_clear_outputs} 🧹")  
                            btn_paintbyex.click(fn=hide_download_file_paintbyex, inputs=None, outputs=download_file_paintbyex)                             
                            btn_paintbyex.click(
                                fn=image_paintbyex,
                                inputs=[
                                    model_paintbyex,
                                    sampler_paintbyex,
                                    img_paintbyex,
                                    rotation_img_paintbyex,
                                    example_img_paintbyex,
                                    num_images_per_prompt_paintbyex,
                                    num_prompt_paintbyex,
                                    guidance_scale_paintbyex,
                                    num_inference_step_paintbyex,
                                    height_paintbyex,
                                    width_paintbyex,
                                    seed_paintbyex,
                                    use_gfpgan_paintbyex,
                                    nsfw_filter,
                                    tkme_paintbyex,
                                ],
                                outputs=[out_paintbyex, gs_out_paintbyex], 
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        paintbyex_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        paintbyex_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        paintbyex_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        paintbyex_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        paintbyex_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}") 
                                        paintbyex_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}") 
                                        paintbyex_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        paintbyex_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}") 
                                        paintbyex_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        paintbyex_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        paintbyex_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        paintbyex_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        paintbyex_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        paintbyex_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        paintbyex_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        paintbyex_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        paintbyex_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box(): 
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# outpaint
                if ram_size() >= 16 :
                    titletab_outpaint = f"{biniou_lang_tab_outpaint} 🖌️"
                else :
                    titletab_outpaint = f"{biniou_lang_tab_outpaint} ⛔"

                with gr.TabItem(titletab_outpaint, id=294) as tab_outpaint:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_outpaint}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_outpaint_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_outpaint_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_outpaint)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_outpaint_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}
                                </div>
                                """
                            )                   
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_outpaint = gr.Dropdown(choices=model_list_outpaint, value=model_list_outpaint[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_outpaint = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_outpaint = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_outpaint = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_outpaint= gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_outpaint = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_outpaint = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_outpaint = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_outpaint = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_outpaint = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_outpaint = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_outpaint = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_outpaint = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_outpaint = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_outpaint = gr.Textbox(value="outpaint", visible=False, interactive=False)
                                del_ini_btn_outpaint = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_outpaint.value) else False)
                                save_ini_btn_outpaint.click(
                                    fn=write_ini_outpaint,
                                    inputs=[
                                        module_name_outpaint, 
                                        model_outpaint, 
                                        num_inference_step_outpaint,
                                        sampler_outpaint,
                                        guidance_scale_outpaint,
                                        num_images_per_prompt_outpaint,
                                        num_prompt_outpaint,
                                        width_outpaint,
                                        height_outpaint,
                                        seed_outpaint,
                                        use_gfpgan_outpaint,
                                        tkme_outpaint,
                                        clipskip_outpaint,
                                        use_ays_outpaint,
                                        ]
                                    )
                                save_ini_btn_outpaint.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_outpaint.click(fn=lambda: del_ini_btn_outpaint.update(interactive=True), outputs=del_ini_btn_outpaint)
                                del_ini_btn_outpaint.click(fn=lambda: del_ini(module_name_outpaint.value))
                                del_ini_btn_outpaint.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_outpaint.click(fn=lambda: del_ini_btn_outpaint.update(interactive=False), outputs=del_ini_btn_outpaint)
                        if test_ini_exist(module_name_outpaint.value):
                            with open(f".ini/{module_name_outpaint.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                rotation_img_outpaint = gr.Number(value=0, visible=False)
                                img_outpaint = gr.Image(label=biniou_lang_img_input_label, type="pil", height=350)
                                gs_img_outpaint = gr.Image(type="pil", visible=False)
                                gs_img_outpaint.change(image_upload_event_inpaint_b, inputs=gs_img_outpaint, outputs=[width_outpaint, height_outpaint], preprocess=False)
                            with gr.Column():
                                with gr.Row():									
                                    top_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label=biniou_lang_tab_outpaint_top_label, info=biniou_lang_tab_outpaint_top_info)
                                    bottom_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label=biniou_lang_tab_outpaint_bottom_label, info=biniou_lang_tab_outpaint_bottom_info)
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column(): 
                                        btn_outpaint_preview = gr.Button(f"{biniou_lang_tab_outpaint_mask_btn} 👁️")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    mask_outpaint = gr.Image(label=biniou_lang_tab_outpaint_mask_img, height=350, type="pil")
                                    gs_mask_outpaint = gr.Image(type="pil", visible=False)
                                    scale_preview_outpaint = gr.Number(value=2048, visible=False)
                                    mask_outpaint.upload(fn=scale_image, inputs=[mask_outpaint, scale_preview_outpaint], outputs=[width_outpaint, height_outpaint, mask_outpaint])
                                    gs_mask_outpaint.change(image_upload_event_inpaint_b, inputs=gs_mask_outpaint, outputs=[width_outpaint, height_outpaint], preprocess=False)
                                with gr.Column():
                                    with gr.Row():
                                        left_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label=biniou_lang_tab_outpaint_left_label, info=biniou_lang_tab_outpaint_left_info)
                                        right_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label=biniou_lang_tab_outpaint_right_label, info=biniou_lang_tab_outpaint_right_info)
                                        btn_outpaint_preview.click(fn=prepare_outpaint, inputs=[img_outpaint, top_outpaint, bottom_outpaint, left_outpaint, right_outpaint], outputs=[img_outpaint, gs_img_outpaint, mask_outpaint, gs_mask_outpaint], show_progress="full")
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_outpaint = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label=biniou_lang_image_denoising_label, info=biniou_lang_image_denoising_info)
                            with gr.Row():
                                with gr.Column():
                                    prompt_outpaint = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                                with gr.Column():
                                    negative_prompt_outpaint = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                            use_ays_outpaint.change(fn=change_ays_outpaint, inputs=use_ays_outpaint, outputs=[num_inference_step_outpaint, sampler_outpaint])
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    out_outpaint = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_outpaint",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                    )
                                    gs_out_outpaint = gr.State()
                                    sel_out_outpaint = gr.Number(precision=0, visible=False)
                                    out_outpaint.select(get_select_index, None, sel_out_outpaint)
                                    with gr.Row():
                                        with gr.Column():
                                            download_btn_outpaint = gr.Button(f"{biniou_lang_image_zip} 💾")
                                        with gr.Column():
                                            download_file_outpaint = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                            download_btn_outpaint.click(fn=zip_download_file_outpaint, inputs=out_outpaint, outputs=[download_file_outpaint, download_file_outpaint])
                    with gr.Row():
                        with gr.Column():
                            btn_outpaint = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_outpaint_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_outpaint_cancel.click(fn=initiate_stop_outpaint, inputs=None, outputs=None)
                        with gr.Column():
                            btn_outpaint_clear_input = gr.ClearButton(components=[img_outpaint, gs_img_outpaint, prompt_outpaint, negative_prompt_outpaint], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_outpaint_clear_output = gr.ClearButton(components=[out_outpaint, gs_out_outpaint], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_outpaint.click(fn=hide_download_file_outpaint, inputs=None, outputs=download_file_outpaint)
                            btn_outpaint.click(
                                fn=image_outpaint,
                                inputs=[
                                    model_outpaint,
                                    sampler_outpaint,
                                    img_outpaint,
                                    mask_outpaint,
                                    rotation_img_outpaint,
                                    prompt_outpaint,
                                    negative_prompt_outpaint,
                                    num_images_per_prompt_outpaint,
                                    num_prompt_outpaint,
                                    guidance_scale_outpaint,
                                    denoising_strength_outpaint,
                                    num_inference_step_outpaint,
                                    height_outpaint,
                                    width_outpaint,
                                    seed_outpaint,
                                    use_gfpgan_outpaint,
                                    nsfw_filter,
                                    tkme_outpaint,
                                    clipskip_outpaint,
                                    use_ays_outpaint,
                                ],
                                outputs=[out_outpaint, gs_out_outpaint], 
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        outpaint_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        outpaint_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        outpaint_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        outpaint_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        outpaint_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        outpaint_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        outpaint_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        outpaint_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        outpaint_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        outpaint_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        outpaint_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        outpaint_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        outpaint_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        outpaint_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        outpaint_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        outpaint_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        outpaint_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        outpaint_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        outpaint_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        outpaint_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        outpaint_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        outpaint_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        outpaint_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        outpaint_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        outpaint_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        outpaint_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                                        outpaint_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        outpaint_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        outpaint_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        outpaint_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        outpaint_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")
                                        outpaint_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")
# ControlNet
                with gr.TabItem(f"{biniou_lang_tab_controlnet} 🖼️", id=295) as tab_controlnet:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):                
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_controlnet}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_controlnet_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a>, <a href='https://stablediffusionweb.com/ControlNet' target='_blank'>ControlNet</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_controlnet_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_controlnet)}<br />
                                <b>HF ControlNet models pages : </b>
                                {autodoc(variant_list_controlnet)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_controlnet_about_instruct}</br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}</br>
                                <b>{biniou_lang_about_lora}</b></br>
                                - {biniou_lang_tab_image_about_lora_inst1}</br>
                                </div>
                                """
                            )                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_controlnet = gr.Dropdown(choices=model_list_controlnet, value=model_list_controlnet[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_controlnet = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_controlnet = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_controlnet = gr.Slider(0.0, 20.0, step=0.1, value=7.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_controlnet = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_controlnet = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_controlnet = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_controlnet = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_controlnet = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                low_threshold_controlnet = gr.Slider(0, 255, step=1, value=100, label=biniou_lang_tab_controlnet_lowthres_label, info=biniou_lang_tab_controlnet_lowthres_info)
                            with gr.Column():
                                high_threshold_controlnet = gr.Slider(0, 255, step=1, value=200, label=biniou_lang_tab_controlnet_highthres_label, info=biniou_lang_tab_controlnet_highthres_info)
                            with gr.Column():
                                strength_controlnet = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label=biniou_lang_tab_controlnet_strength_label, info=biniou_lang_tab_controlnet_strength_info)
                        with gr.Row():
                            with gr.Column():
                                start_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label=biniou_lang_tab_controlnet_start_label, info=biniou_lang_tab_controlnet_start_info)
                            with gr.Column():
                                stop_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label=biniou_lang_tab_controlnet_stop_label, info=biniou_lang_tab_controlnet_stop_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_controlnet = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_controlnet = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                            with gr.Column():
                                use_ays_controlnet = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_image_ays_label, info=biniou_lang_tab_image_ays_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_controlnet = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_controlnet = gr.Textbox(value="controlnet", visible=False, interactive=False)
                                del_ini_btn_controlnet = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_controlnet.value) else False)
                                save_ini_btn_controlnet.click(
                                    fn=write_ini_controlnet, 
                                    inputs=[
                                        module_name_controlnet, 
                                        model_controlnet, 
                                        num_inference_step_controlnet,
                                        sampler_controlnet,
                                        guidance_scale_controlnet,
                                        num_images_per_prompt_controlnet,
                                        num_prompt_controlnet,
                                        width_controlnet,
                                        height_controlnet,
                                        seed_controlnet,
                                        low_threshold_controlnet,
                                        high_threshold_controlnet,
                                        strength_controlnet,
                                        start_controlnet,
                                        stop_controlnet,
                                        use_gfpgan_controlnet,
                                        tkme_controlnet,
                                        clipskip_controlnet,
                                        use_ays_controlnet,
                                        ]
                                    )
                                save_ini_btn_controlnet.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_controlnet.click(fn=lambda: del_ini_btn_controlnet.update(interactive=True), outputs=del_ini_btn_controlnet)
                                del_ini_btn_controlnet.click(fn=lambda: del_ini(module_name_controlnet.value))
                                del_ini_btn_controlnet.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_controlnet.click(fn=lambda: del_ini_btn_controlnet.update(interactive=False), outputs=del_ini_btn_controlnet)
                        if test_ini_exist(module_name_controlnet.value) :
                            with open(f".ini/{module_name_controlnet.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                        with gr.Accordion(biniou_lang_lora_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_controlnet = gr.Dropdown(choices=list(lora_model_list(model_controlnet.value).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info)
                                with gr.Column():
                                    lora_weight_controlnet = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info)
                            with gr.Row():
                                with gr.Column():
                                    lora_model2_controlnet = gr.Dropdown(choices=list(lora_model_list(model_controlnet.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight2_controlnet = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model3_controlnet = gr.Dropdown(choices=list(lora_model_list(model_controlnet.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight3_controlnet = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    lora_model4_controlnet = gr.Dropdown(choices=list(lora_model_list(model_controlnet.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight4_controlnet = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model5_controlnet = gr.Dropdown(choices=list(lora_model_list(model_controlnet.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight5_controlnet = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                        with gr.Accordion(biniou_lang_textinv_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_controlnet = gr.Dropdown(choices=list(txtinv_list(model_controlnet.value).keys()), value="", label=biniou_lang_textinv_label, info=biniou_lang_textinv_info)
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    img_source_controlnet = gr.Image(label=biniou_lang_tab_controlnet_src_img, height=250, type="filepath")
                                    img_source_controlnet.change(fn=image_upload_event, inputs=img_source_controlnet, outputs=[width_controlnet, height_controlnet])
                                    gs_img_source_controlnet = gr.Image(type="pil", visible=False)
                                    gs_img_source_controlnet.change(fn=image_upload_event, inputs=gs_img_source_controlnet, outputs=[width_controlnet, height_controlnet], preprocess=False)
                            with gr.Row():
                                with gr.Column(): 
                                    preprocessor_controlnet = gr.Dropdown(choices=preprocessor_list_controlnet, value=preprocessor_list_controlnet[0], label=biniou_lang_tab_controlnet_prepro_label, info=biniou_lang_tab_controlnet_prepro_info)
                                    btn_controlnet_preview = gr.Button(f"{biniou_lang_tab_controlnet_preview_btn} 👁️")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    img_preview_controlnet = gr.Image(label=biniou_lang_tab_controlnet_preview_img, height=250, type="filepath")
                                    gs_img_preview_controlnet = gr.Image(type="pil", visible=False)
                            with gr.Row():
                                with gr.Column():
                                    variant_controlnet = gr.Dropdown(choices=variant_list_controlnet, value=variant_list_controlnet[0], label=biniou_lang_tab_controlnet_variant_label, info=biniou_lang_tab_controlnet_variant_info)
                                    gs_variant_controlnet = gr.Textbox(visible=False)
                                    variant_controlnet.change(fn=change_preview_controlnet, inputs=variant_controlnet, outputs=gs_variant_controlnet)
                                    gs_variant_controlnet.change(fn=change_preview_gs_controlnet, inputs=gs_variant_controlnet, outputs=variant_controlnet)
                                    scale_preview_controlnet = gr.Number(value=2048, visible=False)
                                    img_preview_controlnet.upload(fn=scale_image, inputs=[img_preview_controlnet, scale_preview_controlnet], outputs=[width_controlnet, height_controlnet, img_preview_controlnet])
                                    gs_img_preview_controlnet.change(image_upload_event_inpaint_b, inputs=gs_img_preview_controlnet, outputs=[width_controlnet, height_controlnet], preprocess=False)
                                    btn_controlnet_preview.click(fn=dispatch_controlnet_preview, inputs=[model_controlnet, low_threshold_controlnet, high_threshold_controlnet, img_source_controlnet, preprocessor_controlnet], outputs=[img_preview_controlnet, gs_img_preview_controlnet, gs_variant_controlnet], show_progress="full")
                            with gr.Row():
                                with gr.Column(): 
                                    btn_controlnet_clear_preview = gr.ClearButton(components=[img_preview_controlnet, gs_img_preview_controlnet], value=f"{biniou_lang_tab_controlnet_preview_clear} 🧹")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_controlnet = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_controlnet = gr.Textbox(lines=6, max_lines=6, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                            model_controlnet.change(
                                fn=change_model_type_controlnet, 
                                inputs=[model_controlnet],
                                outputs=[
                                    sampler_controlnet,
                                    width_controlnet,
                                    height_controlnet,
                                    num_inference_step_controlnet,
                                    guidance_scale_controlnet,
                                    lora_model_controlnet,
                                    txtinv_controlnet,
                                    negative_prompt_controlnet,
                                    img_preview_controlnet,
                                    gs_img_preview_controlnet,
                                ]
                            )
                            model_controlnet.change(fn=change_model_type_controlnet_alternate2, inputs=[model_controlnet],outputs=[lora_model2_controlnet])
                            model_controlnet.change(fn=change_model_type_controlnet_alternate3, inputs=[model_controlnet],outputs=[lora_model3_controlnet])
                            model_controlnet.change(fn=change_model_type_controlnet_alternate4, inputs=[model_controlnet],outputs=[lora_model4_controlnet])
                            model_controlnet.change(fn=change_model_type_controlnet_alternate5, inputs=[model_controlnet],outputs=[lora_model5_controlnet])
                            lora_model_controlnet.change(fn=change_lora_model_controlnet, inputs=[model_controlnet, lora_model_controlnet, prompt_controlnet, num_inference_step_controlnet, guidance_scale_controlnet, sampler_controlnet], outputs=[prompt_controlnet, num_inference_step_controlnet, guidance_scale_controlnet, sampler_controlnet])
                            lora_model2_controlnet.change(fn=change_lora_model2_controlnet, inputs=[model_controlnet, lora_model2_controlnet, prompt_controlnet], outputs=[prompt_controlnet])
                            lora_model3_controlnet.change(fn=change_lora_model3_controlnet, inputs=[model_controlnet, lora_model3_controlnet, prompt_controlnet], outputs=[prompt_controlnet])
                            lora_model4_controlnet.change(fn=change_lora_model4_controlnet, inputs=[model_controlnet, lora_model4_controlnet, prompt_controlnet], outputs=[prompt_controlnet])
                            lora_model5_controlnet.change(fn=change_lora_model5_controlnet, inputs=[model_controlnet, lora_model5_controlnet, prompt_controlnet], outputs=[prompt_controlnet])
                            txtinv_controlnet.change(fn=change_txtinv_controlnet, inputs=[model_controlnet, txtinv_controlnet, prompt_controlnet, negative_prompt_controlnet], outputs=[prompt_controlnet, negative_prompt_controlnet])
                            use_ays_controlnet.change(fn=change_ays_controlnet, inputs=use_ays_controlnet, outputs=[num_inference_step_controlnet, sampler_controlnet])
                        with gr.Column():
                            out_controlnet = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=1,
                                height=400,
                                preview=True,                                 
                            )    
                            gs_out_controlnet = gr.State()
                            sel_out_controlnet = gr.Number(precision=0, visible=False)
                            out_controlnet.select(get_select_index, None, sel_out_controlnet)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_controlnet = gr.Button(f"{biniou_lang_image_zip} 💾")
                                with gr.Column():
                                    download_file_controlnet = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_controlnet.click(fn=zip_download_file_controlnet, inputs=out_controlnet, outputs=[download_file_controlnet, download_file_controlnet])
                    with gr.Row():
                        with gr.Column():
                            btn_controlnet = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_controlnet_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_controlnet_cancel.click(fn=initiate_stop_controlnet, inputs=None, outputs=None)
                        with gr.Column():
                            btn_controlnet_clear_input = gr.ClearButton(components=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, img_preview_controlnet], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_controlnet_clear_output = gr.ClearButton(components=[out_controlnet, gs_out_controlnet], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_controlnet.click(fn=hide_download_file_controlnet, inputs=None, outputs=download_file_controlnet)
                            btn_controlnet.click(
                            fn=image_controlnet,
                            inputs=[
                                model_controlnet,
                                sampler_controlnet,
                                prompt_controlnet,
                                negative_prompt_controlnet,
                                num_images_per_prompt_controlnet,
                                num_prompt_controlnet,
                                guidance_scale_controlnet,
                                num_inference_step_controlnet,
                                height_controlnet,
                                width_controlnet,
                                seed_controlnet,
                                low_threshold_controlnet,
                                high_threshold_controlnet,
                                strength_controlnet,
                                start_controlnet,
                                stop_controlnet,
                                use_gfpgan_controlnet,
                                preprocessor_controlnet,
                                variant_controlnet,
                                img_preview_controlnet,
                                nsfw_filter,
                                tkme_controlnet,
                                clipskip_controlnet,
                                use_ays_controlnet,
                                lora_model_controlnet,
                                lora_weight_controlnet,
                                lora_model2_controlnet,
                                lora_weight2_controlnet,
                                lora_model3_controlnet,
                                lora_weight3_controlnet,
                                lora_model4_controlnet,
                                lora_weight4_controlnet,
                                lora_model5_controlnet,
                                lora_weight5_controlnet,
                                txtinv_controlnet,
                            ],
                                outputs=[out_controlnet, gs_out_controlnet],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        controlnet_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        controlnet_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        controlnet_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        controlnet_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        controlnet_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        controlnet_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        controlnet_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        controlnet_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        controlnet_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        controlnet_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        controlnet_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        controlnet_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        controlnet_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        controlnet_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        controlnet_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        controlnet_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        controlnet_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        controlnet_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        controlnet_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        controlnet_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        controlnet_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        controlnet_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        controlnet_img2img_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img}")
                                        controlnet_img2img_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_img2img_ip}")
                                        controlnet_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        controlnet_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        controlnet_faceid_ip_input = gr.Button(f"✍️ >> {biniou_lang_tab_faceid_ip}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        controlnet_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        controlnet_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        controlnet_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        controlnet_img2img_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img}")
                                        controlnet_img2img_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_img2img_ip}")
                                        controlnet_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        controlnet_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        controlnet_faceid_ip_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_faceid_ip}")


# faceid_ip
                with gr.TabItem(f"{biniou_lang_tab_faceid_ip} 🖼️", id=296) as tab_faceid_ip:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_faceid_ip}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_faceid_ip_about_desc}<a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a>, <a href='https://huggingface.co/h94/IP-Adapter-FaceID' target='_blank'>IP-Adapter FaceID</a>, <a href='https://github.com/deepinsight/insightface' target='_blank'>Insight face</a>, <a href='https://photo-maker.github.io/' target='_blank'>Photomaker</a>.</br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_img_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_faceid_ip)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_faceid_ip_about_instruct}
                                </br>
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}</br>
                                <b>{biniou_lang_about_lora}</b></br>
                                - {biniou_lang_tab_image_about_lora_inst1}</br>
                                </div>
                                """
                            )               
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_faceid_ip = gr.Dropdown(choices=model_list_faceid_ip, value=model_list_faceid_ip[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_faceid_ip = gr.Slider(2, biniou_global_steps_max, step=1, value=35, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_faceid_ip = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value="DDIM", label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_faceid_ip = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_faceid_ip = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info)
                            with gr.Column():
                                num_prompt_faceid_ip = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_faceid_ip = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_faceid_ip = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                seed_faceid_ip = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_faceid_ip = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_faceid_ip = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                            with gr.Column():
                                clipskip_faceid_ip = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_faceid_ip = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_faceid_ip = gr.Textbox(value="faceid_ip", visible=False, interactive=False)
                                del_ini_btn_faceid_ip = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_faceid_ip.value) else False)
                                save_ini_btn_faceid_ip.click(
                                    fn=write_ini_faceid_ip,
                                    inputs=[
                                        module_name_faceid_ip, 
                                        model_faceid_ip, 
                                        num_inference_step_faceid_ip,
                                        sampler_faceid_ip,
                                        guidance_scale_faceid_ip,
                                        num_images_per_prompt_faceid_ip,
                                        num_prompt_faceid_ip,
                                        width_faceid_ip,
                                        height_faceid_ip,
                                        seed_faceid_ip,
                                        use_gfpgan_faceid_ip,
                                        tkme_faceid_ip,
                                        clipskip_faceid_ip,
                                        ]
                                    )
                                save_ini_btn_faceid_ip.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_faceid_ip.click(fn=lambda: del_ini_btn_faceid_ip.update(interactive=True), outputs=del_ini_btn_faceid_ip)
                                del_ini_btn_faceid_ip.click(fn=lambda: del_ini(module_name_faceid_ip.value))
                                del_ini_btn_faceid_ip.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_faceid_ip.click(fn=lambda: del_ini_btn_faceid_ip.update(interactive=False), outputs=del_ini_btn_faceid_ip)
                        if test_ini_exist(module_name_faceid_ip.value) :
                            with open(f".ini/{module_name_faceid_ip.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                        with gr.Accordion(biniou_lang_lora_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_faceid_ip = gr.Dropdown(choices=list(lora_model_list(model_faceid_ip.value).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info)
                                with gr.Column():
                                    lora_weight_faceid_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info)
                            with gr.Row():
                                with gr.Column():
                                    lora_model2_faceid_ip = gr.Dropdown(choices=list(lora_model_list(model_faceid_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight2_faceid_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model3_faceid_ip = gr.Dropdown(choices=list(lora_model_list(model_faceid_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight3_faceid_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    lora_model4_faceid_ip = gr.Dropdown(choices=list(lora_model_list(model_faceid_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight4_faceid_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)
                                with gr.Column():
                                    lora_model5_faceid_ip = gr.Dropdown(choices=list(lora_model_list(model_faceid_ip.value, True).keys()), value="", label=biniou_lang_lora_label, info=biniou_lang_lora_info, interactive=True)
                                with gr.Column():
                                    lora_weight5_faceid_ip = gr.Slider(-5.0, 5.0, step=0.01, value=1.0, label=biniou_lang_lora_weight_label, info=biniou_lang_lora_weight_info, interactive=True)

                        with gr.Accordion(biniou_lang_textinv_label, open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_faceid_ip = gr.Dropdown(choices=list(txtinv_list(model_faceid_ip.value).keys()), value="", label=biniou_lang_textinv_label, info=biniou_lang_textinv_info)
                    with gr.Row():
                        with gr.Column():
                            img_faceid_ip = gr.Image(label=biniou_lang_img_input_label, height=400, type="filepath")
                            scale_preview_faceid_ip = gr.Number(value=512, visible=False)
                            img_faceid_ip.upload(fn=scale_image_any, inputs=[img_faceid_ip, scale_preview_faceid_ip], outputs=[img_faceid_ip])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    denoising_strength_faceid_ip = gr.Slider(0.01, 2.0, step=0.01, value=1.0, label=biniou_lang_tab_faceid_ip_denoising_label, info=biniou_lang_tab_faceid_ip_denoising_info)
                            with gr.Row():
                                with gr.Column():
                                    prompt_faceid_ip = gr.Textbox(lines=5, max_lines=5, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_image_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_faceid_ip = gr.Textbox(lines=5, max_lines=5, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_image_negprompt_info, placeholder=biniou_lang_image_negprompt_placeholder)
                        model_faceid_ip.change(
                            fn=change_model_type_faceid_ip, 
                            inputs=[model_faceid_ip, prompt_faceid_ip],
                            outputs=[
                                sampler_faceid_ip,
                                width_faceid_ip,
                                height_faceid_ip,
                                num_inference_step_faceid_ip,
                                guidance_scale_faceid_ip,
                                lora_model_faceid_ip,
                                txtinv_faceid_ip,
                                negative_prompt_faceid_ip,
                                prompt_faceid_ip,
                            ]
                        )
                        model_faceid_ip.change(fn=change_model_type_faceid_ip_alternate2, inputs=[model_faceid_ip],outputs=[lora_model2_faceid_ip])
                        model_faceid_ip.change(fn=change_model_type_faceid_ip_alternate3, inputs=[model_faceid_ip],outputs=[lora_model3_faceid_ip])
                        model_faceid_ip.change(fn=change_model_type_faceid_ip_alternate4, inputs=[model_faceid_ip],outputs=[lora_model4_faceid_ip])
                        model_faceid_ip.change(fn=change_model_type_faceid_ip_alternate5, inputs=[model_faceid_ip],outputs=[lora_model5_faceid_ip])
                        lora_model_faceid_ip.change(fn=change_lora_model_faceid_ip, inputs=[model_faceid_ip, lora_model_faceid_ip, prompt_faceid_ip, num_inference_step_faceid_ip, guidance_scale_faceid_ip, sampler_faceid_ip], outputs=[prompt_faceid_ip, num_inference_step_faceid_ip, guidance_scale_faceid_ip, sampler_faceid_ip])
                        lora_model2_faceid_ip.change(fn=change_lora_model2_faceid_ip, inputs=[model_faceid_ip, lora_model2_faceid_ip, prompt_faceid_ip], outputs=[prompt_faceid_ip])
                        lora_model3_faceid_ip.change(fn=change_lora_model3_faceid_ip, inputs=[model_faceid_ip, lora_model3_faceid_ip, prompt_faceid_ip], outputs=[prompt_faceid_ip])
                        lora_model4_faceid_ip.change(fn=change_lora_model4_faceid_ip, inputs=[model_faceid_ip, lora_model4_faceid_ip, prompt_faceid_ip], outputs=[prompt_faceid_ip])
                        lora_model5_faceid_ip.change(fn=change_lora_model5_faceid_ip, inputs=[model_faceid_ip, lora_model5_faceid_ip, prompt_faceid_ip], outputs=[prompt_faceid_ip])
                        txtinv_faceid_ip.change(fn=change_txtinv_faceid_ip, inputs=[model_faceid_ip, txtinv_faceid_ip, prompt_faceid_ip, negative_prompt_faceid_ip], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_faceid_ip = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_photobooth",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                )
                                gs_out_faceid_ip = gr.State()
                                sel_out_faceid_ip = gr.Number(precision=0, visible=False)
                                out_faceid_ip.select(get_select_index, None, sel_out_faceid_ip)
                                with gr.Row():
                                    with gr.Column():
                                        download_btn_faceid_ip = gr.Button(f"{biniou_lang_image_zip} 💾")
                                    with gr.Column():
                                        download_file_faceid_ip = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                        download_btn_faceid_ip.click(fn=zip_download_file_faceid_ip, inputs=out_faceid_ip, outputs=[download_file_faceid_ip, download_file_faceid_ip])
                    with gr.Row():
                        with gr.Column():
                            btn_faceid_ip = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_faceid_ip_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_faceid_ip_cancel.click(fn=initiate_stop_faceid_ip, inputs=None, outputs=None)
                        with gr.Column():
                            btn_faceid_ip_clear_input = gr.ClearButton(components=[img_faceid_ip, prompt_faceid_ip, negative_prompt_faceid_ip], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_faceid_ip_clear_output = gr.ClearButton(components=[out_faceid_ip, gs_out_faceid_ip], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_faceid_ip.click(fn=hide_download_file_faceid_ip, inputs=None, outputs=download_file_faceid_ip)
                            btn_faceid_ip.click(
                                fn=image_faceid_ip,
                                inputs=[
                                    model_faceid_ip,
                                    sampler_faceid_ip,
                                    img_faceid_ip,
                                    prompt_faceid_ip,
                                    negative_prompt_faceid_ip,
                                    num_images_per_prompt_faceid_ip,
                                    num_prompt_faceid_ip,
                                    guidance_scale_faceid_ip,
                                    denoising_strength_faceid_ip,
                                    num_inference_step_faceid_ip,
                                    height_faceid_ip,
                                    width_faceid_ip,
                                    seed_faceid_ip,
                                    use_gfpgan_faceid_ip,
                                    nsfw_filter,
                                    tkme_faceid_ip,
                                    clipskip_faceid_ip,
                                    lora_model_faceid_ip,
                                    lora_weight_faceid_ip,
                                    lora_model2_faceid_ip,
                                    lora_weight2_faceid_ip,
                                    lora_model3_faceid_ip,
                                    lora_weight3_faceid_ip,
                                    lora_model4_faceid_ip,
                                    lora_weight4_faceid_ip,
                                    lora_model5_faceid_ip,
                                    lora_weight5_faceid_ip,
                                    txtinv_faceid_ip,
                                ],
                                outputs=[out_faceid_ip, gs_out_faceid_ip],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        faceid_ip_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        faceid_ip_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        faceid_ip_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        faceid_ip_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        faceid_ip_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        faceid_ip_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        faceid_ip_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        faceid_ip_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        faceid_ip_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        faceid_ip_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        faceid_ip_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        faceid_ip_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        faceid_ip_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        faceid_ip_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        faceid_ip_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        faceid_ip_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value)
                                        faceid_ip_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        faceid_ip_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        faceid_ip_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        faceid_ip_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        faceid_ip_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        faceid_ip_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        faceid_ip_pix2pix_input = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                                        faceid_ip_inpaint_input = gr.Button(f"✍️ >> {biniou_lang_tab_inpaint}")
                                        faceid_ip_controlnet_input = gr.Button(f"✍️ >> {biniou_lang_tab_controlnet}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        faceid_ip_pix2pix_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_pix2pix}")
                                        faceid_ip_inpaint_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_inpaint}")
                                        faceid_ip_controlnet_both = gr.Button(f"🖼️ + ✍️ >> {biniou_lang_tab_controlnet}")

# faceswap    
                with gr.TabItem(f"{biniou_lang_tab_faceswap} 🎭", id=297) as tab_faceswap:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_faceswap}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_faceswap_about_desc}<a href='https://github.com/deepinsight/insightface' target='_blank'>Insight Face</a>, <a href='https://github.com/microsoft/onnxruntime' target='_blank'>Onnx runtime</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_faceswap_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_image_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_faceswap.keys())}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_faceswap_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_faceswap = gr.Dropdown(choices=list(model_list_faceswap.keys()), value=list(model_list_faceswap.keys())[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                width_faceswap = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_faceswap = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_faceswap = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_faceswap = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_faceswap = gr.Textbox(value="faceswap", visible=False, interactive=False)
                                del_ini_btn_faceswap = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_faceswap.value) else False)
                                save_ini_btn_faceswap.click(
                                    fn=write_ini_faceswap,
                                    inputs=[
                                        module_name_faceswap,
                                        model_faceswap,
                                        width_faceswap,
                                        height_faceswap,
                                        use_gfpgan_faceswap,
                                        ]
                                    )
                                save_ini_btn_faceswap.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_faceswap.click(fn=lambda: del_ini_btn_faceswap.update(interactive=True), outputs=del_ini_btn_faceswap)
                                del_ini_btn_faceswap.click(fn=lambda: del_ini(module_name_faceswap.value))
                                del_ini_btn_faceswap.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_faceswap.click(fn=lambda: del_ini_btn_faceswap.update(interactive=False), outputs=del_ini_btn_faceswap)
                        if test_ini_exist(module_name_faceswap.value) :
                            with open(f".ini/{module_name_faceswap.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            img_source_faceswap = gr.Image(label=biniou_lang_tab_faceswap_src_img, height=400, type="filepath")
                            scale_preview_faceswap = gr.Number(value=512, visible=False)
                            img_source_faceswap.upload(fn=scale_image_any, inputs=[img_source_faceswap, scale_preview_faceswap], outputs=[img_source_faceswap])
                            with gr.Row():
                                source_index_faceswap = gr.Textbox(value=0, lines=1, label=biniou_lang_tab_faceswap_src_index_label, info=biniou_lang_tab_faceswap_src_index_info)
                        with gr.Column():
                             img_target_faceswap = gr.Image(label=biniou_lang_tab_faceswap_tgt_img, type="filepath", height=400)
                             gs_img_target_faceswap = gr.Image(type="pil", visible=False)
                             img_target_faceswap.change(image_upload_event, inputs=img_target_faceswap, outputs=[width_faceswap, height_faceswap])
                             with gr.Row():
                                 target_index_faceswap = gr.Textbox(value=0, lines=1, label=biniou_lang_tab_faceswap_tgt_index_label, info=biniou_lang_tab_faceswap_tgt_index_info)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_faceswap = gr.Gallery(
                                        label=biniou_lang_image_gallery_label,
                                        show_label=True,
                                        elem_id="gallery_fsw",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                    )
                                    gs_out_faceswap = gr.State()
                                    sel_out_faceswap = gr.Number(precision=0, visible=False)
                                    out_faceswap.select(get_select_index, None, sel_out_faceswap)
                                    with gr.Row():
                                        with gr.Column():
                                            download_btn_faceswap = gr.Button(f"{biniou_lang_image_zip} 💾")
                                        with gr.Column():
                                            download_file_faceswap = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                            download_btn_faceswap.click(fn=zip_download_file_faceswap, inputs=out_faceswap, outputs=[download_file_faceswap, download_file_faceswap])
                    with gr.Row():
                        with gr.Column():
                            btn_faceswap = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_faceswap_clear_input = gr.ClearButton(components=[img_source_faceswap, img_target_faceswap, source_index_faceswap, target_index_faceswap, gs_img_target_faceswap], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_faceswap_clear_output = gr.ClearButton(components=[out_faceswap, gs_out_faceswap], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_faceswap.click(fn=hide_download_file_faceswap, inputs=None, outputs=download_file_faceswap)
                            btn_faceswap.click(
                                fn=image_faceswap,
                                inputs=[
                                    model_faceswap, 
                                    img_source_faceswap, 
                                    img_target_faceswap, 
                                    source_index_faceswap, 
                                    target_index_faceswap,
                                    use_gfpgan_faceswap,
                                ],
                                outputs=[out_faceswap, gs_out_faceswap],
                                show_progress="full",
                            )  
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        faceswap_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        faceswap_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        faceswap_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        faceswap_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        faceswap_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        faceswap_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        faceswap_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        faceswap_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        faceswap_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        faceswap_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        faceswap_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        faceswap_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        faceswap_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        faceswap_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        faceswap_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        faceswap_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        faceswap_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# Real ESRGAN    
                with gr.TabItem(f"{biniou_lang_tab_resrgan} 🔎", id=298) as tab_resrgan:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_resrgan}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_resrgan_about_desc}<a href='https://github.com/xinntao/Real-ESRGAN' target='_blank'>Real ESRGAN</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_image}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_resrgan_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_resrgan)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_resrgan_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_resrgan = gr.Dropdown(choices=model_list_resrgan, value=model_list_resrgan[1], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                scale_resrgan = gr.Dropdown(choices=list(RESRGAN_SCALES.keys()), value=list(RESRGAN_SCALES.keys())[1], label=biniou_lang_tab_resrgan_scale_label, info=biniou_lang_tab_resrgan_scale_info)
                                scale_resrgan.change(scale_resrgan_change, inputs=scale_resrgan, outputs=model_resrgan)
                        with gr.Row():
                            with gr.Column():
                                width_resrgan = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_tab_resrgan_width_info, interactive=False)
                            with gr.Column():
                                height_resrgan = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_tab_resrgan_height_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_resrgan = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_resrgan = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_resrgan = gr.Textbox(value="resrgan", visible=False, interactive=False)
                                del_ini_btn_resrgan = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_resrgan.value) else False)
                                save_ini_btn_resrgan.click(
                                    fn=write_ini_resrgan,
                                    inputs=[
                                        module_name_resrgan,
                                        model_resrgan,
                                        scale_resrgan,
                                        width_resrgan,
                                        height_resrgan,
                                        use_gfpgan_resrgan,
                                        ]
                                    )
                                save_ini_btn_resrgan.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_resrgan.click(fn=lambda: del_ini_btn_resrgan.update(interactive=True), outputs=del_ini_btn_resrgan)
                                del_ini_btn_resrgan.click(fn=lambda: del_ini(module_name_resrgan.value))
                                del_ini_btn_resrgan.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_resrgan.click(fn=lambda: del_ini_btn_resrgan.update(interactive=False), outputs=del_ini_btn_resrgan)
                        if test_ini_exist(module_name_resrgan.value) :
                            with open(f".ini/{module_name_resrgan.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                             img_resrgan = gr.Image(label=biniou_lang_img_input_label, type="filepath", height=400)
                             img_resrgan.change(image_upload_event, inputs=img_resrgan, outputs=[width_resrgan, height_resrgan])
                        with gr.Column():
                            out_resrgan = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery_resr",
                                columns=1,
                                height=400,
                                preview=True,
                            )
                        gs_out_resrgan = gr.State()
                        sel_out_resrgan = gr.Number(precision=0, visible=False)
                    with gr.Row():
                        with gr.Column():
                            btn_resrgan = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_resrgan_clear_input = gr.ClearButton(components=[img_resrgan], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_resrgan_clear_output = gr.ClearButton(components=[out_resrgan, gs_out_resrgan], value=f"{biniou_lang_clear_outputs} 🧹") 
                            btn_resrgan.click(
                                fn=image_resrgan,
                                inputs=[
                                    model_resrgan,
                                    scale_resrgan,
                                    img_resrgan,
                                    use_gfpgan_resrgan,
                                ],
                                outputs=[out_resrgan, gs_out_resrgan],
                                show_progress="full",
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        resrgan_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        resrgan_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}") 
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        resrgan_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        resrgan_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        resrgan_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        resrgan_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        resrgan_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        resrgan_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        resrgan_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}") 
                                        resrgan_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        resrgan_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        resrgan_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        resrgan_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        resrgan_gfpgan = gr.Button(f"🖼️ >> {biniou_lang_tab_gfpgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        resrgan_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        resrgan_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# GFPGAN    
                with gr.TabItem(f"{biniou_lang_tab_gfpgan} 🔎", id=299) as tab_gfpgan:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_gfpgan}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_gfpgan_about_desc}<a href='https://github.com/TencentARC/GFPGAN' target='_blank'>GFPGAN</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_image}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_gfpgan_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_gfpgan)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_gfpgan_about_instruct}
                                </div>
                                """
                            )                     
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_gfpgan = gr.Dropdown(choices=model_list_gfpgan, value=model_list_gfpgan[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                variant_gfpgan = gr.Dropdown(choices=variant_list_gfpgan, value=variant_list_gfpgan[4], label=biniou_lang_tab_gfpgan_variant_label, info=biniou_lang_tab_gfpgan_variant_info)
                        with gr.Row():
                            with gr.Column():
                                width_gfpgan = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_gfpgan = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_gfpgan = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_gfpgan = gr.Textbox(value="gfpgan", visible=False, interactive=False)
                                del_ini_btn_gfpgan = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_gfpgan.value) else False)
                                save_ini_btn_gfpgan.click(
                                    fn=write_ini_gfpgan,
                                    inputs=[
                                        module_name_gfpgan, 
                                        model_gfpgan, 
                                        variant_gfpgan,
                                        width_gfpgan,
                                        height_gfpgan,
                                        ]
                                    )
                                save_ini_btn_gfpgan.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_gfpgan.click(fn=lambda: del_ini_btn_gfpgan.update(interactive=True), outputs=del_ini_btn_gfpgan)
                                del_ini_btn_gfpgan.click(fn=lambda: del_ini(module_name_gfpgan.value))
                                del_ini_btn_gfpgan.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_gfpgan.click(fn=lambda: del_ini_btn_gfpgan.update(interactive=False), outputs=del_ini_btn_gfpgan)
                        if test_ini_exist(module_name_gfpgan.value):
                            with open(f".ini/{module_name_gfpgan.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            img_gfpgan = gr.Image(label=biniou_lang_img_input_label, type="filepath", height=400)
                            img_gfpgan.change(image_upload_event, inputs=img_gfpgan, outputs=[width_gfpgan, height_gfpgan])
                        with gr.Column():
                            out_gfpgan = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery_gfp",
                                columns=1,
                                height=400,
                                preview=True,
                            )
                        gs_out_gfpgan = gr.State()
                        sel_out_gfpgan = gr.Number(precision=0, visible=False)
                    with gr.Row():
                        with gr.Column():
                            btn_gfpgan = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_gfpgan_clear_input = gr.ClearButton(components=[img_gfpgan], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_gfpgan_clear_output = gr.ClearButton(components=[out_gfpgan, gs_out_gfpgan], value=f"{biniou_lang_clear_outputs} 🧹") 
                        btn_gfpgan.click(
                            fn=image_gfpgan,
                            inputs=[
                                model_gfpgan,
                                variant_gfpgan,
                                img_gfpgan,
                            ],
                            outputs=[out_gfpgan, gs_out_gfpgan],
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        gfpgan_llava = gr.Button(f"🖼️ >> {biniou_lang_tab_llava}")
                                        gfpgan_img2txt_git = gr.Button(f"🖼️ >> {biniou_lang_tab_img2txt_git}")
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        gfpgan_img2img = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img}")
                                        gfpgan_img2img_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_img2img_ip}")
                                        gfpgan_img2var = gr.Button(f"🖼️ >> {biniou_lang_tab_img2var}")
                                        gfpgan_pix2pix = gr.Button(f"🖼️ >> {biniou_lang_tab_pix2pix}")
                                        gfpgan_magicmix = gr.Button(f"🖼️ >> {biniou_lang_tab_magicmix}")
                                        gfpgan_inpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_inpaint}")
                                        gfpgan_paintbyex = gr.Button(f"🖼️ >> {biniou_lang_tab_paintbyex}")
                                        gfpgan_outpaint = gr.Button(f"🖼️ >> {biniou_lang_tab_outpaint}")
                                        gfpgan_controlnet = gr.Button(f"🖼️ >> {biniou_lang_tab_controlnet}")
                                        gfpgan_faceid_ip = gr.Button(f"🖼️ >> {biniou_lang_tab_faceid_ip}")
                                        gfpgan_faceswap = gr.Button(f"🖼️ >> {biniou_lang_tab_faceswap}")
                                        gfpgan_resrgan = gr.Button(f"🖼️ >> {biniou_lang_tab_resrgan}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        gfpgan_img2vid = gr.Button(f"🖼️ >> {biniou_lang_tab_img2vid}")
                                        gr.HTML(value=biniou_lang_send_3d_value) 
                                        gfpgan_img2shape = gr.Button(f"🖼️ >> {biniou_lang_tab_img2shape}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Audio
        with gr.TabItem(f"{biniou_lang_tab_audio} 🎵", id=3) as tab_audio:
            with gr.Tabs() as tabs_audio:        
# Musicgen
                with gr.TabItem(f"{biniou_lang_tab_musicgen} 🎶", id=31) as tab_musicgen:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_musicgen}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_musicgen_about_desc}<a href='https://github.com/facebookresearch/audiocraft' target='_blank'>MusicGen</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_audio_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(modellist_musicgen)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_musicgen_about_instruct}
                                </div>
                                """
                            )                           
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_musicgen= gr.Dropdown(choices=modellist_musicgen, value=modellist_musicgen[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                duration_musicgen = gr.Slider(1, 160, step=1, value=5, label=biniou_lang_audio_length_label)
                            with gr.Column():
                                cfg_coef_musicgen = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_batch_musicgen = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                use_sampling_musicgen = gr.Checkbox(value=True, label=biniou_lang_audio_sampling_label)
                            with gr.Column():
                                temperature_musicgen = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label=biniou_lang_temperature_label)
                            with gr.Column():
                                top_k_musicgen = gr.Slider(0, 500, step=1, value=250, label=biniou_lang_top_k_label)
                            with gr.Column():
                                top_p_musicgen = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label=biniou_lang_top_p_label)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_musicgen = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_musicgen = gr.Textbox(value="musicgen", visible=False, interactive=False)
                                del_ini_btn_musicgen = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_musicgen.value) else False)
                                save_ini_btn_musicgen.click(
                                    fn=write_ini_musicgen, 
                                    inputs=[
                                        module_name_musicgen,
                                        model_musicgen,
                                        duration_musicgen,
                                        cfg_coef_musicgen,
                                        num_batch_musicgen,
                                        use_sampling_musicgen,
                                        temperature_musicgen,
                                        top_k_musicgen,
                                        top_p_musicgen,
                                        ]
                                    )
                                save_ini_btn_musicgen.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_musicgen.click(fn=lambda: del_ini_btn_musicgen.update(interactive=True), outputs=del_ini_btn_musicgen)
                                del_ini_btn_musicgen.click(fn=lambda: del_ini(module_name_musicgen.value))
                                del_ini_btn_musicgen.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_musicgen.click(fn=lambda: del_ini_btn_musicgen.update(interactive=False), outputs=del_ini_btn_musicgen)
                        if test_ini_exist(module_name_musicgen.value) :
                            with open(f".ini/{module_name_musicgen.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())

                    with gr.Row():
                        with gr.Column():
                            prompt_musicgen = gr.Textbox(label=biniou_lang_audio_prompt_label, lines=2, max_lines=2, show_copy_button=True, placeholder=biniou_lang_audio_prompt_placeholder)
                        with gr.Column():
                            out_musicgen = gr.Audio(label=biniou_lang_audio_generated_label, type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_musicgen = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_musicgen_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_musicgen_cancel.click(fn=initiate_stop_musicgen, inputs=None, outputs=None)
                        with gr.Column():
                            btn_musicgen_clear_input = gr.ClearButton(components=prompt_musicgen, value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_musicgen_clear_output = gr.ClearButton(components=out_musicgen, value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_musicgen.click(
                            fn=music_musicgen,
                            inputs=[
                                prompt_musicgen,
                                model_musicgen,
                                duration_musicgen,
                                num_batch_musicgen,
                                temperature_musicgen,
                                top_k_musicgen,
                                top_p_musicgen,
                                use_sampling_musicgen,
                                cfg_coef_musicgen,
                            ], 
                            outputs=out_musicgen,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        musicgen_musicgen_mel = gr.Button(f"🎶 >> {biniou_lang_tab_musicgen_mel}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        musicgen_musicgen_mel_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen_mel}")
                                        musicgen_musicldm_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicldm}")
                                        musicgen_audiogen_input = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# Musicgen Melody
                if ram_size() >= 16 :
                    titletab_musicgen_mel = f"{biniou_lang_tab_musicgen_mel} 🎶"
                else :
                    titletab_musicgen_mel = f"{biniou_lang_tab_musicgen_mel} ⛔"

                with gr.TabItem(titletab_musicgen_mel, id=32) as tab_musicgen_mel:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_musicgen_mel}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_musicgen_mel_about_desc}<a href='https://github.com/facebookresearch/audiocraft' target='_blank'>MusicGen</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_musicgen_mel_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_audio_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(modellist_musicgen_mel)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_musicgen_mel_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_musicgen_mel= gr.Dropdown(choices=modellist_musicgen_mel, value=modellist_musicgen_mel[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                duration_musicgen_mel = gr.Slider(1, 160, step=1, value=5, label=biniou_lang_audio_length_label)
                            with gr.Column():
                                cfg_coef_musicgen_mel = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_batch_musicgen_mel = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                use_sampling_musicgen_mel = gr.Checkbox(value=True, label=biniou_lang_audio_sampling_label)
                            with gr.Column():
                                temperature_musicgen_mel = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label=biniou_lang_temperature_label)
                            with gr.Column():
                                top_k_musicgen_mel = gr.Slider(0, 500, step=1, value=250, label=biniou_lang_top_k_label)
                            with gr.Column():
                                top_p_musicgen_mel = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label=biniou_lang_top_p_label)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_musicgen_mel = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_musicgen_mel = gr.Textbox(value="musicgen_mel", visible=False, interactive=False)
                                del_ini_btn_musicgen_mel = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_musicgen_mel.value) else False)
                                save_ini_btn_musicgen_mel.click(
                                    fn=write_ini_musicgen_mel,
                                    inputs=[
                                        module_name_musicgen_mel,
                                        model_musicgen_mel,
                                        duration_musicgen_mel,
                                        cfg_coef_musicgen_mel,
                                        num_batch_musicgen_mel,
                                        use_sampling_musicgen_mel,
                                        temperature_musicgen_mel,
                                        top_k_musicgen_mel,
                                        top_p_musicgen_mel,
                                        ]
                                    )
                                save_ini_btn_musicgen_mel.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_musicgen_mel.click(fn=lambda: del_ini_btn_musicgen_mel.update(interactive=True), outputs=del_ini_btn_musicgen_mel)
                                del_ini_btn_musicgen_mel.click(fn=lambda: del_ini(module_name_musicgen_mel.value))
                                del_ini_btn_musicgen_mel.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_musicgen_mel.click(fn=lambda: del_ini_btn_musicgen_mel.update(interactive=False), outputs=del_ini_btn_musicgen_mel)
                        if test_ini_exist(module_name_musicgen_mel.value) :
                            with open(f".ini/{module_name_musicgen_mel.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                source_type_musicgen_mel = gr.Radio(choices=["audio", "micro"], value="audio", label=biniou_lang_tab_musicgen_mel_src_type_label, info=biniou_lang_tab_musicgen_mel_src_type_info)
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            source_audio_musicgen_mel = gr.Audio(label=biniou_lang_tab_musicgen_mel_src_audio, source="upload", type="filepath")
                            source_type_musicgen_mel.change(fn=change_source_type_musicgen_mel, inputs=source_type_musicgen_mel, outputs=source_audio_musicgen_mel)
                        with gr.Column():
                            prompt_musicgen_mel = gr.Textbox(label=biniou_lang_audio_prompt_label, lines=8, max_lines=8, show_copy_button=True, placeholder=biniou_lang_audio_prompt_placeholder)
                        with gr.Column():
                            out_musicgen_mel = gr.Audio(label=biniou_lang_audio_generated_label, type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_musicgen_mel = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_musicgen_mel_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_musicgen_mel_cancel.click(fn=initiate_stop_musicgen_mel, inputs=None, outputs=None)
                        with gr.Column():
                            btn_musicgen_mel_clear_input = gr.ClearButton(components=[prompt_musicgen_mel, source_audio_musicgen_mel], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_musicgen_mel_clear_output = gr.ClearButton(components=out_musicgen_mel, value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_musicgen_mel.click(
                            fn=music_musicgen_mel,
                            inputs=[
                                prompt_musicgen_mel,
                                model_musicgen_mel,
                                duration_musicgen_mel,
                                num_batch_musicgen_mel,
                                temperature_musicgen_mel,
                                top_k_musicgen_mel,
                                top_p_musicgen_mel,
                                use_sampling_musicgen_mel,
                                cfg_coef_musicgen_mel,
                                source_audio_musicgen_mel,
                                source_type_musicgen_mel,
                            ],
                            outputs=out_musicgen_mel,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        musicgen_mel_musicgen_mel = gr.Button(f"🎶 >> {biniou_lang_tab_musicgen_mel}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        musicgen_mel_musicgen_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        musicgen_mel_musicldm_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicldm}")
                                        musicgen_mel_audiogen_input = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# MusicLDM
                with gr.TabItem(f"{biniou_lang_tab_musicldm} 🎶", id=33) as tab_musicldm:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_musicldm}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_musicldm_about_desc}<a href='https://musicldm.github.io' target='_blank'>MusicLDM</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_audio_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_musicldm)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_musicldm_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_musicldm = gr.Dropdown(choices=model_list_musicldm, value=model_list_musicldm[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_musicldm = gr.Slider(1, 400, step=1, value=50, label=biniou_lang_steps_label, info=biniou_lang_tab_audio_steps_info)
                            with gr.Column():
                                sampler_musicldm = gr.Dropdown(choices=list(SCHEDULER_MAPPING_MUSICLDM.keys()), value=list(SCHEDULER_MAPPING_MUSICLDM.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_musicldm = gr.Slider(0.1, 20.0, step=0.1, value=2.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                audio_length_musicldm=gr.Slider(0, 160, step=1, value=10, label=biniou_lang_tab_musicldm_length_label, info=biniou_lang_tab_musicldm_length_info)
                            with gr.Column():
                                seed_musicldm = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)    
                        with gr.Row():
                            with gr.Column():
                                num_audio_per_prompt_musicldm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_audio_batch_size_info)
                            with gr.Column():
                                num_prompt_musicldm = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_musicldm = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_musicldm = gr.Textbox(value="musicldm", visible=False, interactive=False)
                                del_ini_btn_musicldm = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_musicldm.value) else False)
                                save_ini_btn_musicldm.click(
                                    fn=write_ini_musicldm, 
                                    inputs=[
                                        module_name_musicldm,
                                        model_musicldm,
                                        num_inference_step_musicldm,
                                        sampler_musicldm,
                                        guidance_scale_musicldm,
                                        audio_length_musicldm,
                                        seed_musicldm,
                                        num_audio_per_prompt_musicldm,
                                        num_prompt_musicldm,
                                        ]
                                    )
                                save_ini_btn_musicldm.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_musicldm.click(fn=lambda: del_ini_btn_musicldm.update(interactive=True), outputs=del_ini_btn_musicldm)
                                del_ini_btn_musicldm.click(fn=lambda: del_ini(module_name_musicldm.value))
                                del_ini_btn_musicldm.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_musicldm.click(fn=lambda: del_ini_btn_musicldm.update(interactive=False), outputs=del_ini_btn_musicldm)
                        if test_ini_exist(module_name_musicldm.value):
                            with open(f".ini/{module_name_musicldm.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                prompt_musicldm = gr.Textbox(label=biniou_lang_prompt_label, lines=2, max_lines=2, show_copy_button=True, info=biniou_lang_tab_musicldm_prompt_info, placeholder=biniou_lang_tab_musicldm_prompt_placeholder)
                            with gr.Row():
                                negative_prompt_musicldm = gr.Textbox(label=biniou_lang_negprompt_label, lines=2, max_lines=2, show_copy_button=True, info=biniou_lang_tab_musicldm_negprompt_info, placeholder=biniou_lang_tab_musicldm_negprompt_placeholder)
                        with gr.Column():
                            out_musicldm = gr.Audio(label=biniou_lang_audio_generated_label, type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_musicldm = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_musicldm_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_musicldm_cancel.click(fn=initiate_stop_musicldm, inputs=None, outputs=None)
                        with gr.Column():
                            btn_musicldm_clear_input = gr.ClearButton(components=prompt_musicldm, value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_musicldm_clear_output = gr.ClearButton(components=out_musicldm, value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_musicldm.click(
                            fn=music_musicldm,
                            inputs=[
                                model_musicldm,
                                sampler_musicldm,
                                prompt_musicldm,
                                negative_prompt_musicldm,
                                num_audio_per_prompt_musicldm,
                                num_prompt_musicldm,
                                guidance_scale_musicldm,
                                num_inference_step_musicldm,
                                audio_length_musicldm,
                                seed_musicldm,
                            ],
                            outputs=out_musicldm,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        musicldm_musicgen_mel = gr.Button(f"🎶 >> {biniou_lang_tab_musicgen_mel}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        musicldm_musicgen_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        musicldm_musicgen_mel_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen_mel}")
                                        musicldm_audiogen_input = gr.Button(f"✍️ >> {biniou_lang_tab_audiogen}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Audiogen
                if ram_size() >= 16 :
                    titletab_audiogen = f"{biniou_lang_tab_audiogen} 🔊"
                else :
                    titletab_audiogen = f"{biniou_lang_tab_audiogen} ⛔"
                
                with gr.TabItem(titletab_audiogen, id=34) as tab_audiogen:

                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_audiogen}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_audiogen_about_desc}<a href='https://github.com/facebookresearch/audiocraft' target='_blank'>Audiogen</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_prompt_label}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_audiogen_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(modellist_audiogen)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_audiogen_about_instruct}
                                </div>
                                """
                            )                       
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_audiogen= gr.Dropdown(choices=modellist_audiogen, value=modellist_audiogen[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():    
                                duration_audiogen = gr.Slider(1, 160, step=1, value=5, label=biniou_lang_audio_length_label)
                            with gr.Column():
                                cfg_coef_audiogen = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_batch_audiogen = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)  
                        with gr.Row():
                            with gr.Column():    
                                use_sampling_audiogen = gr.Checkbox(value=True, label=biniou_lang_audio_sampling_label)
                            with gr.Column():    
                                temperature_audiogen = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label=biniou_lang_temperature_label)
                            with gr.Column():
                                top_k_audiogen = gr.Slider(0, 500, step=1, value=250, label=biniou_lang_top_k_label)
                            with gr.Column():
                                top_p_audiogen = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label=biniou_lang_top_p_label)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_audiogen = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_audiogen = gr.Textbox(value="audiogen", visible=False, interactive=False)
                                del_ini_btn_audiogen = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_audiogen.value) else False)
                                save_ini_btn_audiogen.click(
                                    fn=write_ini_audiogen, 
                                    inputs=[
                                        module_name_audiogen, 
                                        model_audiogen,
                                        duration_audiogen,
                                        cfg_coef_audiogen,
                                        num_batch_audiogen,
                                        use_sampling_audiogen,
                                        temperature_audiogen,
                                        top_k_audiogen,
                                        top_p_audiogen,
                                        ]
                                    )
                                save_ini_btn_audiogen.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_audiogen.click(fn=lambda: del_ini_btn_audiogen.update(interactive=True), outputs=del_ini_btn_audiogen)
                                del_ini_btn_audiogen.click(fn=lambda: del_ini(module_name_audiogen.value))
                                del_ini_btn_audiogen.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_audiogen.click(fn=lambda: del_ini_btn_audiogen.update(interactive=False), outputs=del_ini_btn_audiogen)
                        if test_ini_exist(module_name_audiogen.value):
                            with open(f".ini/{module_name_audiogen.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            prompt_audiogen = gr.Textbox(label=biniou_lang_tab_audiogen_prompt_label, lines=2, max_lines=2, show_copy_button=True, placeholder=biniou_lang_tab_audiogen_prompt_placeholder)
                        with gr.Column():
                            out_audiogen = gr.Audio(label=biniou_lang_tab_audiogen_output, type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_audiogen = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():                            
                            btn_audiogen_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_audiogen_cancel.click(fn=initiate_stop_audiogen, inputs=None, outputs=None)
                        with gr.Column():
                            btn_audiogen_clear_input = gr.ClearButton(components=prompt_audiogen, value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_audiogen_clear_output = gr.ClearButton(components=out_audiogen, value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_audiogen.click(
                            fn=music_audiogen,
                            inputs=[
                                prompt_audiogen,
                                model_audiogen,
                                duration_audiogen,
                                num_batch_audiogen,
                                temperature_audiogen,
                                top_k_audiogen,
                                top_p_audiogen,
                                use_sampling_audiogen,
                                cfg_coef_audiogen,
                            ],
                            outputs=out_audiogen,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        audiogen_musicgen_mel = gr.Button(f"🎶 >> {biniou_lang_tab_musicgen_mel}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        audiogen_musicgen_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen}")
                                        audiogen_musicgen_mel_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicgen_mel}")
                                        audiogen_musicldm_input = gr.Button(f"✍️ >> {biniou_lang_tab_musicldm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# Harmonai
                with gr.TabItem(f"{biniou_lang_tab_harmonai} 🔊", id=35) as tab_harmonai:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_harmonai}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_harmonai_about_desc}<a href='https://www.harmonai.org/' target='_blank'>Harmonai</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_harmonai_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_harmonai_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_harmonai)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_harmonai_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_harmonai = gr.Dropdown(choices=model_list_harmonai, value=model_list_harmonai[4], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                steps_harmonai = gr.Slider(1, biniou_global_steps_max, step=1, value=50, label=biniou_lang_steps_label, info=biniou_lang_tab_audio_batch_size_info)
                            with gr.Column():
                                seed_harmonai = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                length_harmonai = gr.Slider(1, 1200, value=5, step=1, label=biniou_lang_audio_length_label)
                            with gr.Column():
                                batch_size_harmonai = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_audio_batch_size_info)
                            with gr.Column():
                                batch_repeat_harmonai = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_harmonai = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_harmonai = gr.Textbox(value="harmonai", visible=False, interactive=False)
                                del_ini_btn_harmonai = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_harmonai.value) else False)
                                save_ini_btn_harmonai.click(
                                    fn=write_ini_harmonai, 
                                    inputs=[
                                        module_name_harmonai,
                                        model_harmonai,
                                        steps_harmonai,
                                        seed_harmonai,
                                        length_harmonai,
                                        batch_size_harmonai,
                                        batch_repeat_harmonai,
                                        ]
                                    )
                                save_ini_btn_harmonai.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_harmonai.click(fn=lambda: del_ini_btn_harmonai.update(interactive=True), outputs=del_ini_btn_harmonai)
                                del_ini_btn_harmonai.click(fn=lambda: del_ini(module_name_harmonai.value))
                                del_ini_btn_harmonai.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_harmonai.click(fn=lambda: del_ini_btn_harmonai.update(interactive=False), outputs=del_ini_btn_harmonai)
                        if test_ini_exist(module_name_harmonai.value) :
                            with open(f".ini/{module_name_harmonai.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        out_harmonai = gr.Audio(label="Output", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_harmonai = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_harmonai_clear_output = gr.ClearButton(components=out_harmonai, value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_harmonai.click(
                            fn=music_harmonai,
                            inputs=[
                            length_harmonai,
                            model_harmonai,
                            steps_harmonai,
                            seed_harmonai,
                            batch_size_harmonai,
                            batch_repeat_harmonai,
                            ],
                            outputs=out_harmonai,
                            show_progress="full",    
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        harmonai_musicgen_mel = gr.Button(f"🎶 >> {biniou_lang_tab_musicgen_mel}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Bark
                with gr.TabItem(f"{biniou_lang_tab_bark} 🗣️", id=36) as tab_bark:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_bark}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_bark_about_desc}<a href='https://github.com/suno-ai/bark' target='_blank'>Bark</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_bark_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_bark)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_bark_about_instruct}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_bark = gr.Dropdown(choices=model_list_bark, value=model_list_bark[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                voice_preset_bark = gr.Dropdown(choices=list(voice_preset_list_bark.keys()), value=list(voice_preset_list_bark.keys())[2], label="Voice")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_bark = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_bark = gr.Textbox(value="bark", visible=False, interactive=False)
                                del_ini_btn_bark = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_bark.value) else False)
                                save_ini_btn_bark.click(
                                    fn=write_ini_bark,
                                    inputs=[
                                        module_name_bark,
                                        model_bark,
                                        voice_preset_bark,
                                        ]
                                    )
                                save_ini_btn_bark.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_bark.click(fn=lambda: del_ini_btn_bark.update(interactive=True), outputs=del_ini_btn_bark)
                                del_ini_btn_bark.click(fn=lambda: del_ini(module_name_bark.value))
                                del_ini_btn_bark.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_bark.click(fn=lambda: del_ini_btn_bark.update(interactive=False), outputs=del_ini_btn_bark)
                        if test_ini_exist(module_name_bark.value) :
                            with open(f".ini/{module_name_bark.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            prompt_bark = gr.Textbox(label=biniou_lang_tab_bark_prompt_label, lines=2, max_lines=2, show_copy_button=True, placeholder=biniou_lang_tab_bark_prompt_placeholder)
                        with gr.Column():
                            out_bark = gr.Audio(label=biniou_lang_tab_bark_output, type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_bark = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary")
                        with gr.Column():
                            btn_bark_clear_input = gr.ClearButton(components=prompt_bark, value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_bark_clear_output = gr.ClearButton(components=out_bark, value=f"{biniou_lang_clear_outputs} 🧹")
                        btn_bark.click(
                            fn=music_bark,
                            inputs=[
                                prompt_bark,
                                model_bark,
                                voice_preset_bark,
                            ],
                            outputs=out_bark,
                            show_progress="full",
                        )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_audio_value)
                                        bark_musicgen_mel = gr.Button(f"🎶 >> {biniou_lang_tab_musicgen_mel}")
                                        gr.HTML(value=biniou_lang_send_text_value)
                                        bark_whisper = gr.Button(f"🗣️ >> {biniou_lang_tab_whisper}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Video
        with gr.TabItem(f"{biniou_lang_tab_video} 🎬", id=4) as tab_video:
            with gr.Tabs() as tabs_video:
# Modelscope
                if ram_size() >= 16 :
                    titletab_txt2vid_ms = f"{biniou_lang_tab_txt2vid_ms} 📼"
                else :
                    titletab_txt2vid_ms = f"{biniou_lang_tab_txt2vid_ms} ⛔"
                    
                with gr.TabItem(titletab_txt2vid_ms, id=41) as tab_txt2vid_ms:
                        
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2vid_ms}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_video_about_desc}<a href='https://github.com/modelscope/modelscope' target='_blank'>Modelscope</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_video_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2vid_ms)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_video_about_instruct}
                                </div>
                                """
                            )                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2vid_ms = gr.Dropdown(choices=model_list_txt2vid_ms, value=model_list_txt2vid_ms[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2vid_ms = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_video_steps_info)
                            with gr.Column():
                                sampler_txt2vid_ms = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2vid_ms = gr.Slider(0.1, 20.0, step=0.1, value=4.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                seed_txt2vid_ms = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                            with gr.Column():
                                num_frames_txt2vid_ms = gr.Slider(1, 1200, step=1, value=8, label=biniou_lang_video_length_label, info=biniou_lang_video_length_info)
                            with gr.Column():
                                num_fps_txt2vid_ms = gr.Slider(1, 120, step=1, value=8, label=biniou_lang_video_fps_label, info=biniou_lang_video_fps_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2vid_ms = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=576, label=biniou_lang_video_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2vid_ms = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=320, label=biniou_lang_video_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                num_prompt_txt2vid_ms = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2vid_ms = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2vid_ms = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2vid_ms = gr.Textbox(value="txt2vid_ms", visible=False, interactive=False)
                                del_ini_btn_txt2vid_ms = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2vid_ms.value) else False)
                                save_ini_btn_txt2vid_ms.click(
                                    fn=write_ini_txt2vid_ms, 
                                    inputs=[
                                        module_name_txt2vid_ms, 
                                        model_txt2vid_ms, 
                                        num_inference_step_txt2vid_ms,
                                        sampler_txt2vid_ms,
                                        guidance_scale_txt2vid_ms,
                                        seed_txt2vid_ms,
                                        num_frames_txt2vid_ms,
                                        num_fps_txt2vid_ms,
                                        width_txt2vid_ms,
                                        height_txt2vid_ms,
                                        num_prompt_txt2vid_ms,
                                        use_gfpgan_txt2vid_ms,
                                        ]
                                    )
                                save_ini_btn_txt2vid_ms.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2vid_ms.click(fn=lambda: del_ini_btn_txt2vid_ms.update(interactive=True), outputs=del_ini_btn_txt2vid_ms)
                                del_ini_btn_txt2vid_ms.click(fn=lambda: del_ini(module_name_txt2vid_ms.value))
                                del_ini_btn_txt2vid_ms.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2vid_ms.click(fn=lambda: del_ini_btn_txt2vid_ms.update(interactive=False), outputs=del_ini_btn_txt2vid_ms)
                        if test_ini_exist(module_name_txt2vid_ms.value) :
                            with open(f".ini/{module_name_txt2vid_ms.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2vid_ms = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_tab_video_prompt_info, placeholder=biniou_lang_tab_txt2vid_ms_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2vid_ms = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_tab_video_negprompt_info, placeholder=biniou_lang_tab_txt2vid_ms_negprompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                   output_type_txt2vid_ms = gr.Radio(choices=["mp4", "gif"], value="mp4", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
                                with gr.Column():
                                    gr.Number(visible=False)
                        with gr.Column():
                            out_txt2vid_ms = gr.Video(label=biniou_lang_video_generated_label, height=400, visible=True, interactive=False)
                            gif_out_txt2vid_ms = gr.Gallery(
                                label=biniou_lang_video_generated_gif_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=False
                            )
                    with gr.Row():
                        with gr.Column():
                            btn_txt2vid_ms = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_txt2vid_ms_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False)
                        with gr.Column():
                            btn_txt2vid_ms_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2vid_ms_cancel.click(fn=initiate_stop_txt2vid_ms, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2vid_ms_clear_input = gr.ClearButton(components=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_txt2vid_ms_clear_output = gr.ClearButton(components=[out_txt2vid_ms, gif_out_txt2vid_ms], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2vid_ms.click(
                                fn=video_txt2vid_ms,
                                inputs=[
                                    model_txt2vid_ms,
                                    sampler_txt2vid_ms,
                                    prompt_txt2vid_ms,
                                    negative_prompt_txt2vid_ms,
                                    output_type_txt2vid_ms,
                                    num_frames_txt2vid_ms,
                                    num_fps_txt2vid_ms,
                                    num_prompt_txt2vid_ms,
                                    guidance_scale_txt2vid_ms,
                                    num_inference_step_txt2vid_ms,
                                    height_txt2vid_ms,
                                    width_txt2vid_ms,
                                    seed_txt2vid_ms,
                                    use_gfpgan_txt2vid_ms,
                                ],
                                outputs=out_txt2vid_ms,
                                show_progress="full",
                            )
                            btn_txt2vid_ms_gif.click(
                                fn=video_txt2vid_ms,
                                inputs=[
                                    model_txt2vid_ms,
                                    sampler_txt2vid_ms,
                                    prompt_txt2vid_ms,
                                    negative_prompt_txt2vid_ms,
                                    output_type_txt2vid_ms,
                                    num_frames_txt2vid_ms,
                                    num_fps_txt2vid_ms,
                                    num_prompt_txt2vid_ms,
                                    guidance_scale_txt2vid_ms,
                                    num_inference_step_txt2vid_ms,
                                    height_txt2vid_ms,
                                    width_txt2vid_ms,
                                    seed_txt2vid_ms,
                                    use_gfpgan_txt2vid_ms,
                                ],
                                outputs=gif_out_txt2vid_ms,
                                show_progress="full",
                            )
                            output_type_txt2vid_ms.change(
                                fn=change_output_type_txt2vid_ms,
                                inputs=[
                                    output_type_txt2vid_ms,
                                ],
                                outputs=[
                                    out_txt2vid_ms,
                                    gif_out_txt2vid_ms,
                                    btn_txt2vid_ms,
                                    btn_txt2vid_ms_gif,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2vid_ms_vid2vid_ze = gr.Button(f"📼 >> {biniou_lang_tab_vid2vid_ze}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2vid_ms_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2vid_ms_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2vid_ms_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        txt2vid_ms_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        txt2vid_ms_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2vid_ms_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                                        txt2vid_ms_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Txt2vid_zero            
                with gr.TabItem(f"{biniou_lang_tab_txt2vid_ze} 📼", id=42) as tab_txt2vid_ze:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2vid_ze}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_video_about_desc}<a href='https://github.com/Picsart-AI-Research/Text2Video-Zero' target='_blank'>Text2Video-Zero</a>, <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> Models</br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_video_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2vid_ze)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_video_about_instruct}
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}
                                </div>
                                """
                            )                      
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2vid_ze = gr.Dropdown(choices=model_list_txt2vid_ze, value=model_list_txt2vid_ze[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2vid_ze = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_video_steps_info)
                            with gr.Column():
                                sampler_txt2vid_ze = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                            with gr.Column():
                                guidance_scale_txt2vid_ze = gr.Slider(0.1, 20.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                        with gr.Row():
                            with gr.Column():
                                seed_txt2vid_ze = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                            with gr.Column():
                                num_frames_txt2vid_ze = gr.Slider(1, 1200, step=1, value=8, label=biniou_lang_video_length_label, info=biniou_lang_video_length_info)
                            with gr.Column():
                                num_fps_txt2vid_ze = gr.Slider(1, 120, step=1, value=4, label=biniou_lang_video_fps_label, info=biniou_lang_video_fps_info)
                            with gr.Column():
                                num_chunks_txt2vid_ze = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_tab_video_chunks_label, info=biniou_lang_tab_txt2vid_ze_chunks_info)
                        with gr.Row():
                            with gr.Column():
                                width_txt2vid_ze = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=512, label=biniou_lang_video_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_txt2vid_ze = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=512, label=biniou_lang_video_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                num_videos_per_prompt_txt2vid_ze = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_video_batch_size_info, interactive=False)
                            with gr.Column():
                                num_prompt_txt2vid_ze = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Accordion(biniou_lang_tab_txt2vid_ze_avd_settings, open=False):
                            with gr.Row():
                                with gr.Column():
                                    motion_field_strength_x_txt2vid_ze = gr.Slider(0, 50, step=1, value=12, label=biniou_lang_tab_txt2vid_ze_strengthx_label, info=biniou_lang_tab_txt2vid_ze_strengthx_info)
                                with gr.Column():
                                    motion_field_strength_y_txt2vid_ze = gr.Slider(0, 50, step=1, value=12, label=biniou_lang_tab_txt2vid_ze_strengthy_label, info=biniou_lang_tab_txt2vid_ze_strengthy_info)
                                with gr.Column():
                                    timestep_t0_txt2vid_ze = gr.Slider(0, biniou_global_steps_max, step=1, value=7, label=biniou_lang_tab_txt2vid_ze_t0_label, interactive=False)
                                with gr.Column():
                                    timestep_t1_txt2vid_ze = gr.Slider(1, biniou_global_steps_max, step=1, value=8, label=biniou_lang_tab_txt2vid_ze_t1_label, interactive=False)
                                    num_inference_step_txt2vid_ze.change(set_timestep_vid_ze, inputs=[num_inference_step_txt2vid_ze, model_txt2vid_ze], outputs=[timestep_t0_txt2vid_ze, timestep_t1_txt2vid_ze])
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2vid_ze = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():    
                                tkme_txt2vid_ze = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2vid_ze = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2vid_ze = gr.Textbox(value="txt2vid_ze", visible=False, interactive=False)
                                del_ini_btn_txt2vid_ze = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2vid_ze.value) else False)
                                save_ini_btn_txt2vid_ze.click(
                                    fn=write_ini_txt2vid_ze, 
                                    inputs=[
                                        module_name_txt2vid_ze,
                                        model_txt2vid_ze, 
                                        num_inference_step_txt2vid_ze,
                                        sampler_txt2vid_ze,
                                        guidance_scale_txt2vid_ze,
                                        seed_txt2vid_ze,
                                        num_frames_txt2vid_ze,
                                        num_fps_txt2vid_ze,
                                        num_chunks_txt2vid_ze,
                                        width_txt2vid_ze,
                                        height_txt2vid_ze,
                                        num_videos_per_prompt_txt2vid_ze,
                                        num_prompt_txt2vid_ze,
                                        motion_field_strength_x_txt2vid_ze,
                                        motion_field_strength_y_txt2vid_ze,
                                        timestep_t0_txt2vid_ze,
                                        timestep_t1_txt2vid_ze,
                                        use_gfpgan_txt2vid_ze,
                                        tkme_txt2vid_ze,
                                        ]
                                    )
                                save_ini_btn_txt2vid_ze.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2vid_ze.click(fn=lambda: del_ini_btn_txt2vid_ze.update(interactive=True), outputs=del_ini_btn_txt2vid_ze)
                                del_ini_btn_txt2vid_ze.click(fn=lambda: del_ini(module_name_txt2vid_ze.value))
                                del_ini_btn_txt2vid_ze.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2vid_ze.click(fn=lambda: del_ini_btn_txt2vid_ze.update(interactive=False), outputs=del_ini_btn_txt2vid_ze)
                        if test_ini_exist(module_name_txt2vid_ze.value) :
                            with open(f".ini/{module_name_txt2vid_ze.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2vid_ze = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_tab_video_prompt_info, placeholder=biniou_lang_tab_txt2vid_ze_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2vid_ze = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_tab_video_negprompt_info, placeholder=biniou_lang_tab_txt2vid_ze_negprompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    output_type_txt2vid_ze = gr.Radio(choices=["mp4", "gif"], value="mp4", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
                                with gr.Column():
                                    gr.Number(visible=False)
                        model_txt2vid_ze.change(
                            fn=change_model_type_txt2vid_ze, 
                            inputs=[model_txt2vid_ze],
                            outputs=[
                                sampler_txt2vid_ze,
                                width_txt2vid_ze,
                                height_txt2vid_ze,
                                num_inference_step_txt2vid_ze,
                                guidance_scale_txt2vid_ze,
                                negative_prompt_txt2vid_ze,
                            ]
                        )
                        with gr.Column():
                            out_txt2vid_ze = gr.Video(label=biniou_lang_video_generated_label, height=400, visible=True, interactive=False)
                            gif_out_txt2vid_ze = gr.Gallery(
                                label=biniou_lang_video_generated_gif_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=False
                            )
                    with gr.Row():
                        with gr.Column():
                            btn_txt2vid_ze = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_txt2vid_ze_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False)
                        with gr.Column():                            
                            btn_txt2vid_ze_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_txt2vid_ze_cancel.click(fn=initiate_stop_txt2vid_ze, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2vid_ze_clear_input = gr.ClearButton(components=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_txt2vid_ze_clear_output = gr.ClearButton(components=[out_txt2vid_ze, gif_out_txt2vid_ze], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2vid_ze.click(
                                fn=video_txt2vid_ze,
                                inputs=[
                                    model_txt2vid_ze,
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
                                    timestep_t0_txt2vid_ze,
                                    timestep_t1_txt2vid_ze,
                                    prompt_txt2vid_ze,
                                    negative_prompt_txt2vid_ze,
                                    output_type_txt2vid_ze,
                                    nsfw_filter,
                                    num_chunks_txt2vid_ze,
                                    use_gfpgan_txt2vid_ze,
                                    tkme_txt2vid_ze,
                                ],
                                outputs=out_txt2vid_ze,
                                show_progress="full",
                            )
                            btn_txt2vid_ze_gif.click(
                                fn=video_txt2vid_ze,
                                inputs=[
                                    model_txt2vid_ze,
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
                                    timestep_t0_txt2vid_ze,
                                    timestep_t1_txt2vid_ze,
                                    prompt_txt2vid_ze,
                                    negative_prompt_txt2vid_ze,
                                    output_type_txt2vid_ze,
                                    nsfw_filter,
                                    num_chunks_txt2vid_ze,
                                    use_gfpgan_txt2vid_ze,
                                    tkme_txt2vid_ze,
                                ],
                                outputs=gif_out_txt2vid_ze,
                                show_progress="full",
                            )
                            output_type_txt2vid_ze.change(
                                fn=change_output_type_txt2vid_ze,
                                inputs=[
                                    output_type_txt2vid_ze,
                                ],
                                outputs=[
                                    out_txt2vid_ze,
                                    gif_out_txt2vid_ze,
                                    btn_txt2vid_ze,
                                    btn_txt2vid_ze_gif,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2vid_ze_vid2vid_ze = gr.Button(f"📼 >> {biniou_lang_tab_vid2vid_ze}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        txt2vid_ze_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        txt2vid_ze_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        txt2vid_ze_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        txt2vid_ze_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}") 
                                        txt2vid_ze_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}") 
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        txt2vid_ze_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        txt2vid_ze_animatediff_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_animatediff_lcm}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# animatediff
                if ram_size() >= 16 :
                    titletab_tab_animatediff_lcm = f"{biniou_lang_tab_animatediff_lcm} 📼"
                else :
                    titletab_tab_animatediff_lcm = f"{biniou_lang_tab_animatediff_lcm} ⛔"
                with gr.TabItem(titletab_tab_animatediff_lcm, id=43) as tab_animatediff_lcm:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_animatediff_lcm}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_video_about_desc}<a href='https://animatelcm.github.io/' target='_blank'>AnimateLCM</a> / <a href='https://huggingface.co/ByteDance/AnimateDiff-Lightning' target='_blank'>ByteDance/AnimateDiff-Lightning</a>, <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> Models</br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt_neg}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_video_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_animatediff_lcm)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_video_about_instruct}
                                <b>{biniou_lang_about_models}</b></br>
                                - {biniou_lang_tab_image_about_models_inst1}
                                </div>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_animatediff_lcm = gr.Dropdown(choices=model_list_animatediff_lcm, value=model_list_animatediff_lcm[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                model_adapters_animatediff_lcm = gr.Dropdown(choices=list(model_list_adapters_animatediff_lcm.keys()), value=list(model_list_adapters_animatediff_lcm.keys())[0], label=biniou_lang_tab_animatediff_adapter_label, info=biniou_lang_tab_animatediff_adapter_info)
                            with gr.Column():
                                num_inference_step_animatediff_lcm = gr.Slider(1, biniou_global_steps_max, step=1, value=4, label=biniou_lang_steps_label, info=biniou_lang_video_steps_info)
                            with gr.Column():
                                sampler_animatediff_lcm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value="LCM", label=biniou_lang_sampler_label, info=biniou_lang_sampler_info, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_animatediff_lcm = gr.Slider(0.1, 20.0, step=0.1, value=2.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                seed_animatediff_lcm = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                            with gr.Column():
                                num_frames_animatediff_lcm = gr.Slider(1, 32, step=1, value=16, label=biniou_lang_video_length_label, info=biniou_lang_video_length_info)
                            with gr.Column():
                                num_fps_animatediff_lcm = gr.Slider(1, 120, step=1, value=8, label=biniou_lang_video_fps_label, info=biniou_lang_video_fps_info)
                        with gr.Row():
                            with gr.Column():
                                width_animatediff_lcm = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label=biniou_lang_video_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_animatediff_lcm = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label=biniou_lang_video_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                num_videos_per_prompt_animatediff_lcm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_video_batch_size_info, interactive=False)
                            with gr.Column():
                                num_prompt_animatediff_lcm = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_animatediff_lcm = gr.Checkbox(value=False, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info, interactive=False)
                            with gr.Column():
                                tkme_animatediff_lcm = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info, interactive=False)
                            with gr.Column():
                                clipskip_animatediff_lcm = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_clipskip_label, info=biniou_lang_clipskip_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_animatediff_lcm = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_animatediff_lcm = gr.Textbox(value="animatediff_lcm", visible=False, interactive=False)
                                del_ini_btn_animatediff_lcm = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_animatediff_lcm.value) else False)
                                save_ini_btn_animatediff_lcm.click(
                                    fn=write_ini_animatediff_lcm,
                                    inputs=[
                                        module_name_animatediff_lcm,
                                        model_animatediff_lcm,
                                        model_adapters_animatediff_lcm,
                                        num_inference_step_animatediff_lcm,
                                        sampler_animatediff_lcm,
                                        guidance_scale_animatediff_lcm,
                                        seed_animatediff_lcm,
                                        num_frames_animatediff_lcm,
                                        num_fps_animatediff_lcm,
                                        width_animatediff_lcm,
                                        height_animatediff_lcm,
                                        num_videos_per_prompt_animatediff_lcm,
                                        num_prompt_animatediff_lcm,
                                        use_gfpgan_animatediff_lcm,
                                        tkme_animatediff_lcm,
                                        clipskip_animatediff_lcm,
                                        ]
                                    )
                                save_ini_btn_animatediff_lcm.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_animatediff_lcm.click(fn=lambda: del_ini_btn_animatediff_lcm.update(interactive=True), outputs=del_ini_btn_animatediff_lcm)
                                del_ini_btn_animatediff_lcm.click(fn=lambda: del_ini(module_name_animatediff_lcm.value))
                                del_ini_btn_animatediff_lcm.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_animatediff_lcm.click(fn=lambda: del_ini_btn_animatediff_lcm.update(interactive=False), outputs=del_ini_btn_animatediff_lcm)
                        if test_ini_exist(module_name_animatediff_lcm.value) :
                            with open(f".ini/{module_name_animatediff_lcm.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_animatediff_lcm = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_tab_video_prompt_info, placeholder=biniou_lang_tab_animatediff_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_animatediff_lcm = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_tab_video_negprompt_info, placeholder=biniou_lang_tab_animatediff_negprompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    output_type_animatediff_lcm = gr.Radio(choices=["mp4", "gif"], value="mp4", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
                                with gr.Column():
                                    gr.Number(visible=False)
                        model_animatediff_lcm.change(
                            fn=change_model_type_animatediff_lcm,
                            inputs=[
                                model_animatediff_lcm,
                                model_adapters_animatediff_lcm,
                            ],
                            outputs=[
                                sampler_animatediff_lcm,
                                width_animatediff_lcm,
                                height_animatediff_lcm,
                                num_inference_step_animatediff_lcm,
                                guidance_scale_animatediff_lcm,
                                negative_prompt_animatediff_lcm,
                            ]
                        )
                        model_adapters_animatediff_lcm.change(
                            fn=change_model_type_animatediff_lcm,
                            inputs=[
                                model_animatediff_lcm,
                                model_adapters_animatediff_lcm,
                            ],
                            outputs=[
                                sampler_animatediff_lcm,
                                width_animatediff_lcm,
                                height_animatediff_lcm,
                                num_inference_step_animatediff_lcm,
                                guidance_scale_animatediff_lcm,
                                negative_prompt_animatediff_lcm,
                            ]
                        )
                        with gr.Column(scale=1):
                            out_animatediff_lcm = gr.Video(label=biniou_lang_video_generated_label, height=400, visible=True, interactive=False)
                            gif_out_animatediff_lcm = gr.Gallery(
                                label=biniou_lang_video_generated_gif_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=False
                            )
                    with gr.Row():
                        with gr.Column():
                            btn_animatediff_lcm = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_animatediff_lcm_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False)
                        with gr.Column():
                            btn_animatediff_lcm_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_animatediff_lcm_cancel.click(fn=initiate_stop_animatediff_lcm, inputs=None, outputs=None)
                        with gr.Column():
                            btn_animatediff_lcm_clear_input = gr.ClearButton(components=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_animatediff_lcm_clear_output = gr.ClearButton(components=[out_animatediff_lcm, gif_out_animatediff_lcm], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_animatediff_lcm.click(
                                fn=video_animatediff_lcm,
                                inputs=[
                                    model_animatediff_lcm,
                                    model_adapters_animatediff_lcm,
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
                                    output_type_animatediff_lcm,
                                    nsfw_filter,
                                    use_gfpgan_animatediff_lcm,
                                    tkme_animatediff_lcm,
                                    clipskip_animatediff_lcm,
                                ],
                                outputs=out_animatediff_lcm,
                                show_progress="full",
                            )
                            btn_animatediff_lcm_gif.click(
                                fn=video_animatediff_lcm,
                                inputs=[
                                    model_animatediff_lcm,
                                    model_adapters_animatediff_lcm,
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
                                    output_type_animatediff_lcm,
                                    nsfw_filter,
                                    use_gfpgan_animatediff_lcm,
                                    tkme_animatediff_lcm,
                                    clipskip_animatediff_lcm,
                                ],
                                outputs=gif_out_animatediff_lcm,
                                show_progress="full",
                            )
                            output_type_animatediff_lcm.change(
                                fn=change_output_type_animatediff_lcm,
                                inputs=[
                                    output_type_animatediff_lcm,
                                ],
                                outputs=[
                                    out_animatediff_lcm,
                                    gif_out_animatediff_lcm,
                                    btn_animatediff_lcm,
                                    btn_animatediff_lcm_gif,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        animatediff_lcm_vid2vid_ze = gr.Button(f"📼 >> {biniou_lang_tab_vid2vid_ze}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        animatediff_lcm_txt2img_sd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_sd}")
                                        animatediff_lcm_txt2img_kd_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_kd}")
                                        animatediff_lcm_txt2img_lcm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_lcm}")
                                        animatediff_lcm_txt2img_mjm_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_mjm}")
                                        animatediff_lcm_txt2img_paa_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2img_paa}")
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        animatediff_lcm_txt2vid_ms_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ms}")
                                        animatediff_lcm_txt2vid_ze_input = gr.Button(f"✍️ >> {biniou_lang_tab_txt2vid_ze}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# img2vid
                if ram_size() >= 16 :
                    titletab_img2vid = f"{biniou_lang_tab_img2vid} 📼"
                else :
                    titletab_img2vid = f"{biniou_lang_tab_img2vid} ⛔"
                with gr.TabItem(titletab_img2vid, id=44) as tab_img2vid:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_img2vid}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_img2vid_about_desc}<a href='https://stability.ai/news/stable-video-diffusion-open-ai-video-model' target='_blank'>Stable Video Diffusion</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_image}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_video_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_img2vid)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_img2vid_about_instruct}
                                </br>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2vid = gr.Dropdown(choices=model_list_img2vid, value=model_list_img2vid[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_steps_img2vid = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label=biniou_lang_steps_label, info=biniou_lang_video_steps_info)
                            with gr.Column():
                                sampler_img2vid = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[5], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                min_guidance_scale_img2vid = gr.Slider(0.1, 20.0, step=0.1, value=1.0, label=biniou_lang_tab_animatediff_min_cfg_label, info=biniou_lang_tab_animatediff_min_cfg_info)
                            with gr.Column():
                                max_guidance_scale_img2vid = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label=biniou_lang_tab_animatediff_max_cfg_label, info=biniou_lang_tab_animatediff_max_cfg_info)
                            with gr.Column():
                                seed_img2vid = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                num_frames_img2vid = gr.Slider(1, 1200, step=1, value=14, label=biniou_lang_video_length_label, info=biniou_lang_video_length_info)
                            with gr.Column():
                                num_fps_img2vid = gr.Slider(1, 120, step=1, value=7, label=biniou_lang_video_fps_label, info=biniou_lang_video_fps_info)
                            with gr.Column():
                                decode_chunk_size_img2vid = gr.Slider(1, 32, step=1, value=7, label=biniou_lang_tab_video_chunks_label, info=biniou_lang_tab_img2vid_chunks_info)
                        with gr.Row():
                            with gr.Column():
                                width_img2vid = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sdxl_width, label=biniou_lang_video_width_label, info=biniou_lang_image_width_info)
                            with gr.Column():
                                height_img2vid = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=576, label=biniou_lang_video_height_label, info=biniou_lang_image_height_info)
                            with gr.Column():
                                num_videos_per_prompt_img2vid = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_video_batch_size_info, interactive=False)
                            with gr.Column():
                                num_prompt_img2vid = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
#                       with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                motion_bucket_id_img2vid = gr.Slider(0, 256, step=1, value=127, label=biniou_lang_tab_img2vid_bucket_label, info=biniou_lang_tab_img2vid_bucket_info)
                            with gr.Column():
                                noise_aug_strength_img2vid = gr.Slider(0.01, 1.0, step=0.01, value=0.02, label=biniou_lang_tab_img2vid_noise_label, info=biniou_lang_tab_img2vid_noise_info)
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_img2vid = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info, visible=False)
                            with gr.Column():
                                tkme_img2vid = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info, visible=False)
                        model_img2vid.change(fn=change_model_type_img2vid, inputs=model_img2vid, outputs=num_frames_img2vid)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2vid = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_img2vid = gr.Textbox(value="img2vid", visible=False, interactive=False)
                                del_ini_btn_img2vid = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_img2vid.value) else False)
                                save_ini_btn_img2vid.click(
                                    fn=write_ini_img2vid,
                                    inputs=[
                                        module_name_img2vid,
                                        model_img2vid,
                                        num_inference_steps_img2vid,
                                        sampler_img2vid,
                                        min_guidance_scale_img2vid,
                                        max_guidance_scale_img2vid,
                                        seed_img2vid,
                                        num_frames_img2vid,
                                        num_fps_img2vid,
                                        decode_chunk_size_img2vid,
                                        width_img2vid,
                                        height_img2vid,
                                        num_prompt_img2vid,
                                        num_videos_per_prompt_img2vid,
                                        motion_bucket_id_img2vid,
                                        noise_aug_strength_img2vid,
                                        use_gfpgan_img2vid,
                                        tkme_img2vid,
                                        ]
                                    )
                                save_ini_btn_img2vid.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_img2vid.click(fn=lambda: del_ini_btn_img2vid.update(interactive=True), outputs=del_ini_btn_img2vid)
                                del_ini_btn_img2vid.click(fn=lambda: del_ini(module_name_img2vid.value))
                                del_ini_btn_img2vid.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_img2vid.click(fn=lambda: del_ini_btn_img2vid.update(interactive=False), outputs=del_ini_btn_img2vid)
                        if test_ini_exist(module_name_img2vid.value) :
                            with open(f".ini/{module_name_img2vid.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        img_img2vid = gr.Image(label=biniou_lang_img_input_label, type="filepath", height=275)
                                        img_img2vid.change(image_upload_event, inputs=img_img2vid, outputs=[width_img2vid, height_img2vid])
                                    with gr.Row():
                                        with gr.Column():
                                            output_type_img2vid = gr.Radio(choices=["mp4", "gif"], value="mp4", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
#                                        with gr.Column():
#                                            gr.Number(visible=False)
                        with gr.Column():
                            out_img2vid = gr.Video(label=biniou_lang_video_generated_label, height=400, visible=True, interactive=False)
                            gif_out_img2vid = gr.Gallery(
                                label=biniou_lang_video_generated_gif_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=False
                            )
                    with gr.Row():
                        with gr.Column():
                            btn_img2vid = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_img2vid_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False)
                        with gr.Column():
                            btn_img2vid_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_img2vid_cancel.click(fn=initiate_stop_img2vid, inputs=None, outputs=None)
                        with gr.Column():
                            btn_img2vid_clear_input = gr.ClearButton(components=img_img2vid, value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_img2vid_clear_output = gr.ClearButton(components=[out_img2vid, gif_out_img2vid], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_img2vid.click(
                                fn=video_img2vid,
                                inputs=[
                                    model_img2vid,
                                    num_inference_steps_img2vid,
                                    sampler_img2vid,
                                    min_guidance_scale_img2vid,
                                    max_guidance_scale_img2vid,
                                    seed_img2vid,
                                    num_frames_img2vid,
                                    num_fps_img2vid,
                                    decode_chunk_size_img2vid,
                                    width_img2vid,
                                    height_img2vid,
                                    num_prompt_img2vid,
                                    num_videos_per_prompt_img2vid,
                                    motion_bucket_id_img2vid,
                                    noise_aug_strength_img2vid,
                                    nsfw_filter,
                                    img_img2vid,
                                    output_type_img2vid,
                                    use_gfpgan_img2vid,
                                    tkme_img2vid,
                                ],
                                outputs=out_img2vid,
                                show_progress="full",
                            )

                            btn_img2vid_gif.click(
                                fn=video_img2vid,
                                inputs=[
                                    model_img2vid,
                                    num_inference_steps_img2vid,
                                    sampler_img2vid,
                                    min_guidance_scale_img2vid,
                                    max_guidance_scale_img2vid,
                                    seed_img2vid,
                                    num_frames_img2vid,
                                    num_fps_img2vid,
                                    decode_chunk_size_img2vid,
                                    width_img2vid,
                                    height_img2vid,
                                    num_prompt_img2vid,
                                    num_videos_per_prompt_img2vid,
                                    motion_bucket_id_img2vid,
                                    noise_aug_strength_img2vid,
                                    nsfw_filter,
                                    img_img2vid,
                                    output_type_img2vid,
                                    use_gfpgan_img2vid,
                                    tkme_img2vid,
                                ],
                                outputs=gif_out_img2vid,
                                show_progress="full",
                            )

                            output_type_img2vid.change(
                                fn=change_output_type_img2vid,
                                inputs=[
                                    output_type_img2vid,
                                ],
                                outputs=[
                                    out_img2vid,
                                    gif_out_img2vid,
                                    btn_img2vid,
                                    btn_img2vid_gif,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                                        gr.HTML(value=biniou_lang_send_video_value)
                                        img2vid_vid2vid_ze = gr.Button(f"📼 >> {biniou_lang_tab_vid2vid_ze}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

# vid2vid_ze    
                if ram_size() >= 16 :
                    titletab_vid2vid_ze = f"{biniou_lang_tab_vid2vid_ze} 🖌️"
                else :
                    titletab_vid2vid_ze = f"{biniou_lang_tab_vid2vid_ze} ⛔"

                with gr.TabItem(titletab_vid2vid_ze, id=45) as tab_vid2vid_ze:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_vid2vid_ze}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_vid2vid_ze_about_desc}<a href='https://github.com/timothybrooks/instruct-pix2pix' target='_blank'>Instructpix2pix</a>, <a href='https://github.com/Picsart-AI-Research/Text2Video-Zero' target='_blank'>Text2Video-Zero</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_tab_vid2vid_ze_about_input_text}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_video_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_vid2vid_ze)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_vid2vid_ze_about_instruct}
                                <b>{biniou_lang_about_examples}</b><a href='https://www.timothybrooks.com/instruct-pix2pix/' target='_blank'>Instructpix2pix : Learning to Follow Image Editing Instructions</a>
                                </div>
                                """
                            )                
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_vid2vid_ze = gr.Dropdown(choices=model_list_vid2vid_ze, value=model_list_vid2vid_ze[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_vid2vid_ze = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_vid2vid_ze = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_vid2vid_ze = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                image_guidance_scale_vid2vid_ze = gr.Slider(0.0, 10.0, step=0.1, value=1.5, label=biniou_lang_imgcfg_label, info=biniou_lang_tab_vid2vid_ze_imgcfg_info)
                            with gr.Column():
                                num_images_per_prompt_vid2vid_ze = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_tab_video_batch_size_info, interactive=False)
                            with gr.Column():
                                num_prompt_vid2vid_ze = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                width_vid2vid_ze = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label=biniou_lang_image_width_label, info=biniou_lang_image_width_info, interactive=False)
                            with gr.Column():
                                height_vid2vid_ze = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label=biniou_lang_image_height_label, info=biniou_lang_image_height_info, interactive=False)
                            with gr.Column():
                                seed_vid2vid_ze = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info)
                        with gr.Row():
                            with gr.Column():
                                num_frames_vid2vid_ze = gr.Slider(0, 1200, step=1, value=8, label=biniou_lang_video_length_label, info=biniou_lang_tab_vid2vid_ze_video_length_info)
                            with gr.Column():
                                num_fps_vid2vid_ze = gr.Slider(1, 120, step=1, value=4, label=biniou_lang_video_fps_label, info=biniou_lang_video_fps_info)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_vid2vid_ze = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_gfpgan_label, info=biniou_lang_gfpgan_info)
                            with gr.Column():
                                tkme_vid2vid_ze = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tkme_label, info=biniou_lang_tkme_info)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_vid2vid_ze = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_vid2vid_ze = gr.Textbox(value="vid2vid_ze", visible=False, interactive=False)
                                del_ini_btn_vid2vid_ze = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_vid2vid_ze.value) else False)
                                save_ini_btn_vid2vid_ze.click(
                                    fn=write_ini_vid2vid_ze, 
                                    inputs=[
                                        module_name_vid2vid_ze, 
                                        model_vid2vid_ze, 
                                        num_inference_step_vid2vid_ze,
                                        sampler_vid2vid_ze,
                                        guidance_scale_vid2vid_ze,
                                        image_guidance_scale_vid2vid_ze, 
                                        num_images_per_prompt_vid2vid_ze,
                                        num_prompt_vid2vid_ze,
                                        width_vid2vid_ze,
                                        height_vid2vid_ze,
                                        seed_vid2vid_ze,
                                        num_frames_vid2vid_ze,
                                        num_fps_vid2vid_ze,
                                        use_gfpgan_vid2vid_ze,
                                        tkme_vid2vid_ze,
                                        ]
                                    )
                                save_ini_btn_vid2vid_ze.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_vid2vid_ze.click(fn=lambda: del_ini_btn_vid2vid_ze.update(interactive=True), outputs=del_ini_btn_vid2vid_ze)
                                del_ini_btn_vid2vid_ze.click(fn=lambda: del_ini(module_name_vid2vid_ze.value))
                                del_ini_btn_vid2vid_ze.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_vid2vid_ze.click(fn=lambda: del_ini_btn_vid2vid_ze.update(interactive=False), outputs=del_ini_btn_vid2vid_ze)
                        if test_ini_exist(module_name_vid2vid_ze.value) :
                            with open(f".ini/{module_name_vid2vid_ze.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                             vid_vid2vid_ze = gr.Video(label=biniou_lang_tab_vid2vid_ze_vid_label, height=400)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_vid2vid_ze = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_tab_vid2vid_ze_prompt_info, placeholder=biniou_lang_tab_vid2vid_ze_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_vid2vid_ze = gr.Textbox(lines=3, max_lines=3, show_copy_button=True, label=biniou_lang_negprompt_label, info=biniou_lang_tab_vid2vid_ze_negprompt_info, placeholder=biniou_lang_tab_vid2vid_ze_negprompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    output_type_vid2vid_ze = gr.Radio(choices=["mp4", "gif"], value="mp4", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
                                with gr.Column():
                                    gr.Number(visible=False)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_vid2vid_ze = gr.Video(label=biniou_lang_video_generated_label, height=400, visible=True, interactive=False)
                                    gif_out_vid2vid_ze = gr.Gallery(
                                        label=biniou_lang_video_generated_gif_label,
                                        show_label=True,
                                        elem_id="gallery",
                                        columns=3,
                                        height=400,
                                        visible=False
                                    )
                                    gs_out_vid2vid_ze = gr.State()
                    with gr.Row():
                        with gr.Column():
                            btn_vid2vid_ze = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_vid2vid_ze_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False)
                        with gr.Column():                            
                            btn_vid2vid_ze_cancel = gr.Button(f"{biniou_lang_cancel} 🛑", variant="stop")
                            btn_vid2vid_ze_cancel.click(fn=initiate_stop_vid2vid_ze, inputs=None, outputs=None)
                        with gr.Column():
                            btn_vid2vid_ze_clear_input = gr.ClearButton(components=[vid_vid2vid_ze, prompt_vid2vid_ze, negative_prompt_vid2vid_ze], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():                            
                            btn_vid2vid_ze_clear_output = gr.ClearButton(components=[out_vid2vid_ze, gif_out_vid2vid_ze, gs_out_vid2vid_ze], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_vid2vid_ze.click(
                                fn=image_vid2vid_ze,
                                inputs=[
                                    model_vid2vid_ze,
                                    sampler_vid2vid_ze,
                                    vid_vid2vid_ze,
                                    prompt_vid2vid_ze,
                                    negative_prompt_vid2vid_ze,
                                    output_type_vid2vid_ze,
                                    num_images_per_prompt_vid2vid_ze,
                                    num_prompt_vid2vid_ze,
                                    guidance_scale_vid2vid_ze,
                                    image_guidance_scale_vid2vid_ze,
                                    num_inference_step_vid2vid_ze,
                                    height_vid2vid_ze,
                                    width_vid2vid_ze,
                                    seed_vid2vid_ze,
                                    num_frames_vid2vid_ze,
                                    num_fps_vid2vid_ze,
                                    use_gfpgan_vid2vid_ze,
                                    nsfw_filter,
                                    tkme_vid2vid_ze,
                                ],
                                outputs=[out_vid2vid_ze, gs_out_vid2vid_ze],
                                show_progress="full",
                            )
                            btn_vid2vid_ze_gif.click(
                                fn=image_vid2vid_ze,
                                inputs=[
                                    model_vid2vid_ze,
                                    sampler_vid2vid_ze,
                                    vid_vid2vid_ze,
                                    prompt_vid2vid_ze,
                                    negative_prompt_vid2vid_ze,
                                    output_type_vid2vid_ze,
                                    num_images_per_prompt_vid2vid_ze,
                                    num_prompt_vid2vid_ze,
                                    guidance_scale_vid2vid_ze,
                                    image_guidance_scale_vid2vid_ze,
                                    num_inference_step_vid2vid_ze,
                                    height_vid2vid_ze,
                                    width_vid2vid_ze,
                                    seed_vid2vid_ze,
                                    num_frames_vid2vid_ze,
                                    num_fps_vid2vid_ze,
                                    use_gfpgan_vid2vid_ze,
                                    nsfw_filter,
                                    tkme_vid2vid_ze,
                                ],
                                outputs=[gif_out_vid2vid_ze, gs_out_vid2vid_ze],
                                show_progress="full",
                            )
                            output_type_vid2vid_ze.change(
                                fn=change_output_type_vid2vid_ze,
                                inputs=[
                                    output_type_vid2vid_ze,
                                ],
                                outputs=[
                                    out_vid2vid_ze,
                                    gif_out_vid2vid_ze,
                                    btn_vid2vid_ze,
                                    btn_vid2vid_ze_gif,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                                        gr.HTML(value=biniou_lang_send_image_value)
                                        vid2vid_ze_pix2pix = gr.Button(f"✍️ >> {biniou_lang_tab_pix2pix}")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# 3d
        with gr.TabItem(f"{biniou_lang_tab_3d} 🧊", id=5) as tab_3d:
            with gr.Tabs() as tabs_3d:
# txt2shape
                with gr.TabItem(f"{biniou_lang_tab_txt2shape} 🧊", id=51) as tab_txt2shape:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_txt2shape}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_txt2shape_about_desc}<a href='https://github.com/openai/shap-e' target='_blank'>Shap-E</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_prompt}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_3d_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_txt2shape)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_txt2shape_about_instruct}
                                </br>
                                """
                            ) 
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2shape = gr.Dropdown(choices=model_list_txt2shape, value=model_list_txt2shape[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_txt2shape = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_txt2shape = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[11], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2shape = gr.Slider(0.1, 50.0, step=0.1, value=15.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_txt2shape = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info, interactive=False)
                            with gr.Column():
                                num_prompt_txt2shape = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                frame_size_txt2shape = gr.Slider(0, biniou_global_width_max_img_create, step=8, value=64, label=biniou_lang_3d_frame_size_label, info=biniou_lang_3d_frame_size_info)
                            with gr.Column():
                                seed_txt2shape = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info, interactive=False) 
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2shape = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_txt2shape = gr.Textbox(value="txt2shape", visible=False, interactive=False)
                                del_ini_btn_txt2shape = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_txt2shape.value) else False)
                                save_ini_btn_txt2shape.click(
                                    fn=write_ini_txt2shape, 
                                    inputs=[
                                        module_name_txt2shape, 
                                        model_txt2shape, 
                                        num_inference_step_txt2shape,
                                        sampler_txt2shape,
                                        guidance_scale_txt2shape,
                                        num_images_per_prompt_txt2shape,
                                        num_prompt_txt2shape,
                                        frame_size_txt2shape,
                                        seed_txt2shape,
                                        ]
                                    )
                                save_ini_btn_txt2shape.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_txt2shape.click(fn=lambda: del_ini_btn_txt2shape.update(interactive=True), outputs=del_ini_btn_txt2shape)
                                del_ini_btn_txt2shape.click(fn=lambda: del_ini(module_name_txt2shape.value))
                                del_ini_btn_txt2shape.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_txt2shape.click(fn=lambda: del_ini_btn_txt2shape.update(interactive=False), outputs=del_ini_btn_txt2shape)
                        if test_ini_exist(module_name_txt2shape.value) :
                            with open(f".ini/{module_name_txt2shape.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2shape = gr.Textbox(lines=12, max_lines=12, show_copy_button=True, label=biniou_lang_prompt_label, info=biniou_lang_image_prompt_info, placeholder=biniou_lang_tab_txt2shape_prompt_placeholder)
                            with gr.Row():
                                with gr.Column():
                                    output_type_txt2shape = gr.Radio(choices=["gif", "mesh"], value="gif", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
                        with gr.Column(scale=2):
                            out_txt2shape = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=True
                            )    
                            out_size_txt2shape = gr.Number(value=64, visible=False)
                            mesh_out_txt2shape = gr.Model3D(
                                label=biniou_lang_3d_generated_label,
#                                clear_color=[255.0, 255.0, 255.0, 255.0],
                                height=400,
                                zoom_speed=5,
                                visible=False,
#                                interactive=False,
                            )    
                            mesh_out_size_txt2shape = gr.Number(value=512, visible=False)
                            bool_output_type_txt2shape = gr.Checkbox(value=True, visible=False, interactive=False) 
                            gs_out_txt2shape = gr.State()
                            sel_out_txt2shape = gr.Number(precision=0, visible=False)
                            out_txt2shape.select(get_select_index, None, sel_out_txt2shape)
                            gs_mesh_out_txt2shape = gr.Textbox(visible=False)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_txt2shape_gif = gr.Button(f"{biniou_lang_image_zip} 💾", visible=True) 
                                    download_btn_txt2shape_mesh = gr.Button("Zip model 💾", visible=False) 
                                with gr.Column():
                                    download_file_txt2shape = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_txt2shape_gif.click(fn=zip_download_file_txt2shape, inputs=[out_txt2shape], outputs=[download_file_txt2shape, download_file_txt2shape]) 
                                    download_btn_txt2shape_mesh.click(fn=zip_mesh_txt2shape, inputs=[gs_mesh_out_txt2shape], outputs=[download_file_txt2shape, download_file_txt2shape]) 
                    with gr.Row():
                        with gr.Column():
                            btn_txt2shape_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_txt2shape_mesh = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False) 
                        with gr.Column():
                            btn_txt2shape_clear_input = gr.ClearButton(components=[prompt_txt2shape], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_txt2shape_clear_output = gr.ClearButton(components=[out_txt2shape, gs_out_txt2shape, mesh_out_txt2shape, gs_mesh_out_txt2shape], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_txt2shape_gif.click(fn=hide_download_file_txt2shape, inputs=None, outputs=download_file_txt2shape)
                            btn_txt2shape_gif.click(
                            fn=image_txt2shape,
                            inputs=[
                                model_txt2shape,
                                sampler_txt2shape,
                                prompt_txt2shape,
                                num_images_per_prompt_txt2shape,
                                num_prompt_txt2shape,
                                guidance_scale_txt2shape,
                                num_inference_step_txt2shape,
                                frame_size_txt2shape,
                                seed_txt2shape,
                                output_type_txt2shape,
                                nsfw_filter,
                                ],
                                outputs=[out_txt2shape, gs_out_txt2shape],
                                show_progress="full",
                            )
                            btn_txt2shape_mesh.click(fn=hide_download_file_txt2shape, inputs=None, outputs=download_file_txt2shape)
                            btn_txt2shape_mesh.click(
                            fn=image_txt2shape,
                            inputs=[
                                model_txt2shape,
                                sampler_txt2shape,
                                prompt_txt2shape,
                                num_images_per_prompt_txt2shape,
                                num_prompt_txt2shape,
                                guidance_scale_txt2shape,
                                num_inference_step_txt2shape,
                                frame_size_txt2shape,
                                seed_txt2shape,
                                output_type_txt2shape,
                                nsfw_filter,
                                ],
                                outputs=[mesh_out_txt2shape, gs_mesh_out_txt2shape],
                                show_progress="full",
                            )
                            output_type_txt2shape.change(
                                fn=change_output_type_txt2shape,
                                inputs=[
                                    output_type_txt2shape,
                                    out_size_txt2shape,
                                    mesh_out_size_txt2shape,
                                    ],
                                outputs=[
                                    out_txt2shape,
                                    mesh_out_txt2shape,
                                    bool_output_type_txt2shape,
                                    btn_txt2shape_gif,
                                    btn_txt2shape_mesh,
                                    download_btn_txt2shape_gif,
                                    download_btn_txt2shape_mesh,
                                    download_file_txt2shape,
                                    frame_size_txt2shape,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)

                if ram_size() >= 16 :
                    titletab_img2shape = f"{biniou_lang_tab_img2shape} 🧊"
                else :
                    titletab_img2shape = f"{biniou_lang_tab_img2shape} ⛔"
# img2shape
                with gr.TabItem(titletab_img2shape, id=52) as tab_img2shape:
                    with gr.Accordion(f"{biniou_lang_about}", open=False):
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_infos}</h1>
                                <b>{biniou_lang_about_module}</b>{biniou_lang_tab_img2shape}</br>
                                <b>{biniou_lang_about_function}</b>{biniou_lang_tab_img2shape_about_desc}<a href='https://github.com/openai/shap-e' target='_blank'>Shap-E</a></br>
                                <b>{biniou_lang_about_inputs}</b>{biniou_lang_about_input_image}</br>
                                <b>{biniou_lang_about_outputs}</b>{biniou_lang_tab_3d_about_output_text}</br>
                                <b>{biniou_lang_about_modelpage}</b>
                                {autodoc(model_list_img2shape)}<br />
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                f"""
                                <h1 style='text-align: left;'>{biniou_lang_about_help}</h1>
                                <div style='text-align: justified'>
                                <b>{biniou_lang_about_usage}</b></br>
                                {biniou_lang_tab_img2shape_about_instruct}
                                </br>
                                """
                            )
                    with gr.Accordion(biniou_lang_settings, open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2shape = gr.Dropdown(choices=model_list_img2shape, value=model_list_img2shape[0], label=biniou_lang_model_label, info=biniou_lang_model_info)
                            with gr.Column():
                                num_inference_step_img2shape = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label=biniou_lang_steps_label, info=biniou_lang_steps_info)
                            with gr.Column():
                                sampler_img2shape = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[11], label=biniou_lang_sampler_label, info=biniou_lang_sampler_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2shape = gr.Slider(0.1, 50.0, step=0.1, value=3.0, label=biniou_lang_cfgscale_label, info=biniou_lang_cfgscale_info)
                            with gr.Column():
                                num_images_per_prompt_img2shape = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label=biniou_lang_batch_size_label, info=biniou_lang_batch_size_image_info, interactive=False)
                            with gr.Column():
                                num_prompt_img2shape = gr.Slider(1, 32, step=1, value=1, label=biniou_lang_batch_count_label, info=biniou_lang_batch_count_info)
                        with gr.Row():
                            with gr.Column():
                                frame_size_img2shape = gr.Slider(0, biniou_global_width_max_img_create, step=8, value=64, label=biniou_lang_3d_frame_size_label, info=biniou_lang_3d_frame_size_info)
                            with gr.Column():
                                seed_img2shape = gr.Slider(0, 10000000000, step=1, value=0, label=biniou_lang_seed_label, info=biniou_lang_seed_info, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2shape = gr.Button(f"{biniou_lang_save_settings} 💾")
                            with gr.Column():
                                module_name_img2shape = gr.Textbox(value="img2shape", visible=False, interactive=False)
                                del_ini_btn_img2shape = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_ini_exist(module_name_img2shape.value) else False)
                                save_ini_btn_img2shape.click(
                                    fn=write_ini_img2shape,
                                    inputs=[
                                        module_name_img2shape,
                                        model_img2shape,
                                        num_inference_step_img2shape,
                                        sampler_img2shape,
                                        guidance_scale_img2shape,
                                        num_images_per_prompt_img2shape,
                                        num_prompt_img2shape,
                                        frame_size_img2shape,
                                        seed_img2shape,
                                        ]
                                    )
                                save_ini_btn_img2shape.click(fn=lambda: gr.Info(biniou_lang_save_settings_msg))
                                save_ini_btn_img2shape.click(fn=lambda: del_ini_btn_img2shape.update(interactive=True), outputs=del_ini_btn_img2shape)
                                del_ini_btn_img2shape.click(fn=lambda: del_ini(module_name_img2shape.value))
                                del_ini_btn_img2shape.click(fn=lambda: gr.Info(biniou_lang_delete_settings_msg))
                                del_ini_btn_img2shape.click(fn=lambda: del_ini_btn_img2shape.update(interactive=False), outputs=del_ini_btn_img2shape)
                        if test_ini_exist(module_name_img2shape.value):
                            with open(f".ini/{module_name_img2shape.value}.ini", "r", encoding="utf-8") as fichier:
                                exec(fichier.read())
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    img_img2shape = gr.Image(label=biniou_lang_img_input_label, height=320, type="pil")
                            with gr.Row():
                                with gr.Column():
                                    output_type_img2shape = gr.Radio(choices=["gif", "mesh"], value="gif", label=biniou_lang_output_type_label, info=biniou_lang_output_type_info)
                        with gr.Column(scale=2):
                            out_img2shape = gr.Gallery(
                                label=biniou_lang_image_gallery_label,
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=True
                            )    
                            out_size_img2shape = gr.Number(value=64, visible=False)
                            mesh_out_img2shape = gr.Model3D(
                                label=biniou_lang_3d_generated_label,
#                                clear_color=[255.0, 255.0, 255.0, 255.0],
                                height=400,
                                zoom_speed=5,
                                visible=False,
#                                interactive=False,
                            )    
                            mesh_out_size_img2shape = gr.Number(value=512, visible=False)
                            bool_output_type_img2shape = gr.Checkbox(value=True, visible=False, interactive=False)
                            gs_out_img2shape = gr.State()
                            sel_out_img2shape = gr.Number(precision=0, visible=False)
                            out_img2shape.select(get_select_index, None, sel_out_img2shape)
                            gs_mesh_out_img2shape = gr.Textbox(visible=False)
                            with gr.Row():
                                with gr.Column():
                                    download_btn_img2shape_gif = gr.Button(f"{biniou_lang_image_zip} 💾", visible=True)
                                    download_btn_img2shape_mesh = gr.Button("Zip model 💾", visible=False)
                                with gr.Column():
                                    download_file_img2shape = gr.File(label=biniou_lang_image_zip_file, height=30, interactive=False, visible=False)
                                    download_btn_img2shape_gif.click(fn=zip_download_file_img2shape, inputs=[out_img2shape], outputs=[download_file_img2shape, download_file_img2shape])
                                    download_btn_img2shape_mesh.click(fn=zip_mesh_img2shape, inputs=[gs_mesh_out_img2shape], outputs=[download_file_img2shape, download_file_img2shape])
                    with gr.Row():
                        with gr.Column():
                            btn_img2shape_gif = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=True)
                            btn_img2shape_mesh = gr.Button(f"{biniou_lang_generate} 🚀", variant="primary", visible=False)
                        with gr.Column():
                            btn_img2shape_clear_input = gr.ClearButton(components=[img_img2shape], value=f"{biniou_lang_clear_inputs} 🧹")
                        with gr.Column():
                            btn_img2shape_clear_output = gr.ClearButton(components=[out_img2shape, gs_out_img2shape, mesh_out_img2shape, gs_mesh_out_img2shape], value=f"{biniou_lang_clear_outputs} 🧹")
                            btn_img2shape_gif.click(fn=hide_download_file_img2shape, inputs=None, outputs=download_file_img2shape)
                            btn_img2shape_gif.click(
                            fn=image_img2shape,
                            inputs=[
                                model_img2shape,
                                sampler_img2shape,
                                img_img2shape,
                                num_images_per_prompt_img2shape,
                                num_prompt_img2shape,
                                guidance_scale_img2shape,
                                num_inference_step_img2shape,
                                frame_size_img2shape,
                                seed_img2shape,
                                output_type_img2shape,
                                nsfw_filter,
                                ],
                                outputs=[out_img2shape, gs_out_img2shape],
                                show_progress="full",
                            )
                            btn_img2shape_mesh.click(fn=hide_download_file_img2shape, inputs=None, outputs=download_file_img2shape)
                            btn_img2shape_mesh.click(
                            fn=image_img2shape,
                            inputs=[
                                model_img2shape,
                                sampler_img2shape,
                                img_img2shape,
                                num_images_per_prompt_img2shape,
                                num_prompt_img2shape,
                                guidance_scale_img2shape,
                                num_inference_step_img2shape,
                                frame_size_img2shape,
                                seed_img2shape,
                                output_type_img2shape,
                                nsfw_filter,
                            ],
                            outputs=[mesh_out_img2shape, gs_mesh_out_img2shape],
                            show_progress="full",
                            )
                            output_type_img2shape.change(
                                fn=change_output_type_img2shape,
                                inputs=[
                                    output_type_img2shape,
                                    out_size_img2shape,
                                    mesh_out_size_img2shape
                                ],
                                outputs=[
                                    out_img2shape,
                                    mesh_out_img2shape,
                                    bool_output_type_img2shape,
                                    btn_img2shape_gif,
                                    btn_img2shape_mesh,
                                    download_btn_img2shape_gif,
                                    download_btn_img2shape_mesh,
                                    download_file_img2shape,
                                    frame_size_img2shape,
                                    ]
                            )
                    with gr.Accordion(biniou_lang_send_label, open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_sel_output_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_input_prompt_value)
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value=biniou_lang_send_both_value)
# Global settings
        with gr.TabItem(f"{biniou_lang_tab_settings} ⚙️", id=6) as tab_settings:
            with gr.Tabs() as tabs_settings:
# Authentication
                with gr.TabItem(f"{biniou_lang_tab_login} 🔐", id=60) as tab_login:
                    with gr.Row():
                         with gr.Accordion(biniou_lang_tab_login_acc, open=True):
                             with gr.Row():
                                 with gr.Column():
                                     biniou_login_user = gr.Textbox(value="", lines=1, max_lines=1, label=biniou_lang_tab_login_user_label, info=biniou_lang_tab_login_user_info)
                                 with gr.Column():
                                     biniou_login_pass = gr.Textbox(value="", lines=1, max_lines=1, type="password", label=biniou_lang_tab_login_pass_label, info=biniou_lang_tab_login_pass_info)
                             with gr.Row():
                                 with gr.Column():
                                     btn_biniou_login = gr.Button(f"{biniou_lang_tab_login_btn_login} 🔑")
                                 with gr.Column():
                                     btn_biniou_logout = gr.Button(f"{biniou_lang_tab_login_btn_logout} 🔓")
                                 with gr.Column():
                                     btn_biniou_login_clear_input = gr.ClearButton(components=[biniou_login_user, biniou_login_pass], value=f"{biniou_lang_clear_inputs} 🧹")
                                     biniou_login_test = gr.Textbox(value="", visible=False)
# UI settings
                with gr.TabItem(f"{biniou_lang_tab_webui} 🧠", id=61) as tab_webui:
                    with gr.Accordion(biniou_lang_tab_webui, open=True, visible=False) as acc_webui:
                        with gr.Row():
                             with gr.Accordion(biniou_lang_tab_webui_system, open=True):
                                 with gr.Row():
                                     with gr.Column():
                                         btn_restart_ui_settings = gr.Button(f"{biniou_lang_tab_webui_restart} ↩️")
                                         btn_restart_ui_settings.click(fn=biniouUIControl.restart_program)
                                     with gr.Column():
                                         btn_reload_ui_settings = gr.Button(f"{biniou_lang_tab_webui_reload} ♻️")
#                                         btn_reload_ui_settings.click(fn=biniouUIControl.reload_ui, _js="window.location.reload()")
                                         btn_reload_ui_settings.click(fn=biniouUIControl.reload_ui, _js="location.replace(location.href)")
                                     with gr.Column():
                                         btn_close_ui_settings = gr.Button(f"{biniou_lang_tab_webui_shutdown} 🛑")
                                         btn_close_ui_settings.click(fn=biniouUIControl.close_program)
                                     with gr.Column():
                                         gr.Number(visible=False)
                        with gr.Row():
                             with gr.Accordion(biniou_lang_tab_webui_update_title, open=True):
                                 with gr.Row():
                                     with gr.Column():
                                         optimizer_update_ui = gr.Radio(choices=["cpu", "cuda", "rocm"], value=biniouUIControl.detect_optimizer(), label=biniou_lang_tab_webui_update_label, info=biniou_lang_tab_webui_update_info)
                                 with gr.Row():
                                     with gr.Column():
                                         btn_update_ui = gr.Button(f"{biniou_lang_tab_webui_update_btn_label} ⤵️", variant="primary")
                                         btn_update_ui.click(fn=biniouUIControl.biniou_update, inputs=optimizer_update_ui, outputs=optimizer_update_ui)
                                     with gr.Column():
                                         gr.Number(visible=False)
                                     with gr.Column():
                                         gr.Number(visible=False)
                                     with gr.Column():
                                         gr.Number(visible=False)
                        with gr.Row():
                            with gr.Accordion(biniou_lang_tab_webui_backend_title, open=True):
                                with gr.Row():
                                    with gr.Column():
                                        llama_backend_ui = gr.Radio(choices=["none", "openblas", "cuda", "metal", "opencl/clblast", "rocm/hipblas", "vulkan", "kompute"], value=biniouUIControl.detect_llama_backend(), label=biniou_lang_tab_webui_backend_label, info=biniou_lang_tab_webui_backend_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_llama_backend_ui = gr.Button(f"{biniou_lang_tab_webui_backend_btn_label} ⤵️", variant="primary")
                                        btn_llama_backend_ui.click(fn=biniouUIControl.biniou_llama_backend, inputs=llama_backend_ui, outputs=llama_backend_ui)
                                    with gr.Column():
                                        gr.Number(visible=False)
                                    with gr.Column():
                                        gr.Number(visible=False)
                                    with gr.Column():
                                        gr.Number(visible=False)
                        with gr.Row():
                            with gr.Accordion(biniou_lang_tab_webui_settings_title, open=True):
                                with gr.Accordion(biniou_lang_tab_webui_settings_lang_title, open=True):
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_lang_ui = gr.Dropdown(choices=biniouUIControl.biniou_languages_list(), value=biniou_global_lang_ui, label=biniou_lang_tab_webui_settings_lang_label, info=biniou_lang_tab_webui_settings_lang_info, interactive=True)
                                with gr.Accordion(biniou_lang_tab_webui_settings_backend_title, open=True):
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_server_name = gr.Checkbox(value=biniou_global_server_name, label=biniou_lang_tab_webui_settings_server_name_label, info=biniou_lang_tab_webui_settings_server_name_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_server_port = gr.Slider(0, 65535, step=1, precision=0, value=biniou_global_server_port, label=biniou_lang_tab_webui_settings_server_port_label, info=biniou_lang_tab_webui_settings_server_port_info)
                                        with gr.Column():
                                            biniou_global_settings_inbrowser = gr.Checkbox(value=biniou_global_inbrowser, label=biniou_lang_tab_webui_settings_inbrowser_label, info=biniou_lang_tab_webui_settings_inbrowser_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_auth = gr.Checkbox(value=biniou_global_auth, label=biniou_lang_tab_webui_settings_auth_label, info=biniou_lang_tab_webui_settings_auth_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_auth_message = gr.Textbox(value=biniou_global_auth_message, lines=1, max_lines=3, label=biniou_lang_tab_webui_settings_auth_msg_label, info=biniou_lang_tab_webui_settings_auth_msg_info, interactive=True if biniou_global_auth else False)
                                        with gr.Column():
                                            biniou_global_settings_share = gr.Checkbox(value=biniou_global_share, label=biniou_lang_tab_webui_settings_share_label, info=f"⚠️ {biniou_lang_tab_webui_settings_share_info}⚠️", interactive=True if biniou_global_auth else False)
                                            biniou_global_settings_auth.change(biniou_global_settings_auth_switch, biniou_global_settings_auth, [biniou_global_settings_auth_message, biniou_global_settings_share])
                                with gr.Accordion(biniou_lang_tab_webui_settings_image, open=True):
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_steps_max = gr.Slider(0, 512, step=1, value=biniou_global_steps_max, label=biniou_lang_tab_webui_settings_steps_max_label, info=biniou_lang_tab_webui_settings_steps_max_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_batch_size_max = gr.Slider(1, 512, step=1, value=biniou_global_batch_size_max, label=biniou_lang_tab_webui_settings_batch_size_label, info=biniou_lang_tab_webui_settings_batch_size_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_width_max_img_create = gr.Slider(128, 16384, step=64, value=biniou_global_width_max_img_create, label=biniou_lang_tab_webui_settings_max_w_crea_label, info=biniou_lang_tab_webui_settings_max_w_crea_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_height_max_img_create = gr.Slider(128, 16384, step=64, value=biniou_global_height_max_img_create, label=biniou_lang_tab_webui_settings_max_h_crea_label, info=biniou_lang_tab_webui_settings_max_h_crea_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_width_max_img_modify = gr.Slider(128, 16384, step=64, value=biniou_global_width_max_img_modify, label=biniou_lang_tab_webui_settings_max_w_mod_label, info=biniou_lang_tab_webui_settings_max_w_mod_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_height_max_img_modify = gr.Slider(128, 16384, step=64, value=biniou_global_height_max_img_modify, label=biniou_lang_tab_webui_settings_max_h_mod_label, info=biniou_lang_tab_webui_settings_max_h_mod_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_sd15_width = gr.Slider(128, 16384, step=64, value=biniou_global_sd15_width, label=biniou_lang_tab_webui_settings_def_w_sd15_label, info=biniou_lang_tab_webui_settings_def_w_sd15_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_sd15_height = gr.Slider(128, 16384, step=64, value=biniou_global_sd15_height, label=biniou_lang_tab_webui_settings_def_h_sd15_label, info=biniou_lang_tab_webui_settings_def_h_sd15_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_sdxl_width = gr.Slider(128, 16384, step=64, value=biniou_global_sdxl_width, label=biniou_lang_tab_webui_settings_def_w_sdxl_label, info=biniou_lang_tab_webui_settings_def_w_sdxl_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_sdxl_height = gr.Slider(128, 16384, step=64, value=biniou_global_sdxl_height, label=biniou_lang_tab_webui_settings_def_h_sdxl_label, info=biniou_lang_tab_webui_settings_def_h_sdxl_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_gfpgan = gr.Checkbox(value=biniou_global_gfpgan, label=biniou_lang_tab_webui_settings_gfpgan_label, info=biniou_lang_tab_webui_settings_gfpgan_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_tkme = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label=biniou_lang_tab_webui_settings_tkme_label, info=biniou_lang_tab_webui_settings_tkme_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_clipskip = gr.Slider(0, 12, step=1, value=biniou_global_clipskip, label=biniou_lang_tab_webui_settings_clipskip_label, info=biniou_lang_tab_webui_settings_clipskip_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_ays = gr.Checkbox(value=biniou_global_ays, label=biniou_lang_tab_webui_settings_ays_label, info=biniou_lang_tab_webui_settings_ays_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_img_fmt = gr.Dropdown(choices=img_fmt_list(), value=biniou_global_img_fmt, label=biniou_lang_tab_webui_settings_img_fmt_label, info=biniou_lang_tab_webui_settings_img_fmt_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_text_metadatas = gr.Checkbox(value=biniou_global_text_metadatas, label=biniou_lang_tab_webui_settings_text_metadatas_label, info=biniou_lang_tab_webui_settings_text_metadatas_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_img_exif = gr.Checkbox(value=biniou_global_img_exif, label=biniou_lang_tab_webui_settings_exif_label, info=biniou_lang_tab_webui_settings_exif_info, interactive=True)
                                    with gr.Row():
                                        with gr.Column():
                                            biniou_global_settings_gif_exif = gr.Checkbox(value=biniou_global_gif_exif, label=biniou_lang_tab_webui_settings_gif_exif_label, info=biniou_lang_tab_webui_settings_gif_exif_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_mp4_metadatas = gr.Checkbox(value=biniou_global_mp4_metadatas, label=biniou_lang_tab_webui_settings_mp4_metadatas_label, info=biniou_lang_tab_webui_settings_mp4_metadatas_info, interactive=True)
                                        with gr.Column():
                                            biniou_global_settings_audio_metadatas = gr.Checkbox(value=biniou_global_audio_metadatas, label=biniou_lang_tab_webui_settings_audio_metadatas_label, info=biniou_lang_tab_webui_settings_audio_metadatas_info, interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        save_ini_btn_settings = gr.Button(f"{biniou_lang_save_settings} 💾")
                                    with gr.Column():
                                        module_name_settings = gr.Textbox(value="settings", visible=False, interactive=False)
                                        del_ini_btn_settings = gr.Button(f"{biniou_lang_delete_settings} 🗑️", interactive=True if test_cfg_exist(module_name_settings.value) else False)
                                        save_ini_btn_settings.click(fn=write_settings_ini, 
                                            inputs=[
                                                module_name_settings,
                                                biniou_global_settings_lang_ui,
                                                biniou_global_settings_server_name,
                                                biniou_global_settings_server_port,
                                                biniou_global_settings_inbrowser,
                                                biniou_global_settings_auth,
                                                biniou_global_settings_auth_message,
                                                biniou_global_settings_share,
                                                biniou_global_settings_steps_max,
                                                biniou_global_settings_batch_size_max,
                                                biniou_global_settings_width_max_img_create,
                                                biniou_global_settings_height_max_img_create,
                                                biniou_global_settings_width_max_img_modify,
                                                biniou_global_settings_height_max_img_modify,
                                                biniou_global_settings_sd15_width,
                                                biniou_global_settings_sd15_height,
                                                biniou_global_settings_sdxl_width,
                                                biniou_global_settings_sdxl_height,
                                                biniou_global_settings_gfpgan,
                                                biniou_global_settings_tkme,
                                                biniou_global_settings_clipskip,
                                                biniou_global_settings_ays,
                                                biniou_global_settings_img_fmt,
                                                biniou_global_settings_text_metadatas,
                                                biniou_global_settings_img_exif,
                                                biniou_global_settings_gif_exif,
                                                biniou_global_settings_mp4_metadatas,
                                                biniou_global_settings_audio_metadatas,
                                            ],
                                            outputs=None
                                        )
                                        save_ini_btn_settings.click(fn=lambda: gr.Info(biniou_lang_tab_webui_settings_saved))
                                        save_ini_btn_settings.click(fn=lambda: del_ini_btn_settings.update(interactive=True), outputs=del_ini_btn_settings)
                                        del_ini_btn_settings.click(fn=lambda: del_cfg(module_name_settings.value))
                                        del_ini_btn_settings.click(fn=lambda: gr.Info(biniou_lang_tab_webui_settings_deleted))
                                        del_ini_btn_settings.click(fn=lambda: del_ini_btn_settings.update(interactive=False), outputs=del_ini_btn_settings)
                        with gr.Row():
                            with gr.Accordion(biniou_lang_tab_webui_nsfw_title, open=False):
                                with gr.Row():
                                    with gr.Column():
                                        safety_checker_ui_settings = gr.Checkbox(bool(int(nsfw_filter.value)), label=biniou_lang_tab_webui_nsfw_label, info=f"⚠️ {biniou_lang_tab_webui_nsfw_info} ⚠️", interactive=True)
                                        safety_checker_ui_settings.change(fn=lambda x:int(x), inputs=safety_checker_ui_settings, outputs=nsfw_filter)
# Models cleaner
                with gr.TabItem(f"{biniou_lang_tab_cleaner} 🧹", id=62) as tab_models_cleaner:
                    with gr.Accordion(biniou_lang_tab_cleaner, open=True, visible=False) as acc_models_cleaner:
                        with gr.Row():
                            list_models_cleaner = gr.CheckboxGroup(choices=biniouModelsManager("./models").modelslister(), type="value", label=biniou_lang_tab_settings_list_label, info=biniou_lang_tab_cleaner_list_info)
                        with gr.Row():
                            with gr.Column():
                                btn_models_cleaner = gr.Button(f"{biniou_lang_tab_settings_delete_models} 🧹", variant="primary")
                                btn_models_cleaner.click(fn=biniouModelsManager("./models").modelsdeleter, inputs=[list_models_cleaner])
                                btn_models_cleaner.click(fn=refresh_models_cleaner_list, outputs=list_models_cleaner)
                            with gr.Column():
                                btn_models_cleaner_refresh = gr.Button(f"{biniou_lang_tab_settings_refresh_models} ♻️")
                                btn_models_cleaner_refresh.click(fn=refresh_models_cleaner_list, outputs=list_models_cleaner)
                            with gr.Column():
                                gr.Number(visible=False)
                            with gr.Column():
                                gr.Number(visible=False)
# LoRA Models manager
                with gr.TabItem(f"{biniou_lang_tab_lora_models} 🛠️", id=63) as tab_lora_models_manager:
                    with gr.Accordion(biniou_lang_tab_lora_models, open=True, visible=False) as acc_lora_models_manager:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(f"""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>{biniou_lang_tab_lora_sd15_models}</span>""")
                                with gr.Row():
                                    list_lora_models_manager_sd = gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SD").modelslister(), type="value", label=biniou_lang_tab_settings_list_label, info=biniou_lang_tab_lora_models_list_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_lora_models_manager_sd = gr.Button(f"{biniou_lang_tab_settings_delete_models} 🧹", variant="primary")
                                        btn_lora_models_manager_sd.click(fn=biniouLoraModelsManager("./models/lora/SD").modelsdeleter, inputs=[list_lora_models_manager_sd])
                                        btn_lora_models_manager_sd.click(fn=refresh_lora_models_manager_list_sd, outputs=list_lora_models_manager_sd)
                                    with gr.Column():
                                        btn_lora_models_manager_refresh_sd = gr.Button(f"{biniou_lang_tab_settings_refresh_models} ♻️")
                                        btn_lora_models_manager_refresh_sd.click(fn=refresh_lora_models_manager_list_sd, outputs=list_lora_models_manager_sd)
                                with gr.Row():
                                    with gr.Column():
                                        url_lora_models_manager_sd = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label=biniou_lang_tab_lora_models_url_label, info=biniou_lang_tab_lora_models_url_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_url_lora_models_manager_sd = gr.Button(f"{biniou_lang_tab_lora_models_down} 💾", variant="primary")
                                        btn_url_lora_models_manager_sd.click(biniouLoraModelsManager("./models/lora/SD").modelsdownloader, inputs=url_lora_models_manager_sd, outputs=url_lora_models_manager_sd)
                                    with gr.Column():
                                            gr.Number(visible=False)
                            with gr.Column():
                                gr.HTML(f"""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>{biniou_lang_tab_lora_sdxl_models}</span>""")
                                with gr.Row():
                                    list_lora_models_manager_sdxl = gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SDXL").modelslister(), type="value", label=biniou_lang_tab_settings_list_label, info=biniou_lang_tab_lora_models_list_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_lora_models_manager_sdxl = gr.Button(f"{biniou_lang_tab_settings_delete_models} 🧹", variant="primary")
                                        btn_lora_models_manager_sdxl.click(fn=biniouLoraModelsManager("./models/lora/SDXL").modelsdeleter, inputs=[list_lora_models_manager_sdxl])
                                        btn_lora_models_manager_sdxl.click(fn=refresh_lora_models_manager_list_sdxl, outputs=list_lora_models_manager_sdxl)
                                    with gr.Column():
                                        btn_lora_models_manager_refresh_sdxl = gr.Button(f"{biniou_lang_tab_settings_refresh_models} ♻️")
                                        btn_lora_models_manager_refresh_sdxl.click(fn=refresh_lora_models_manager_list_sdxl, outputs=list_lora_models_manager_sdxl)
                                with gr.Row():
                                    with gr.Column():
                                        url_lora_models_manager_sdxl = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label=biniou_lang_tab_lora_models_url_label, info=biniou_lang_tab_lora_models_url_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_url_lora_models_manager_sdxl = gr.Button(f"{biniou_lang_tab_lora_models_down} 💾", variant="primary")
                                        btn_url_lora_models_manager_sdxl.click(biniouLoraModelsManager("./models/lora/SDXL").modelsdownloader, inputs=url_lora_models_manager_sdxl, outputs=url_lora_models_manager_sdxl)
                                    with gr.Column():
                                        gr.Number(visible=False)

# Textual inversion Models manager
                with gr.TabItem(f"{biniou_lang_tab_textinv} 🛠️", id=64) as tab_textinv_manager:
                    with gr.Accordion(biniou_lang_tab_textinv, open=True, visible=False) as acc_textinv_manager:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(f"""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>{biniou_lang_tab_textinv_sd15_models}</span>""")
                                with gr.Row():
                                    list_textinv_manager_sd = gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SD").modelslister(), type="value", label=biniou_lang_tab_textinv_models_label, info=biniou_lang_tab_textinv_models_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_textinv_manager_sd = gr.Button(f"{biniou_lang_tab_textinv_delete_models} 🧹", variant="primary")
                                        btn_textinv_manager_sd.click(fn=biniouTextinvModelsManager("./models/TextualInversion/SD").modelsdeleter, inputs=[list_textinv_manager_sd])
                                        btn_textinv_manager_sd.click(fn=refresh_textinv_manager_list_sd, outputs=list_textinv_manager_sd)
                                    with gr.Column():
                                        btn_textinv_manager_refresh_sd = gr.Button(f"{biniou_lang_tab_textinv_refresh_models} ♻️")
                                        btn_textinv_manager_refresh_sd.click(fn=refresh_textinv_manager_list_sd, outputs=list_textinv_manager_sd)
                                with gr.Row():
                                    with gr.Column():
                                        url_textinv_manager_sd = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label=biniou_lang_tab_textinv_url_label, info=biniou_lang_tab_textinv_url_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_url_textinv_manager_sd = gr.Button(f"{biniou_lang_tab_textinv_down} 💾", variant="primary")
                                        btn_url_textinv_manager_sd.click(biniouTextinvModelsManager("./models/TextualInversion/SD").modelsdownloader, inputs=url_textinv_manager_sd, outputs=url_textinv_manager_sd)
                                    with gr.Column():
                                            gr.Number(visible=False)
                            with gr.Column():
                                gr.HTML(f"""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>{biniou_lang_tab_textinv_sdxl_models}</span>""")
                                with gr.Row():
                                    list_textinv_manager_sdxl = gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelslister(), type="value", label=biniou_lang_tab_textinv_models_label, info=biniou_lang_tab_textinv_models_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_textinv_manager_sdxl = gr.Button(f"{biniou_lang_tab_textinv_delete_models} 🧹", variant="primary")
                                        btn_textinv_manager_sdxl.click(fn=biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelsdeleter, inputs=[list_textinv_manager_sdxl])
                                        btn_textinv_manager_sdxl.click(fn=refresh_textinv_manager_list_sdxl, outputs=list_textinv_manager_sdxl)
                                    with gr.Column():
                                        btn_textinv_manager_refresh_sdxl = gr.Button(f"{biniou_lang_tab_textinv_refresh_models} ♻️")
                                        btn_textinv_manager_refresh_sdxl.click(fn=refresh_textinv_manager_list_sdxl, outputs=list_textinv_manager_sdxl)
                                with gr.Row():
                                    with gr.Column():
                                        url_textinv_manager_sdxl = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label=biniou_lang_tab_textinv_url_label, info=biniou_lang_tab_textinv_url_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_url_textinv_manager_sdxl = gr.Button(f"{biniou_lang_tab_textinv_down} 💾", variant="primary")
                                        btn_url_textinv_manager_sdxl.click(biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelsdownloader, inputs=url_textinv_manager_sdxl, outputs=url_textinv_manager_sdxl)
                                    with gr.Column():
                                        gr.Number(visible=False)

# SD Models downloader
                with gr.TabItem(f"{biniou_lang_tab_sd_models} 💾", id=65) as tab_sd_models_downloader:
                    with gr.Accordion(biniou_lang_tab_sd_models, open=True, visible=False) as acc_sd_models_downloader:
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        url_sd_models_downloader = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label=biniou_lang_tab_sd_models_url_label, info=biniou_lang_tab_sd_models_url_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_url_sd_models_downloader = gr.Button(f"{biniou_lang_tab_sd_models_down} 💾", variant="primary")
                                        btn_url_sd_models_downloader.click(biniouSDModelsDownloader("./models/Stable_Diffusion").modelsdownloader, inputs=url_sd_models_downloader, outputs=url_sd_models_downloader)
                                    with gr.Column():
                                            gr.Number(visible=False)
                                    with gr.Column():
                                            gr.Number(visible=False)
                                    with gr.Column():
                                            gr.Number(visible=False)

# GGUF Models downloader
                with gr.TabItem(f"{biniou_lang_tab_gguf_models} 💾", id=66) as tab_gguf_models_downloader:
                    with gr.Accordion(biniou_lang_tab_gguf_models, open=True, visible=False) as acc_gguf_models_downloader:
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        url_gguf_models_downloader = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label=biniou_lang_tab_gguf_models_url_label, info=biniou_lang_tab_gguf_models_url_info)
                                with gr.Row():
                                    with gr.Column():
                                        btn_url_gguf_models_downloader = gr.Button(f"{biniou_lang_tab_gguf_models_down} 💾", variant="primary")
                                        btn_url_gguf_models_downloader.click(biniouSDModelsDownloader("./models/llamacpp").modelsdownloader, inputs=url_gguf_models_downloader, outputs=url_gguf_models_downloader)
                                    with gr.Column():
                                            gr.Number(visible=False)
                                    with gr.Column():
                                            gr.Number(visible=False)
                                    with gr.Column():
                                            gr.Number(visible=False)
                btn_biniou_login.click(
                    fn=biniou_settings_login,
                    inputs=[
                        biniou_login_user,
                        biniou_login_pass
                    ],
                    outputs=[
                        acc_webui,
                        acc_models_cleaner,
                        acc_lora_models_manager,
                        acc_textinv_manager,
                        acc_sd_models_downloader,
                        acc_gguf_models_downloader,
                        biniou_login_user,
                        biniou_login_pass,
                        biniou_login_test,
                    ]
                )
                btn_biniou_logout.click(
                    fn=biniou_settings_logout,
                    inputs=None,
                    outputs=[
                        acc_webui,
                        acc_models_cleaner,
                        acc_lora_models_manager,
                        acc_textinv_manager,
                        acc_sd_models_downloader,
                        acc_gguf_models_downloader,
                        biniou_login_user,
                        biniou_login_pass,
                    ]
                )
                btn_biniou_logout.click(fn=lambda: gr.Info(biniou_lang_tab_login_btn_logout_message))
                biniou_login_test.change(fn=biniou_settings_login_test, inputs=biniou_login_test)
                biniou_login_test.change(fn=biniou_settings_login_test_clean, outputs=biniou_login_test)

    tab_text_num = gr.Number(value=tab_text.id, precision=0, visible=False)
    tab_image_num = gr.Number(value=tab_image.id, precision=0, visible=False)
    tab_audio_num = gr.Number(value=tab_audio.id, precision=0, visible=False)
    tab_video_num = gr.Number(value=tab_video.id, precision=0, visible=False)
    tab_3d_num = gr.Number(value=tab_3d.id, precision=0, visible=False)

    tab_llamacpp_num = gr.Number(value=tab_llamacpp.id, precision=0, visible=False)
    tab_llava_num = gr.Number(value=tab_llava.id, precision=0, visible=False)
    tab_img2txt_git_num = gr.Number(value=tab_img2txt_git.id, precision=0, visible=False)
    tab_whisper_num = gr.Number(value=tab_whisper.id, precision=0, visible=False)
    tab_nllb_num = gr.Number(value=tab_nllb.id, precision=0, visible=False)
    tab_txt2prompt_num = gr.Number(value=tab_txt2prompt.id, precision=0, visible=False)
    tab_txt2img_sd_num = gr.Number(value=tab_txt2img_sd.id, precision=0, visible=False)
    tab_txt2img_kd_num = gr.Number(value=tab_txt2img_kd.id, precision=0, visible=False)
    tab_txt2img_lcm_num = gr.Number(value=tab_txt2img_lcm.id, precision=0, visible=False)
    tab_txt2img_mjm_num = gr.Number(value=tab_txt2img_mjm.id, precision=0, visible=False)
    tab_txt2img_paa_num = gr.Number(value=tab_txt2img_paa.id, precision=0, visible=False)
    tab_img2img_num = gr.Number(value=tab_img2img.id, precision=0, visible=False)
    tab_img2img_ip_num = gr.Number(value=tab_img2img_ip.id, precision=0, visible=False)
    tab_img2var_num = gr.Number(value=tab_img2var.id, precision=0, visible=False) 
    tab_pix2pix_num = gr.Number(value=tab_pix2pix.id, precision=0, visible=False) 
    tab_magicmix_num = gr.Number(value=tab_magicmix.id, precision=0, visible=False)
    tab_inpaint_num = gr.Number(value=tab_inpaint.id, precision=0, visible=False) 
    tab_paintbyex_num = gr.Number(value=tab_paintbyex.id, precision=0, visible=False)
    tab_outpaint_num = gr.Number(value=tab_outpaint.id, precision=0, visible=False) 
    tab_controlnet_num = gr.Number(value=tab_controlnet.id, precision=0, visible=False)
    tab_faceid_ip_num = gr.Number(value=tab_faceid_ip.id, precision=0, visible=False)
    tab_faceswap_num = gr.Number(value=tab_faceswap.id, precision=0, visible=False)
    tab_resrgan_num = gr.Number(value=tab_resrgan.id, precision=0, visible=False)
    tab_gfpgan_num = gr.Number(value=tab_gfpgan.id, precision=0, visible=False)
    tab_musicgen_num = gr.Number(value=tab_musicgen.id, precision=0, visible=False)
    tab_musicgen_mel_num = gr.Number(value=tab_musicgen_mel.id, precision=0, visible=False)
    tab_musicldm_num = gr.Number(value=tab_musicldm.id, precision=0, visible=False)
    tab_audiogen_num = gr.Number(value=tab_audiogen.id, precision=0, visible=False)
    tab_harmonai_num = gr.Number(value=tab_harmonai.id, precision=0, visible=False)
    tab_bark_num = gr.Number(value=tab_bark.id, precision=0, visible=False)
    tab_txt2vid_ms_num = gr.Number(value=tab_txt2vid_ms.id, precision=0, visible=False)
    tab_txt2vid_ze_num = gr.Number(value=tab_txt2vid_ze.id, precision=0, visible=False)
    tab_animatediff_lcm_num = gr.Number(value=tab_animatediff_lcm.id, precision=0, visible=False)
    tab_img2vid_num = gr.Number(value=tab_img2vid.id, precision=0, visible=False)
    tab_vid2vid_ze_num = gr.Number(value=tab_vid2vid_ze.id, precision=0, visible=False)
    tab_txt2shape_num = gr.Number(value=tab_txt2shape.id, precision=0, visible=False)
    tab_img2shape_num = gr.Number(value=tab_img2shape.id, precision=0, visible=False)

# Llamacpp outputs
    llamacpp_nllb.click(fn=send_text_to_module_text, inputs=[last_reply_llamacpp, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    llamacpp_txt2img_sd.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    llamacpp_txt2img_kd.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    llamacpp_txt2img_lcm.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    llamacpp_txt2img_mjm.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    llamacpp_txt2img_paa.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    llamacpp_img2img.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    llamacpp_img2img_ip.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    llamacpp_pix2pix.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    llamacpp_inpaint.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    llamacpp_controlnet.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    llamacpp_faceid_ip.click(fn=send_text_to_module_image, inputs=[last_reply_llamacpp, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    llamacpp_musicgen.click(fn=import_to_module_audio, inputs=[last_reply_llamacpp, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    llamacpp_audiogen.click(fn=import_to_module_audio, inputs=[last_reply_llamacpp, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    llamacpp_bark.click(fn=import_to_module_audio, inputs=[last_reply_llamacpp, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])
    llamacpp_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[last_reply_llamacpp, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    llamacpp_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[last_reply_llamacpp, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    llamacpp_animatediff_lcm.click(fn=import_text_to_module_video, inputs=[last_reply_llamacpp, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])

# llava outputs
    llava_nllb.click(fn=send_text_to_module_text, inputs=[last_reply_llava, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    llava_txt2img_sd.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    llava_txt2img_kd.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    llava_txt2img_lcm.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    llava_txt2img_mjm.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    llava_txt2img_paa.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    llava_img2img.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    llava_img2img_ip.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    llava_pix2pix.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    llava_inpaint.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    llava_controlnet.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    llava_faceid_ip.click(fn=send_text_to_module_image, inputs=[last_reply_llava, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    llava_musicgen.click(fn=import_to_module_audio, inputs=[last_reply_llava, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    llava_audiogen.click(fn=import_to_module_audio, inputs=[last_reply_llava, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    llava_bark.click(fn=import_to_module_audio, inputs=[last_reply_llava, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])
    llava_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[last_reply_llava, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    llava_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[last_reply_llava, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    llava_animatediff_lcm.click(fn=import_text_to_module_video, inputs=[last_reply_llava, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])

# GIT Captions outputs
    img2txt_git_nllb.click(fn=send_text_to_module_text, inputs=[out_img2txt_git, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    img2txt_git_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    img2txt_git_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    img2txt_git_txt2img_lcm.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    img2txt_git_txt2img_mjm.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    img2txt_git_txt2img_paa.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    img2txt_git_img2img.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    img2txt_git_img2img_ip.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    img2txt_git_pix2pix.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    img2txt_git_inpaint.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    img2txt_git_controlnet.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    img2txt_git_faceid_ip.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    img2txt_git_musicgen.click(fn=import_to_module_audio, inputs=[out_img2txt_git, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    img2txt_git_audiogen.click(fn=import_to_module_audio, inputs=[out_img2txt_git, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    img2txt_git_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_img2txt_git, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    img2txt_git_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_img2txt_git, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    img2txt_git_animatediff_lcm.click(fn=import_text_to_module_video, inputs=[out_img2txt_git, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])
# GIT Captions both
    img2txt_git_img2img_both.click(fn=both_text_to_module_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_img2img_num], outputs=[img_img2img, prompt_img2img, tabs, tabs_image])
    img2txt_git_img2img_ip_both.click(fn=both_text_to_module_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, prompt_img2img_ip, tabs, tabs_image])
    img2txt_git_pix2pix_both.click(fn=both_text_to_module_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, prompt_pix2pix, tabs, tabs_image])
    img2txt_git_inpaint_both.click(fn=both_text_to_module_inpaint_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, prompt_inpaint, tabs, tabs_image])
    img2txt_git_controlnet_both.click(fn=both_text_to_module_inpaint_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, prompt_controlnet, tabs, tabs_image])
    img2txt_git_faceid_ip_both.click(fn=both_text_to_module_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, prompt_faceid_ip, tabs, tabs_image])

# Whisper outputs
    whisper_nllb.click(fn=send_text_to_module_text, inputs=[out_whisper, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    whisper_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    whisper_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    whisper_txt2img_lcm.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    whisper_txt2img_mjm.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    whisper_txt2img_paa.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    whisper_img2img.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    whisper_img2img_ip.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    whisper_pix2pix.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    whisper_inpaint.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    whisper_controlnet.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    whisper_faceid_ip.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    whisper_musicgen.click(fn=import_to_module_audio, inputs=[out_whisper, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    whisper_audiogen.click(fn=import_to_module_audio, inputs=[out_whisper, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    whisper_bark.click(fn=import_to_module_audio, inputs=[out_whisper, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])
    whisper_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_whisper, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    whisper_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_whisper, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    whisper_animatediff_lcm.click(fn=import_text_to_module_video, inputs=[out_whisper, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])

# Nllb outputs
    nllb_llamacpp.click(fn=send_text_to_module_text, inputs=[out_nllb, tab_text_num, tab_llamacpp_num], outputs=[prompt_llamacpp, tabs, tabs_text])
    nllb_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    nllb_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    nllb_txt2img_lcm.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    nllb_txt2img_mjm.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    nllb_txt2img_paa.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    nllb_img2img.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    nllb_img2img_ip.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    nllb_pix2pix.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    nllb_inpaint.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    nllb_controlnet.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    nllb_faceid_ip.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    nllb_musicgen.click(fn=import_to_module_audio, inputs=[out_nllb, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    nllb_audiogen.click(fn=import_to_module_audio, inputs=[out_nllb, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    nllb_bark.click(fn=import_to_module_audio, inputs=[out_nllb, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])
    nllb_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_nllb, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    nllb_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_nllb, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    nllb_animatediff_lcm.click(fn=import_text_to_module_video, inputs=[out_nllb, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])

# txt2prompt outputs
    txt2prompt_nllb.click(fn=send_text_to_module_text, inputs=[out_txt2prompt, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    txt2prompt_llamacpp.click(fn=send_text_to_module_text, inputs=[out_txt2prompt, tab_text_num, tab_llamacpp_num], outputs=[prompt_llamacpp, tabs, tabs_text])
    txt2prompt_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    txt2prompt_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    txt2prompt_txt2img_lcm.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2prompt_txt2img_mjm.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    txt2prompt_txt2img_paa.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    txt2prompt_img2img.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    txt2prompt_img2img_ip.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    txt2prompt_pix2pix.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    txt2prompt_inpaint.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    txt2prompt_controlnet.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    txt2prompt_faceid_ip.click(fn=send_text_to_module_image, inputs=[out_txt2prompt, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    txt2prompt_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_txt2prompt, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    txt2prompt_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_txt2prompt, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    txt2prompt_animatediff_lcm.click(fn=import_text_to_module_video, inputs=[out_txt2prompt, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])
      
# txt2img_sd outputs
    txt2img_sd_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_sd_img2img_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    txt2img_sd_img2var.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    txt2img_sd_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_sd_magicmix.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    txt2img_sd_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_sd_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    txt2img_sd_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    txt2img_sd_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_sd_faceid_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    txt2img_sd_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    txt2img_sd_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_sd_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_sd_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    txt2img_sd_llava.click(fn=send_to_module_text, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    txt2img_sd_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    txt2img_sd_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# txt2img_sd inputs
    txt2img_sd_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    txt2img_sd_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_sd, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2img_sd_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    txt2img_sd_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    txt2img_sd_img2img_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    txt2img_sd_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    txt2img_sd_pix2pix_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    txt2img_sd_inpaint_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    txt2img_sd_controlnet_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    txt2img_sd_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
    txt2img_sd_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_sd_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    txt2img_sd_animatediff_lcm_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tabs, tabs_video])
# txt2img_sd both
    txt2img_sd_img2img_both.click(fn=both_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_sd_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    txt2img_sd_pix2pix_both.click(fn=both_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_sd_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_sd_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_sd_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# txt2img_kd outputs
    txt2img_kd_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_kd_img2img_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    txt2img_kd_img2var.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    txt2img_kd_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_kd_magicmix.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    txt2img_kd_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_kd_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    txt2img_kd_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    txt2img_kd_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_kd_faceid_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    txt2img_kd_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    txt2img_kd_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_kd_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_kd_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    txt2img_kd_llava.click(fn=send_to_module_text, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    txt2img_kd_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    txt2img_kd_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])
    
# txt2img_kd inputs
    txt2img_kd_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2img_kd_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_kd, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2img_kd_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    txt2img_kd_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    txt2img_kd_img2img_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    txt2img_kd_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    txt2img_kd_pix2pix_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    txt2img_kd_inpaint_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    txt2img_kd_controlnet_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    txt2img_kd_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
    txt2img_kd_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_kd_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    txt2img_kd_animatediff_lcm_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tabs, tabs_video])
# txt2img_kd both
    txt2img_kd_img2img_both.click(fn=both_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_kd_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    txt2img_kd_pix2pix_both.click(fn=both_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_kd_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_kd_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_kd_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# txt2img_lcm outputs
    txt2img_lcm_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_lcm_img2img_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    txt2img_lcm_img2var.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    txt2img_lcm_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_lcm_magicmix.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    txt2img_lcm_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_lcm_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    txt2img_lcm_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    txt2img_lcm_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_lcm_faceid_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    txt2img_lcm_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    txt2img_lcm_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_lcm_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_lcm_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    txt2img_lcm_llava.click(fn=send_to_module_text, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    txt2img_lcm_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    txt2img_lcm_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# txt2img_lcm inputs
    txt2img_lcm_txt2img_sd_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    txt2img_lcm_txt2img_kd_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image]) 
    txt2img_lcm_txt2img_mjm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    txt2img_lcm_txt2img_paa_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    txt2img_lcm_img2img_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    txt2img_lcm_img2img_ip_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, tabs, tabs_image])
    txt2img_lcm_pix2pix_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    txt2img_lcm_inpaint_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    txt2img_lcm_controlnet_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])
    txt2img_lcm_faceid_ip_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_lcm, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, tabs, tabs_image])
    txt2img_lcm_txt2vid_ms_input.click(fn=import_to_module_video_prompt_only, inputs=[prompt_txt2img_lcm, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_lcm_txt2vid_ze_input.click(fn=import_to_module_video_prompt_only, inputs=[prompt_txt2img_lcm, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
    txt2img_lcm_animatediff_lcm_input.click(fn=import_to_module_video_prompt_only, inputs=[prompt_txt2img_lcm, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, tabs, tabs_video])

# txt2img_lcm both
    txt2img_lcm_img2img_both.click(fn=both_to_module_prompt_only, inputs=[prompt_txt2img_lcm, gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_lcm_img2img_ip_both.click(fn=both_to_module_prompt_only, inputs=[prompt_txt2img_lcm, gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    txt2img_lcm_pix2pix_both.click(fn=both_to_module_prompt_only, inputs=[prompt_txt2img_lcm, gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_lcm_inpaint_both.click(fn=both_to_module_inpaint_prompt_only, inputs=[prompt_txt2img_lcm, gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_lcm_controlnet_both.click(fn=both_to_module_inpaint_prompt_only, inputs=[prompt_txt2img_lcm, gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_lcm_faceid_ip_both.click(fn=both_to_module_prompt_only, inputs=[prompt_txt2img_lcm, gs_out_txt2img_lcm, sel_out_txt2img_lcm, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# txt2img_mjm outputs
    txt2img_mjm_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_mjm_img2img_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    txt2img_mjm_img2var.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    txt2img_mjm_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_mjm_magicmix.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    txt2img_mjm_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_mjm_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    txt2img_mjm_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    txt2img_mjm_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_mjm_faceid_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    txt2img_mjm_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    txt2img_mjm_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_mjm_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_mjm_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    txt2img_mjm_llava.click(fn=send_to_module_text, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    txt2img_mjm_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    txt2img_mjm_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# txt2img_mjm inputs
    txt2img_mjm_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2img_mjm_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    txt2img_mjm_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_mjm, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2img_mjm_txt2img_paa_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_mjm, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, tabs, tabs_image])
    txt2img_mjm_img2img_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    txt2img_mjm_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    txt2img_mjm_pix2pix_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    txt2img_mjm_inpaint_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    txt2img_mjm_controlnet_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    txt2img_mjm_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
    txt2img_mjm_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_mjm_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    txt2img_mjm_animatediff_lcm_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tabs, tabs_video])

# txt2img_mjm both
    txt2img_mjm_img2img_both.click(fn=both_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_mjm_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    txt2img_mjm_pix2pix_both.click(fn=both_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_mjm_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_mjm_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_mjm_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, gs_out_txt2img_mjm, sel_out_txt2img_mjm, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# txt2img_paa outputs
    txt2img_paa_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_paa_img2img_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    txt2img_paa_img2var.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    txt2img_paa_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_paa_magicmix.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    txt2img_paa_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_paa_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    txt2img_paa_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    txt2img_paa_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_paa_faceid_ip.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    txt2img_paa_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    txt2img_paa_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_paa_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_paa_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    txt2img_paa_llava.click(fn=send_to_module_text, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    txt2img_paa_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    txt2img_paa_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_txt2img_paa, sel_out_txt2img_paa, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# txt2img_paa inputs
    txt2img_paa_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2img_paa_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    txt2img_paa_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_paa, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2img_paa_txt2img_mjm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2img_paa, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, tabs, tabs_image])
    txt2img_paa_img2img_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    txt2img_paa_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    txt2img_paa_pix2pix_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    txt2img_paa_inpaint_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    txt2img_paa_controlnet_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    txt2img_paa_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
    txt2img_paa_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_paa_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    txt2img_paa_animatediff_lcm_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tabs, tabs_video])
# txt2img_paa both
    txt2img_paa_img2img_both.click(fn=both_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_paa_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    txt2img_paa_pix2pix_both.click(fn=both_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_paa_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_paa_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_paa_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, gs_out_txt2img_paa, sel_out_txt2img_paa, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# img2img outputs
    img2img_img2img.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    img2img_img2img_ip.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    img2img_img2var.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    img2img_pix2pix.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    img2img_magicmix.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    img2img_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2img_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    img2img_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    img2img_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    img2img_faceid_ip.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    img2img_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    img2img_resrgan.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    img2img_gfpgan.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    img2img_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_img2img, sel_out_img2img, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    img2img_llava.click(fn=send_to_module_text, inputs=[gs_out_img2img, sel_out_img2img, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    img2img_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_img2img, sel_out_img2img, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    img2img_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_img2img, sel_out_img2img, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# img2img inputs
    img2img_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    img2img_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    img2img_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_img2img, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image]) 
    img2img_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    img2img_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    img2img_pix2pix_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    img2img_inpaint_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    img2img_controlnet_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    img2img_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
# img2img both
    img2img_pix2pix_both.click(fn=both_to_module, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    img2img_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2img_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    img2img_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# img2img_ip outputs
    img2img_ip_img2img.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    img2img_ip_img2img_ip.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    img2img_ip_img2var.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    img2img_ip_pix2pix.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    img2img_ip_magicmix.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    img2img_ip_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2img_ip_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    img2img_ip_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    img2img_ip_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    img2img_ip_faceid_ip.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    img2img_ip_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    img2img_ip_resrgan.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    img2img_ip_gfpgan.click(fn=send_to_module, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    img2img_ip_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    img2img_ip_llava.click(fn=send_to_module_text, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    img2img_ip_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    img2img_ip_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_img2img_ip, sel_out_img2img_ip, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# img2img_ip inputs
    img2img_ip_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    img2img_ip_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    img2img_ip_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_img2img_ip, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    img2img_ip_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    img2img_ip_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    img2img_ip_pix2pix_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    img2img_ip_inpaint_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    img2img_ip_controlnet_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    img2img_ip_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
# img2img_ip both
    img2img_ip_pix2pix_both.click(fn=both_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    img2img_ip_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2img_ip_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    img2img_ip_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_img2img_ip, negative_prompt_img2img_ip, gs_out_img2img_ip, sel_out_img2img_ip, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# img2var outputs
    img2var_img2img.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    img2var_img2img_ip.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    img2var_img2var.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    img2var_pix2pix.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    img2var_magicmix.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    img2var_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2var_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    img2var_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    img2var_faceid_ip.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    img2var_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    img2var_resrgan.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    img2var_gfpgan.click(fn=send_to_module, inputs=[gs_out_img2var, sel_out_img2var, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    img2var_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_img2var, sel_out_img2var, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    img2var_llava.click(fn=send_to_module_text, inputs=[gs_out_img2var, sel_out_img2var, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    img2var_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_img2var, sel_out_img2var, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    img2var_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_img2var, sel_out_img2var, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# pix2pix outputs
    pix2pix_img2img.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    pix2pix_img2img_ip.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    pix2pix_img2var.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    pix2pix_pix2pix.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    pix2pix_magicmix.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    pix2pix_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    pix2pix_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    pix2pix_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    pix2pix_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    pix2pix_faceid_ip.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    pix2pix_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    pix2pix_resrgan.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    pix2pix_gfpgan.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    pix2pix_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    pix2pix_llava.click(fn=send_to_module_text, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    pix2pix_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    pix2pix_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# pix2pix inputs
    pix2pix_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    pix2pix_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    pix2pix_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_pix2pix, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    pix2pix_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    pix2pix_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    pix2pix_img2img_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    pix2pix_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    pix2pix_inpaint_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    pix2pix_controlnet_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    pix2pix_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
    pix2pix_vid2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_video_num, tab_vid2vid_ze_num], outputs=[prompt_vid2vid_ze, negative_prompt_vid2vid_ze, tabs, tabs_video])

# pix2pix both
    pix2pix_img2img_both.click(fn=both_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    pix2pix_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    pix2pix_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    pix2pix_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet,img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    pix2pix_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# magicmix outputs
    magicmix_img2img.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    magicmix_img2img_ip.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    magicmix_img2var.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    magicmix_pix2pix.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    magicmix_magicmix.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    magicmix_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    magicmix_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    magicmix_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    magicmix_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    magicmix_faceid_ip.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    magicmix_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    magicmix_resrgan.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    magicmix_gfpgan.click(fn=send_to_module, inputs=[gs_out_magicmix, sel_out_magicmix, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    magicmix_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_magicmix, sel_out_magicmix, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    magicmix_llava.click(fn=send_to_module_text, inputs=[gs_out_magicmix, sel_out_magicmix, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    magicmix_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_magicmix, sel_out_magicmix, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    magicmix_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_magicmix, sel_out_magicmix, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# inpaint outputs
    inpaint_img2img.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    inpaint_img2img_ip.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    inpaint_img2var.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    inpaint_pix2pix.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    inpaint_magicmix.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    inpaint_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    inpaint_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    inpaint_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    inpaint_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    inpaint_faceid_ip.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    inpaint_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    inpaint_resrgan.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    inpaint_gfpgan.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    inpaint_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_inpaint, sel_out_inpaint, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    inpaint_llava.click(fn=send_to_module_text, inputs=[gs_out_inpaint, sel_out_inpaint, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    inpaint_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_inpaint, sel_out_inpaint, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    inpaint_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_inpaint, sel_out_inpaint, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# inpaint inputs
    inpaint_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    inpaint_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    inpaint_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_inpaint, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    inpaint_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    inpaint_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    inpaint_img2img_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    inpaint_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    inpaint_pix2pix_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    inpaint_controlnet_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    inpaint_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])

# inpaint both
    inpaint_img2img_both.click(fn=both_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    inpaint_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    inpaint_pix2pix_both.click(fn=both_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    inpaint_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    inpaint_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# paintbyex outputs
    paintbyex_img2img.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    paintbyex_img2img_ip.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    paintbyex_img2var.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    paintbyex_pix2pix.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    paintbyex_magicmix.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    paintbyex_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    paintbyex_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    paintbyex_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    paintbyex_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    paintbyex_faceid_ip.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    paintbyex_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    paintbyex_resrgan.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    paintbyex_gfpgan.click(fn=send_to_module, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    paintbyex_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    paintbyex_llava.click(fn=send_to_module_text, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    paintbyex_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    paintbyex_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_paintbyex, sel_out_paintbyex, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# outpaint outputs
    outpaint_img2img.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    outpaint_img2img_ip.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    outpaint_img2var.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    outpaint_pix2pix.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    outpaint_magicmix.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    outpaint_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    outpaint_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    outpaint_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    outpaint_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    outpaint_faceid_ip.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    outpaint_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    outpaint_resrgan.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    outpaint_gfpgan.click(fn=send_to_module, inputs=[gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    outpaint_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_outpaint, sel_out_outpaint, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    outpaint_llava.click(fn=send_to_module_text, inputs=[gs_out_outpaint, sel_out_outpaint, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    outpaint_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_outpaint, sel_out_outpaint, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    outpaint_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_outpaint, sel_out_outpaint, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# outpaint inputs
    outpaint_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    outpaint_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    outpaint_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_outpaint, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    outpaint_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    outpaint_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    outpaint_img2img_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    outpaint_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    outpaint_pix2pix_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    outpaint_controlnet_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    outpaint_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])
    
# outpaint both
    outpaint_img2img_both.click(fn=both_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    outpaint_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    outpaint_pix2pix_both.click(fn=both_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    outpaint_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_outpaint, negative_prompt_outpaint, gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    outpaint_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_outpaint, negative_prompt_outpaint, gs_out_outpaint, sel_out_outpaint, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# ControlNet outputs
    controlnet_img2img.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    controlnet_img2img_ip.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    controlnet_img2var.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    controlnet_pix2pix.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    controlnet_magicmix.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    controlnet_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    controlnet_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    controlnet_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    controlnet_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    controlnet_faceid_ip.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    controlnet_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    controlnet_resrgan.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    controlnet_gfpgan.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    controlnet_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_controlnet, sel_out_controlnet, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    controlnet_llava.click(fn=send_to_module_text, inputs=[gs_out_controlnet, sel_out_controlnet, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    controlnet_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_controlnet, sel_out_controlnet, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    controlnet_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_controlnet, sel_out_controlnet, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# controlnet inputs
    controlnet_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    controlnet_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    controlnet_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_controlnet, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    controlnet_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    controlnet_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    controlnet_img2img_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    controlnet_img2img_ip_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, tabs, tabs_image])
    controlnet_pix2pix_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    controlnet_inpaint_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    controlnet_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    controlnet_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    controlnet_animatediff_lcm_input.click(fn=import_to_module_video, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tabs, tabs_video])
    controlnet_faceid_ip_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tabs, tabs_image])

# ControlNet both
    controlnet_img2img_both.click(fn=both_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    controlnet_img2img_ip_both.click(fn=both_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2img_ip_num], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip, img_img2img_ip, tabs, tabs_image])
    controlnet_pix2pix_both.click(fn=both_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    controlnet_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    controlnet_faceid_ip_both.click(fn=both_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_faceid_ip_num], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip, img_faceid_ip, tabs, tabs_image])

# faceid_ip outputs
    faceid_ip_img2img.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    faceid_ip_img2img_ip.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    faceid_ip_img2var.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    faceid_ip_pix2pix.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    faceid_ip_magicmix.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    faceid_ip_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    faceid_ip_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    faceid_ip_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    faceid_ip_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    faceid_ip_faceid_ip.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    faceid_ip_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    faceid_ip_resrgan.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    faceid_ip_gfpgan.click(fn=send_to_module, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    faceid_ip_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    faceid_ip_llava.click(fn=send_to_module_text, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    faceid_ip_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    faceid_ip_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_faceid_ip, sel_out_faceid_ip, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# faceid_ip inputs
    faceid_ip_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    faceid_ip_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    faceid_ip_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_faceid_ip, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    faceid_ip_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    faceid_ip_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])
    faceid_ip_pix2pix_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    faceid_ip_inpaint_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    faceid_ip_controlnet_input.click(fn=import_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    
# faceid_ip both
    faceid_ip_pix2pix_both.click(fn=both_to_module, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    faceid_ip_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    faceid_ip_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_faceid_ip, negative_prompt_faceid_ip, gs_out_faceid_ip, sel_out_faceid_ip, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])

# Faceswap outputs
    faceswap_img2img.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    faceswap_img2img_ip.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    faceswap_img2var.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    faceswap_pix2pix.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    faceswap_magicmix.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    faceswap_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    faceswap_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    faceswap_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    faceswap_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet,  gs_img_source_controlnet, tabs, tabs_image])
    faceswap_faceid_ip.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    faceswap_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    faceswap_resrgan.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    faceswap_gfpgan.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    faceswap_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_faceswap, sel_out_faceswap, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    faceswap_llava.click(fn=send_to_module_text, inputs=[gs_out_faceswap, sel_out_faceswap, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    faceswap_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_faceswap, sel_out_faceswap, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    faceswap_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_faceswap, sel_out_faceswap, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# resrgan outputs
    resrgan_img2img.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    resrgan_img2img_ip.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    resrgan_img2var.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    resrgan_pix2pix.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    resrgan_magicmix.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    resrgan_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    resrgan_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    resrgan_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    resrgan_controlnet.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    resrgan_faceid_ip.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    resrgan_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    resrgan_gfpgan.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    resrgan_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_resrgan, sel_out_resrgan, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    resrgan_llava.click(fn=send_to_module_text, inputs=[gs_out_resrgan, sel_out_resrgan, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    resrgan_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_resrgan, sel_out_resrgan, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    resrgan_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_resrgan, sel_out_resrgan, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# gfpgan outputs
    gfpgan_img2img.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    gfpgan_img2img_ip.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_img2img_ip_num], outputs=[img_img2img_ip, tabs, tabs_image])
    gfpgan_img2var.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_img2var_num], outputs=[img_img2var, tabs, tabs_image])
    gfpgan_pix2pix.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    gfpgan_magicmix.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_magicmix_num], outputs=[img_magicmix, tabs, tabs_image])
    gfpgan_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    gfpgan_paintbyex.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_paintbyex_num], outputs=[img_paintbyex, gs_img_paintbyex, tabs, tabs_image])
    gfpgan_outpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_outpaint_num], outputs=[img_outpaint, gs_img_outpaint, tabs, tabs_image])
    gfpgan_controlnet.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    gfpgan_faceid_ip.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_faceid_ip_num], outputs=[img_faceid_ip, tabs, tabs_image])
    gfpgan_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    gfpgan_resrgan.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    gfpgan_img2vid.click(fn=send_image_to_module_video, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_video_num, tab_img2vid_num], outputs=[img_img2vid, tabs, tabs_video])
    gfpgan_llava.click(fn=send_to_module_text, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_text_num, tab_llava_num], outputs=[img_llava, tabs, tabs_text])
    gfpgan_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    gfpgan_img2shape.click(fn=send_to_module_3d, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_3d_num, tab_img2shape_num], outputs=[img_img2shape, tabs, tabs_3d])

# Musicgen outputs
    musicgen_musicgen_mel.click(fn=import_to_module_audio, inputs=[out_musicgen, tab_audio_num, tab_musicgen_mel_num], outputs=[source_audio_musicgen_mel, tabs, tabs_audio])

# Musicgen inputs
    musicgen_musicgen_mel_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen, tab_audio_num, tab_musicgen_mel_num], outputs=[prompt_musicgen_mel, tabs, tabs_audio])
    musicgen_musicldm_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen, tab_audio_num, tab_musicldm_num], outputs=[prompt_musicldm, tabs, tabs_audio])
    musicgen_audiogen_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])

#Musicgen melody outputs
    musicgen_mel_musicgen_mel.click(fn=import_to_module_audio, inputs=[out_musicgen_mel, tab_audio_num, tab_musicgen_mel_num], outputs=[source_audio_musicgen_mel, tabs, tabs_audio])

#Musicgen melody inputs
    musicgen_mel_musicgen_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen_mel, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    musicgen_mel_musicldm_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen_mel, tab_audio_num, tab_musicldm_num], outputs=[prompt_musicldm, tabs, tabs_audio])
    musicgen_mel_audiogen_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen_mel, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    
# Musicgen outputs
    musicldm_musicgen_mel.click(fn=import_to_module_audio, inputs=[out_musicldm, tab_audio_num, tab_musicgen_mel_num], outputs=[source_audio_musicgen_mel, tabs, tabs_audio])

# Musicgen inputs
    musicldm_musicgen_input.click(fn=import_to_module_audio, inputs=[prompt_musicldm, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    musicldm_musicgen_mel_input.click(fn=import_to_module_audio, inputs=[prompt_musicldm, tab_audio_num, tab_musicgen_mel_num], outputs=[prompt_musicgen_mel, tabs, tabs_audio])
    musicldm_audiogen_input.click(fn=import_to_module_audio, inputs=[prompt_musicldm, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])

# Audiogen outputs
    audiogen_musicgen_mel.click(fn=import_to_module_audio, inputs=[out_audiogen, tab_audio_num, tab_musicgen_mel_num], outputs=[source_audio_musicgen_mel, tabs, tabs_audio])
    
# Audiogen inputs    
    audiogen_musicgen_input.click(fn=import_to_module_audio, inputs=[prompt_audiogen, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])
    audiogen_musicgen_mel_input.click(fn=import_to_module_audio, inputs=[prompt_audiogen, tab_audio_num, tab_musicgen_mel_num], outputs=[prompt_musicgen_mel, tabs, tabs_audio])
    audiogen_musicldm_input.click(fn=import_to_module_audio, inputs=[prompt_audiogen, tab_audio_num, tab_musicldm_num], outputs=[prompt_musicldm, tabs, tabs_audio])

# Harmonai outputs
    harmonai_musicgen_mel.click(fn=import_to_module_audio, inputs=[out_harmonai, tab_audio_num, tab_musicgen_mel_num], outputs=[source_audio_musicgen_mel, tabs, tabs_audio])

# Bark outputs
    bark_musicgen_mel.click(fn=import_to_module_audio, inputs=[out_bark, tab_audio_num, tab_musicgen_mel_num], outputs=[source_audio_musicgen_mel, tabs, tabs_audio])

# Bark inputs
    bark_whisper.click(fn=send_audio_to_module_text, inputs=[out_bark, tab_text_num, tab_whisper_num], outputs=[source_audio_whisper, tabs, tabs_text])

# Modelscope outputs
    txt2vid_ms_vid2vid_ze.click(fn=send_to_module_video, inputs=[out_txt2vid_ms, tab_video_num, tab_vid2vid_ze_num], outputs=[vid_vid2vid_ze, tabs, tabs_video])

# Modelscope inputs    
    txt2vid_ms_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    txt2vid_ms_animatediff_lcm_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_video_num, tab_animatediff_lcm_num], outputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tabs, tabs_video])
    txt2vid_ms_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2vid_ms_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    txt2vid_ms_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2vid_ms, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2vid_ms_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    txt2vid_ms_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])

# Text2Video-Zero outputs
    txt2vid_ze_vid2vid_ze.click(fn=send_to_module_video, inputs=[out_txt2vid_ze, tab_video_num, tab_vid2vid_ze_num], outputs=[vid_vid2vid_ze, tabs, tabs_video])

# Text2Video-Zero inputs    
    txt2vid_ze_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2vid_ze_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2vid_ze_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    txt2vid_ze_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_txt2vid_ze, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    txt2vid_ze_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    txt2vid_ze_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])

# AnimateDiff outputs
    animatediff_lcm_vid2vid_ze.click(fn=send_to_module_video, inputs=[out_animatediff_lcm, tab_video_num, tab_vid2vid_ze_num], outputs=[vid_vid2vid_ze, tabs, tabs_video])

# AnimateDiff inputs
    animatediff_lcm_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    animatediff_lcm_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    animatediff_lcm_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    animatediff_lcm_txt2img_lcm_input.click(fn=import_to_module_prompt_only, inputs=[prompt_animatediff_lcm, tab_image_num, tab_txt2img_lcm_num], outputs=[prompt_txt2img_lcm, tabs, tabs_image])
    animatediff_lcm_txt2img_mjm_input.click(fn=import_to_module, inputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tab_image_num, tab_txt2img_mjm_num], outputs=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm, tabs, tabs_image])
    animatediff_lcm_txt2img_paa_input.click(fn=import_to_module, inputs=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm, tab_image_num, tab_txt2img_paa_num], outputs=[prompt_txt2img_paa, negative_prompt_txt2img_paa, tabs, tabs_image])

# Stable Video Diffusion
    img2vid_vid2vid_ze.click(fn=send_to_module_video, inputs=[out_img2vid, tab_video_num, tab_vid2vid_ze_num], outputs=[vid_vid2vid_ze, tabs, tabs_video])
    
# Video Instruct-pix2pix inputs
    vid2vid_ze_pix2pix.click(fn=import_to_module, inputs=[prompt_vid2vid_ze, negative_prompt_vid2vid_ze, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])

# Console output
    with gr.Accordion(biniou_lang_console_title, open=False):
        with gr.Row():
            with gr.Column():
                biniou_console_output = gr.Textbox(label=biniou_lang_console_output, value="", lines=5, max_lines=5, show_copy_button=True)
        with gr.Row():
            with gr.Column():
                download_file_console = gr.File(label=biniou_lang_console_down, value=logfile_biniou, height=30, interactive=False)
                biniou_console_output.change(refresh_logfile, None, download_file_console)
            with gr.Column():
                gr.Number(visible=False)
            with gr.Column():
                gr.Number(visible=False)
            with gr.Column():
                gr.Number(visible=False)

# UI execution:
    demo.load(split_url_params, nsfw_filter, [nsfw_filter, url_params_current, safety_checker_ui_settings], _js=get_window_url_params)
    demo.load(read_logs, None, biniou_console_output, every=1)
#    demo.load(fn=lambda: gr.Info('Biniou loading completed. Ready to work !'))
    if biniou_global_server_name:
        print(f">>>[biniou 🧠]: Up and running at https://{local_ip()}:{biniou_global_server_port}")
    else:
        print(f">>>[biniou 🧠]: Up and running at https://127.0.0.1:{biniou_global_server_port}")

if __name__ == "__main__":
    demo.queue(concurrency_count=8).launch(
        server_name="0.0.0.0" if biniou_global_server_name else "127.0.0.1",
        server_port=biniou_global_server_port,
        ssl_certfile="./ssl/cert.pem" if not biniou_global_share else None,
        favicon_path="./images/biniou_64.ico",
        ssl_keyfile="./ssl/key.pem" if not biniou_global_share else None,
        ssl_verify=False,
        auth=biniou_auth_values if biniou_global_auth else None,
        auth_message=biniou_global_auth_message if biniou_global_auth else None,
        share=biniou_global_share,
        inbrowser=biniou_global_inbrowser,
#        inbrowser=True if len(sys.argv)>1 and sys.argv[1]=="--inbrowser" else biniou_global_inbrowser,
    )
# EOF
