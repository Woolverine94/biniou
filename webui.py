# https://github.com/PixifyAI/pixify-webui
# Webui.py
import diffusers
diffusers.utils.USE_PEFT_BACKEND = False
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings('ignore')
import os
import gradio as gr
import numpy as np
import shutil
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
if os.path.exists(tmp_biniou) :
    shutil.rmtree(tmp_biniou)
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

biniou_global_server_name = True
biniou_global_server_port = 7860
biniou_global_inbrowser = False
biniou_global_auth = False
biniou_global_auth_message = "Welcome to pixify !"
biniou_global_share = False
biniou_global_steps_max = 100
biniou_global_batch_size_max = 4
biniou_global_width_max_img_create = 1280
biniou_global_height_max_img_create = 1280
biniou_global_width_max_img_modify = 8192
biniou_global_height_max_img_modify = 8192
biniou_global_sd15_width = 512
biniou_global_sd15_height = 512
biniou_global_sdxl_width = 1024
biniou_global_sdxl_height = 1024
biniou_global_gfpgan = True
biniou_global_tkme = 0.6

if test_cfg_exist("settings") :
    with open(".ini/settings.cfg", "r", encoding="utf-8") as fichier:
        exec(fichier.read())

if not os.path.isfile(".ini/auth.cfg"):
    write_auth("biniou:biniou")

if biniou_global_auth == True:
    biniou_auth_values = read_auth()

if biniou_global_auth == False:
    biniou_global_share = False

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
    
def send_to_module_video(content, numtab, numtab_item) : 
	return content, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)

def send_image_to_module_video(content, index, numtab, numtab_item) : 
	index = int(index)
	return content[index], gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)

def send_to_module_3d(content, index, numtab, numtab_item) :
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
def read_ini_llamacpp(module) :
    content = read_ini(module)
    if (len(content))>10:
        return str(content[0]), int(content[1]), int(content[2]), bool(int(content[3])), int(content[4]), float(content[5]), float(content[6]), float(content[7]), int(content[8]), str(content[9]), str(content[10])
    else:
        return str(content[0]), int(content[1]), int(content[2]), bool(int(content[3])), int(content[4]), float(content[5]), float(content[6]), float(content[7]), int(content[8]), str(content[9])

def show_download_llamacpp() :
    return btn_download_file_llamacpp.update(visible=False), download_file_llamacpp.update(visible=True)

def hide_download_llamacpp() :
    return btn_download_file_llamacpp.update(visible=True), download_file_llamacpp.update(visible=False)

def change_model_type_llamacpp(model_llamacpp):
    try:
        test_model = model_list_llamacpp[model_llamacpp]
    except KeyError as ke:
        test_model = None
    if (test_model != None):
        return prompt_template_llamacpp.update(value=model_list_llamacpp[model_llamacpp][1]), system_template_llamacpp.update(value=model_list_llamacpp[model_llamacpp][2])
    else:
        return prompt_template_llamacpp.update(value="{prompt}"), system_template_llamacpp.update(value="")

def change_prompt_template_llamacpp(prompt_template):
    return prompt_template_llamacpp.update(value=prompt_template_list_llamacpp[prompt_template][0]), system_template_llamacpp.update(value=prompt_template_list_llamacpp[prompt_template][1])

## Functions specific to llava
def read_ini_llava(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), int(content[2]), bool(int(content[3])), int(content[4]), float(content[5]), float(content[6]), float(content[7]), int(content[8]), str(content[9])

def show_download_llava() :
    return btn_download_file_llava.update(visible=False), download_file_llava.update(visible=True)

def hide_download_llava() :
    return btn_download_file_llava.update(visible=True), download_file_llava.update(visible=False)

# def change_model_type_llava(model_llava):
#     return prompt_template_llava.update(value=model_list_llava[model_llava][1])


        
## Functions specific to img2txt_git
def read_ini_img2txt_git(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), int(content[2]), int(content[3]), int(content[4]), float(content[5])

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

def read_ini_whisper(module) :
    content = read_ini(module)
    return str(content[0]), bool(int(content[1]))
    
## Functions specific to nllb

def read_ini_nllb(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1])

## Functions specific to txt2prompt
def read_ini_txt2prompt(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), float(content[2]), int(content[3]), int(content[4])

def change_output_type_txt2prompt(output_type_txt2prompt) : 
    if output_type_txt2prompt == "ChatGPT" :
        return model_txt2prompt.update(value=model_list_txt2prompt[0]), max_tokens_txt2prompt.update(value=128)
    elif output_type_txt2prompt == "SD" :
        return model_txt2prompt.update(value=model_list_txt2prompt[1]), max_tokens_txt2prompt.update(value=70) 

## Functions specific to Stable Diffusion 
def zip_download_file_txt2img_sd(content):
    savename = zipper(content)
    return savename, download_file_txt2img_sd.update(visible=True) 

def hide_download_file_txt2img_sd():
    return download_file_txt2img_sd.update(visible=False)

def change_model_type_txt2img_sd(model_txt2img_sd):
    if (model_txt2img_sd == "stabilityai/sdxl-turbo"):
        return sampler_txt2img_sd.update(value="Euler a"), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=1), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=False)
    elif (model_txt2img_sd == "thibaud/sdxl_dpo_turbo"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=2), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=False)
    elif (model_txt2img_sd == "stabilityai/sd-turbo"):
        return sampler_txt2img_sd.update(value="Euler a"), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=1), guidance_scale_txt2img_sd.update(value=0.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=False)
    elif ("XL" in model_txt2img_sd.upper()) or ("ETRI-VILAB/KOALA-" in model_txt2img_sd.upper()) or (model_txt2img_sd == "dataautogpt3/OpenDalleV1.1") or (model_txt2img_sd == "dataautogpt3/ProteusV0.4"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "segmind/SSD-1B"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=True)
    elif (model_txt2img_sd == "segmind/Segmind-Vega"):
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sdxl_width), height_txt2img_sd.update(value=biniou_global_sdxl_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=9.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=False), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=True)
    else:
        return sampler_txt2img_sd.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2img_sd.update(value=biniou_global_sd15_width), height_txt2img_sd.update(value=biniou_global_sd15_height), num_inference_step_txt2img_sd.update(value=10), guidance_scale_txt2img_sd.update(value=7.0), lora_model_txt2img_sd.update(choices=list(lora_model_list(model_txt2img_sd).keys()), value="", interactive=True), txtinv_txt2img_sd.update(choices=list(txtinv_list(model_txt2img_sd).keys()), value=""), negative_prompt_txt2img_sd.update(interactive=True)

def change_lora_model_txt2img_sd(model, lora_model, prompt):
    if lora_model != "":
        lora_keyword = lora_model_list(model)[lora_model][1]
        if lora_keyword != "":		
            lora_prompt_txt2img_sd = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_sd = prompt
    else:
        lora_prompt_txt2img_sd = prompt
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

def read_ini_txt2img_sd(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])

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

def read_ini_txt2img_kd(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9]))

## Functions specific to LCM
def zip_download_file_txt2img_lcm(content):
    savename = zipper(content)
    return savename, download_file_txt2img_lcm.update(visible=True) 

def hide_download_file_txt2img_lcm():
    return download_file_txt2img_lcm.update(visible=False)
    
def update_preview_txt2img_lcm(preview):
    return out_txt2img_lcm.update(preview)     

def read_ini_txt2img_lcm(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), bool(int(content[10])), float(content[11])

def change_model_type_txt2img_lcm(model_txt2img_lcm):
    if (model_txt2img_lcm == "latent-consistency/lcm-ssd-1b"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=False), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "latent-consistency/lcm-sdxl"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=8.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=False), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "latent-consistency/lcm-lora-sdxl"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "latent-consistency/lcm-lora-sdv1-5"):
        return width_txt2img_lcm.update(value=biniou_global_sd15_width), height_txt2img_lcm.update(value=biniou_global_sd15_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    elif (model_txt2img_lcm == "segmind/Segmind-VegaRT"):
        return width_txt2img_lcm.update(value=biniou_global_sdxl_width), height_txt2img_lcm.update(value=biniou_global_sdxl_height), guidance_scale_txt2img_lcm.update(value=0.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=False), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")
    else:
        return width_txt2img_lcm.update(value=biniou_global_sd15_width), height_txt2img_lcm.update(value=biniou_global_sd15_height), guidance_scale_txt2img_lcm.update(value=8.0), num_inference_step_txt2img_lcm.update(value=4), lora_model_txt2img_lcm.update(choices=list(lora_model_list(model_txt2img_lcm).keys()), value="", interactive=True), txtinv_txt2img_lcm.update(choices=list(txtinv_list(model_txt2img_lcm).keys()), value="")

def change_lora_model_txt2img_lcm(model, lora_model, prompt):
    if lora_model != "":
        lora_keyword = lora_model_list(model)[lora_model][1]
        if lora_keyword != "":		
            lora_prompt_txt2img_lcm = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_txt2img_lcm = prompt
    else:
        lora_prompt_txt2img_lcm = prompt
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
    
def read_ini_txt2img_mjm(module):
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])

## Functions specific to PixArt-Alpha
def zip_download_file_txt2img_paa(content):
    savename = zipper(content)
    return savename, download_file_txt2img_paa.update(visible=True) 

def hide_download_file_txt2img_paa():
    return download_file_txt2img_paa.update(visible=False)
    
def read_ini_txt2img_paa(module):
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])

def change_model_type_txt2img_paa(model_txt2img_paa):
    if model_txt2img_paa == "PixArt-alpha/PixArt-XL-2-1024-MS":
        return width_txt2img_paa.update(value=biniou_global_sdxl_width), height_txt2img_paa.update(value=biniou_global_sdxl_height)
    else:
        return width_txt2img_paa.update(value=biniou_global_sd15_width), height_txt2img_paa.update(value=biniou_global_sd15_height)

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

def read_ini_img2img(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])

def change_model_type_img2img(model_img2img):
    if (model_img2img == "stabilityai/sdxl-turbo"):
        return sampler_img2img.update(value="Euler a"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=False)
    elif (model_img2img == "thibaud/sdxl_dpo_turbo"):
        return sampler_img2img.update(value="UniPC"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=False)
    elif (model_img2img == "stabilityai/sd-turbo"):
        return sampler_img2img.update(value="Euler a"), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=2), guidance_scale_img2img.update(value=0.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=False)
    elif ("XL" in model_img2img.upper()) or ("ETRI-VILAB/KOALA-" in model_img2img.upper()) or (model_img2img == "dataautogpt3/OpenDalleV1.1")  or (model_img2img == "dataautogpt3/ProteusV0.4"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "segmind/SSD-1B"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=True)
    elif (model_img2img == "segmind/Segmind-Vega"):
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=9.0), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=False), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=True)
    else:
        return sampler_img2img.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img.update(), height_img2img.update(), num_inference_step_img2img.update(value=10), guidance_scale_img2img.update(value=7.5), lora_model_img2img.update(choices=list(lora_model_list(model_img2img).keys()), value="", interactive=True), txtinv_img2img.update(choices=list(txtinv_list(model_img2img).keys()), value=""), negative_prompt_img2img.update(interactive=True)

def change_lora_model_img2img(model, lora_model, prompt):
    if lora_model != "":
        lora_keyword = lora_model_list(model)[lora_model][1]
        if lora_keyword != "":
            lora_prompt_img2img = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img = prompt
    else:
        lora_prompt_img2img = prompt
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
    
def read_ini_img2img_ip(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])

def change_model_type_img2img_ip(model_img2img_ip):
    if (model_img2img_ip == "stabilityai/sdxl-turbo"):
        return sampler_img2img_ip.update(value="Euler a"), width_img2img_ip.update(), height_img2img_ip.update(), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=False)
#    elif (model_img2img_ip == "thibaud/sdxl_dpo_turbo"):
#        return sampler_img2img_ip.update(value="UniPC"), width_img2img_ip.update(value=biniou_global_sd15_width), height_img2img_ip.update(value=biniou_global_sd15_height), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=False)
    elif (model_img2img_ip == "stabilityai/sd-turbo"):
        return sampler_img2img_ip.update(value="Euler a"), width_img2img_ip.update(), height_img2img_ip.update(), num_inference_step_img2img_ip.update(value=2), guidance_scale_img2img_ip.update(value=0.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=False)
    elif ("XL" in model_img2img_ip.upper()) or ("ETRI-VILAB/KOALA-" in model_img2img_ip.upper()) or (model_img2img_ip == "dataautogpt3/OpenDalleV1.1") or (model_img2img_ip == "dataautogpt3/ProteusV0.4"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(), height_img2img_ip.update(), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True)
    elif (model_img2img_ip == "segmind/SSD-1B"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(), height_img2img_ip.update(), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True)
    elif (model_img2img_ip == "segmind/Segmind-Vega"):
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(), height_img2img_ip.update(), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=9.0), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=False), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True)
    else:
        return sampler_img2img_ip.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_img2img_ip.update(), height_img2img_ip.update(), num_inference_step_img2img_ip.update(value=10), guidance_scale_img2img_ip.update(value=7.5), lora_model_img2img_ip.update(choices=list(lora_model_list(model_img2img_ip).keys()), value="", interactive=True), txtinv_img2img_ip.update(choices=list(txtinv_list(model_img2img_ip).keys()), value=""), negative_prompt_img2img_ip.update(interactive=True)

def change_lora_model_img2img_ip(model, lora_model, prompt):
    if lora_model != "":
        lora_keyword = lora_model_list(model)[lora_model][1]
        if lora_keyword != "":
            lora_prompt_img2img_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_img2img_ip = prompt
    else:
        lora_prompt_img2img_ip = prompt
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

## Functions specific to img2var 
def zip_download_file_img2var(content):
    savename = zipper(content)
    return savename, download_file_img2var.update(visible=True) 

def hide_download_file_img2var():
    return download_file_img2var.update(visible=False)        
    
def read_ini_img2var(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])    

## Functions specific to pix2pix 
def zip_download_file_pix2pix(content):
    savename = zipper(content)
    return savename, download_file_pix2pix.update(visible=True) 

def hide_download_file_pix2pix():
    return download_file_pix2pix.update(visible=False) 
    
def read_ini_pix2pix(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), float(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), bool(int(content[10])), float(content[11])

## Functions specific to magicmix 
def zip_download_file_magicmix(content):
    savename = zipper(content)
    return savename, download_file_magicmix.update(visible=True) 

def hide_download_file_magicmix():
    return download_file_magicmix.update(visible=False) 
    
def read_ini_magicmix(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), float(content[4]), float(content[5]), int(content[6]), int(content[7]), bool(int(content[8])), float(content[9])
   
## Functions specific to inpaint 
def zip_download_file_inpaint(content):
    savename = zipper(content)
    return savename, download_file_inpaint.update(visible=True) 

def hide_download_file_inpaint():
    return download_file_inpaint.update(visible=False) 

def read_ini_inpaint(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])    

## Functions specific to paintbyex 
def zip_download_file_paintbyex(content):
    savename = zipper(content)
    return savename, download_file_paintbyex.update(visible=True) 

def hide_download_file_paintbyex():
    return download_file_paintbyex.update(visible=False) 

def read_ini_paintbyex(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])        

## Functions specific to outpaint 
def zip_download_file_outpaint(content):
    savename = zipper(content)
    return savename, download_file_outpaint.update(visible=True) 

def hide_download_file_outpaint():
    return download_file_outpaint.update(visible=False) 

def read_ini_outpaint(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])    

## Functions specific to controlnet 
def zip_download_file_controlnet(content):
    savename = zipper(content)
    return savename, download_file_controlnet.update(visible=True) 

def hide_download_file_controlnet():
    return download_file_controlnet.update(visible=False) 

def read_ini_controlnet(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), int(content[10]), float(content[11]), float(content[12]), float(content[13]), bool(int(content[14])), float(content[15])    

def change_model_type_controlnet(model_controlnet):
    if (model_controlnet == "stabilityai/sdxl-turbo"):
        return sampler_controlnet.update(value="Euler a"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=1), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "thibaud/sdxl_dpo_turbo"):
        return sampler_controlnet.update(value="UniPC"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=2), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "stabilityai/sd-turbo"):
        return sampler_controlnet.update(value="Euler a"), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=1), guidance_scale_controlnet.update(value=0.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=False), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif ("XL" in model_controlnet.upper()) or ("ETRI-VILAB/KOALA-" in model_controlnet.upper()) or (model_controlnet == "dataautogpt3/OpenDalleV1.1") or (model_controlnet == "dataautogpt3/ProteusV0.4"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "segmind/SSD-1B"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    elif (model_controlnet == "segmind/Segmind-Vega"):
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=9.0), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=False), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)
    else:
        return sampler_controlnet.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_controlnet.update(), height_controlnet.update(), num_inference_step_controlnet.update(value=10), guidance_scale_controlnet.update(value=7.5), lora_model_controlnet.update(choices=list(lora_model_list(model_controlnet).keys()), value="", interactive=True), txtinv_controlnet.update(choices=list(txtinv_list(model_controlnet).keys()), value=""), negative_prompt_controlnet.update(interactive=True), img_preview_controlnet.update(value=None), gs_img_preview_controlnet.update(value=None)

def change_lora_model_controlnet(model, lora_model, prompt):
    if lora_model != "":
        lora_keyword = lora_model_list(model)[lora_model][1]
        if lora_keyword != "":
            lora_prompt_controlnet = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_controlnet = prompt
    else:
        lora_prompt_controlnet = prompt
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

def read_ini_faceid_ip(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9])), float(content[10])

def change_model_type_faceid_ip(model_faceid_ip):
    if (model_faceid_ip == "stabilityai/sdxl-turbo"):
        return sampler_faceid_ip.update(value="Euler a"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=0.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=False)
#    elif (model_faceid_ip == "thibaud/sdxl_dpo_turbo"):
#        return sampler_faceid_ip.update(value="UniPC"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=0.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=False)
    elif (model_faceid_ip == "stabilityai/sd-turbo"):
        return sampler_faceid_ip.update(value="Euler a"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=2), guidance_scale_faceid_ip.update(value=0.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=False)
    elif ("XL" in model_faceid_ip.upper()) or (model_faceid_ip == "dataautogpt3/OpenDalleV1.1"):
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=25), guidance_scale_faceid_ip.update(value=7.5), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True)
    elif (model_faceid_ip == "segmind/SSD-1B"):
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=25), guidance_scale_faceid_ip.update(value=7.5), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True)
    elif (model_faceid_ip == "segmind/Segmind-Vega"):
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=25), guidance_scale_faceid_ip.update(value=9.0), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=False), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True)
    else:
        return sampler_faceid_ip.update(value="DDIM"), width_faceid_ip.update(), height_faceid_ip.update(), num_inference_step_faceid_ip.update(value=25), guidance_scale_faceid_ip.update(value=7.5), lora_model_faceid_ip.update(choices=list(lora_model_list(model_faceid_ip).keys()), value="", interactive=True), txtinv_faceid_ip.update(choices=list(txtinv_list(model_faceid_ip).keys()), value=""), negative_prompt_faceid_ip.update(interactive=True)

def change_lora_model_faceid_ip(model, lora_model, prompt):
    if lora_model != "":
        lora_keyword = lora_model_list(model)[lora_model][1]
        if lora_keyword != "":
            lora_prompt_faceid_ip = lora_keyword+ ", "+ prompt
        else:
            lora_prompt_faceid_ip = prompt
    else:
        lora_prompt_faceid_ip = prompt
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

def read_ini_faceswap(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), int(content[2]), bool(int(content[3]))

## Functions specific to Real ESRGAN
def read_ini_resrgan(module) :
    content = read_ini(module)
    return str(content[0]), str(content[1]), int(content[2]), int(content[3]), bool(int(content[4]))

## Functions specific to GFPGAN
def read_ini_gfpgan(module) :
    content = read_ini(module)
    return str(content[0]), str(content[1]), int(content[2]), int(content[3])

## Functions specific to MusicGen
def read_ini_musicgen(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), float(content[2]), int(content[3]), bool(int(content[4])), float(content[5]), int(content[6]), int(content[7])

## Functions specific to MusicGen Melody
def read_ini_musicgen_mel(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), float(content[2]), int(content[3]), bool(int(content[4])), float(content[5]), int(content[6]), int(content[7])

def change_source_type_musicgen_mel(source_type_musicgen_mel):
    if source_type_musicgen_mel == "audio" :
        return source_audio_musicgen_mel.update(source="upload")
    elif source_type_musicgen_mel == "micro" :
        return source_audio_musicgen_mel.update(source="microphone")

## Functions specific to MusicLDM
def read_ini_musicldm(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7])

## Functions specific to AudioGen
def read_ini_audiogen(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), float(content[2]), int(content[3]), bool(int(content[4])), float(content[5]), int(content[6]), int(content[7])

## Functions specific to Harmonai
def read_ini_harmonai(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), int(content[2]), int(content[3]), int(content[4]), int(content[5])

## Functions specific to Bark
def read_ini_bark(module) :
    content = read_ini(module)
    return str(content[0]), str(content[1])
    
## Functions specific to Modelscope
def read_ini_txt2vid_ms(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), bool(int(content[9]))

## Functions specific to Text2Video-Zero
def read_ini_txt2vid_ze(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), int(content[10]), int(content[11]), int(content[12]), int(content[13]), int(content[14]), int(content[15]), bool(int(content[16])), float(content[17])

def change_model_type_txt2vid_ze(model_txt2vid_ze):
    if (model_txt2vid_ze == "stabilityai/sdxl-turbo"):
        return sampler_txt2vid_ze.update(value="Euler a"), width_txt2vid_ze.update(), height_txt2vid_ze.update(), num_inference_step_txt2vid_ze.update(value=2), guidance_scale_txt2vid_ze.update(value=0.0), negative_prompt_txt2vid_ze.update(interactive=False)
    elif ("XL" in model_txt2vid_ze.upper()) or ("ETRI-VILAB/KOALA-" in model_txt2vid_ze.upper()) or (model_txt2vid_ze == "segmind/SSD-1B") or (model_txt2vid_ze == "dataautogpt3/OpenDalleV1.1") or (model_txt2vid_ze == "dataautogpt3/ProteusV0.4"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(), height_txt2vid_ze.update(), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=7.5), negative_prompt_txt2vid_ze.update(interactive=True)
    elif (model_txt2vid_ze == "segmind/Segmind-Vega"):
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(), height_txt2vid_ze.update(), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=9.0), negative_prompt_txt2vid_ze.update(interactive=True)
    else:
        return sampler_txt2vid_ze.update(value=list(SCHEDULER_MAPPING.keys())[0]), width_txt2vid_ze.update(), height_txt2vid_ze.update(), num_inference_step_txt2vid_ze.update(value=10), guidance_scale_txt2vid_ze.update(value=7.5), negative_prompt_txt2vid_ze.update(interactive=True)

## Functions specific to AnimateLCM
def read_ini_animatediff_lcm(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), bool(int(content[10])), float(content[11])

def change_model_type_animatediff_lcm(model_animatediff_lcm):
    if (model_animatediff_lcm == "stabilityai/sdxl-turbo"):
        return sampler_animatediff_lcm.update(value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=2), guidance_scale_animatediff_lcm.update(value=0.0), negative_prompt_animatediff_lcm.update(interactive=False)
    elif ("XL" in model_animatediff_lcm.upper()) or ("ETRI-VILAB/KOALA-" in model_animatediff_lcm.upper()) or (model_animatediff_lcm == "segmind/SSD-1B") or (model_animatediff_lcm == "dataautogpt3/OpenDalleV1.1") or (model_animatediff_lcm == "dataautogpt3/ProteusV0.4"):
        return sampler_animatediff_lcm.update(value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=10), guidance_scale_animatediff_lcm.update(value=7.5), negative_prompt_animatediff_lcm.update(interactive=True)
    elif (model_animatediff_lcm == "segmind/Segmind-Vega"):
        return sampler_animatediff_lcm.update(value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=10), guidance_scale_animatediff_lcm.update(value=9.0), negative_prompt_animatediff_lcm.update(interactive=True)
    else:
        return sampler_animatediff_lcm.update(value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(value=10), guidance_scale_animatediff_lcm.update(), negative_prompt_animatediff_lcm.update(interactive=True)

## Functions specific to Stable Video Diffusion
def read_ini_img2vid(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), float(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), int(content[10]), int(content[11]), int(content[12]), int(content[13]), float(content[14]), bool(int(content[15])), float(content[16])

def change_model_type_img2vid(model_img2vid):
    if (model_img2vid == "stabilityai/stable-video-diffusion-img2vid"):
        return num_frames_img2vid.update(value=14)
    else:
        return num_frames_img2vid.update(value=25)

## Functions specific to Video Instruct-Pix2Pix
def read_ini_vid2vid_ze(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), float(content[4]), int(content[5]), int(content[6]), int(content[7]), int(content[8]), int(content[9]), int(content[10]), int(content[11]), bool(int(content[12])), float(content[13])

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

def read_ini_txt2shape(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7])

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

def read_ini_img2shape(module) :
    content = read_ini(module)
    return str(content[0]), int(content[1]), str(content[2]), float(content[3]), int(content[4]), int(content[5]), int(content[6]), int(content[7])

## Functions specific to Models cleaner
def refresh_models_cleaner_list():
    return gr.CheckboxGroup(choices=biniouModelsManager("./models").modelslister(), value=None, type="value", label="Installed models list", info="Select the models you want to delete and click \"Delete selected models\" button. Restart pixify to re-synchronize models list.")

## Functions specific to LoRA models manager
def refresh_lora_models_manager_list_sd():
    return gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SD").modelslister(), value=None, type="value", label="Installed models list", info="Select the LoRA models you want to delete and click \"Delete selected models\" button. Restart pixify to re-synchronize LoRA models list.")

def refresh_lora_models_manager_list_sdxl():
    return gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SDXL").modelslister(), value=None, type="value", label="Installed models list", info="Select the LoRA models you want to delete and click \"Delete selected models\" button. Restart pixify to re-synchronize LoRA models list.")

## Functions specific to Textual inversion manager
def refresh_textinv_manager_list_sd():
    return gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SD").modelslister(), value=None, type="value", label="Installed textual inversion list", info="Select the textual inversion you want to delete and click \"Delete selected textual inversion\" button. Restart pixify to re-synchronize textual inversion list.")

def refresh_textinv_manager_list_sdxl():
    return gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelslister(), value=None, type="value", label="Installed textual inversion list", info="Select the textual inversion you want to delete and click \"Delete selected textual inversion\" button. Restart pixify to re-synchronize textual inversion list.")

## Functions specific to Common settings
def biniou_global_settings_auth_switch(auth_value):
	if auth_value:
		return biniou_global_settings_auth_message.update(interactive=True), biniou_global_settings_share.update(interactive=True)
	else:
		return biniou_global_settings_auth_message.update(interactive=False), biniou_global_settings_share.update(value=False, interactive=False)

## Functions specific to console
def refresh_logfile():
    return logfile_biniou
        
def show_download_console() :
    return btn_download_file_console.update(visible=False), download_file_console.update(visible=True)

def hide_download_console() :
    return btn_download_file_console.update(visible=True), download_file_console.update(visible=False)

## Functions specific to banner 

def dict_to_url(url) :
    url_final = "./?"
    for key, value in url.items():
        url_final += "&" + key + "=" + value
    return url_final.replace("?&", "?")

def url_params_theme(url) :
    url = eval(url)
    if url.get('__theme') != None and url['__theme'] == "dark" :
        del url['__theme']
        url_final = dict_to_url(url)
        return f"<a href='https://216.230.232.229:7860' target='_blank' style='text-decoration: none;'><span style='text-align: left; font-size: 32px; font-weight: bold; line-height:32px; display: inline-flex; align-items: center;'>Pi<img src='file/images/pixify_64.png' width='32' height='32' style='margin: 0 2px;'/>ifyAI</span><span style='display: inline-flex; align-items: center;'><button onclick=\"window.location.href='{url_final}';\" title='Switch to dark mode and reload page' style='margin-left: 10px;'>💡</button></span></a>", banner_biniou.update(visible=True)
    elif url.get('__theme') == None :
        url['__theme'] = "light"
        url_final = dict_to_url(url)
        return f"<a href='https://216.230.232.229:7860' target='_blank' style='text-decoration: none;'><span style='text-align: left; font-size: 32px; font-weight: bold; line-height:32px; display: inline-flex; align-items: center;'>Pi<img src='file/images/pixify_64.png' width='32' height='32' style='margin: 0 2px;'/>ifyAI</span><span style='display: inline-flex; align-items: center;'><button onclick=\"window.location.href='{url_final}';\" title='Switch to light mode and reload page' style='margin-left: 10px;'>🕶️</button></span></a>", banner_biniou.update(visible=True)


color_label = "#1c1e23"
color_label_button = "#0b16b4"

theme_gradio = gr.themes.Base().set(
    background_fill_primary='*#181a1b',
    background_fill_primary_dark='#181a1b',
    block_label_background_fill=color_label,
    block_label_background_fill_dark=color_label,
    block_label_border_color='#000000',
    block_label_border_color_dark='#000000',
    block_label_text_color='silver',
    block_label_text_color_dark='white',
    block_title_background_fill=color_label,
    block_title_background_fill_dark=color_label,
    block_title_text_color='white',
    block_title_text_color_dark='white',
    block_title_padding='5px',
    block_title_radius='*radius_lg',
    button_primary_background_fill=color_label_button,
    button_primary_background_fill_dark=color_label_button,
    button_primary_border_color=color_label_button,
    button_primary_border_color_dark=color_label_button,
    button_primary_text_color='white',
    button_primary_text_color_dark='white',
)

with gr.Blocks(theme=theme_gradio, title="pixify-webui") as demo:
    nsfw_filter = gr.Textbox(value="1", visible=False)
    url_params_current = gr.Textbox(value="", visible=False)
    banner_biniou = gr.HTML("""""", visible=False)
    url_params_current.change(url_params_theme, url_params_current, [banner_biniou, banner_biniou], show_progress="hidden")
    with gr.Tabs() as tabs:
        # ...
        # tabs content here
# Chat
        with gr.TabItem("AI Chat", id=1) as tab_text:
            with gr.Tabs() as tabs_text:
# llamacpp
                with gr.TabItem("Chatbot Llama-cpp (gguf)", id=11) as tab_llamacpp:
                    with gr.Accordion("About", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Llama-cpp</br>
                                <b>Function : </b>Chat with an AI using <a href='https://github.com/abetlen/llama-cpp-python' target='_blank'>llama-cpp-python</a></br>
                                <b>Input(s) : </b>Input text</br>
                                <b>Output(s) : </b>Output text</br>
                                <b>HF models pages : </b>
                                                                <a href='https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF' target='_blank'>NousResearch/Meta-Llama-3-8B-Instruct-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF' target='_blank'>TheBloke/openchat-3.5-1210-GGUF</a>, 
                                <a href='https://huggingface.co/sayhan/gemma-7b-it-GGUF-quantized' target='_blank'>sayhan/gemma-7b-it-GGUF-quantized</a>, 
                                <a href='https://huggingface.co/mlabonne/gemma-2b-it-GGUF' target='_blank'>mlabonne/gemma-2b-it-GGUF</a>, 
                                <a href='https://huggingface.co/mlabonne/AlphaMonarch-7B-GGUF' target='_blank'>mlabonne/AlphaMonarch-7B-GGUF</a>, 
                                <a href='https://huggingface.co/mlabonne/NeuralBeagle14-7B-GGUF' target='_blank'>mlabonne/NeuralBeagle14-7B-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF' target='_blank'>TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF' target='_blank'>TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/phi-2-GGUF' target='_blank'>TheBloke/phi-2-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF' target='_blank'>TheBloke/Mixtral_7Bx2_MoE-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/mixtralnt-4x7b-test-GGUF' target='_blank'>TheBloke/mixtralnt-4x7b-test-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF' target='_blank'>TheBloke/Mistral-7B-Instruct-v0.2-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/MetaMath-Cybertron-Starling-GGUF' target='_blank'>TheBloke/MetaMath-Cybertron-Starling-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/una-cybertron-7B-v2-GGUF' target='_blank'>TheBloke/una-cybertron-7B-v2-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Starling-LM-7B-alpha-GGUF' target='_blank'>TheBloke/Starling-LM-7B-alpha-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/neural-chat-7B-v3-2-GGUF' target='_blank'>TheBloke/neural-chat-7B-v3-2-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF' target='_blank'>TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF' target='_blank'>TheBloke/zephyr-7B-beta-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-GGUF' target='_blank'>TheBloke/Yarn-Mistral-7B-128k-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF' target='_blank'>TheBloke/CodeLlama-13B-Instruct-GGUF</a></br>
                                """
                            )
#                                <a href='https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF' target='_blank'>TheBloke/Mistral-7B-v0.1-GGUF</a>, 
#                                <a href='https://huggingface.co/TheBloke/Vigogne-2-7B-Instruct-GGUF' target='_blank'>TheBloke/Vigogne-2-7B-Instruct-GGUF</a>, 
#                                <a href='https://huggingface.co/TheBloke/Vigogne-2-13B-Instruct-GGUF' target='_blank'>TheBloke/Vigogne-2-13B-Instruct-GGUF</a>, 
#                                <a href='https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF' target='_blank'>TheBloke/Airoboros-L2-7B-2.1-GGUF</a>, 
#                                <a href='https://huggingface.co/TheBloke/Airoboros-L2-13B-2.1-GGUF' target='_blank'>TheBloke/Airoboros-L2-13B-2.1-GGUF</a>, 
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Type your request in the <b>Input</b> textbox field</br>
                                - (optional) modify settings to use another model, change context size or modify maximum number of tokens generated.</br>
                                - Click the <b>Generate</b> button to generate a response to your input, using the chatbot history to keep a context.</br>
                                - Click the <b>Continue</b> button to complete the last reply.
                                </br>
                                <b>Models :</b></br>
                                - You could place llama-cpp compatible .gguf models in the directory ./biniou/models/llamacpp. Restart Pixify to see them in the models list.</br>
                                - You can also copy/paste in the <b>Model</b> dropdown menu a HF repo ID (e.g : TheBloke/some_model-GGUF) from <a href='https://huggingface.co/models?sort=trending&search=thebloke+gguf' target='_blank'>this list</a>. You must also set manually prompt and system templates according to the model page.
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_llamacpp = gr.Dropdown(choices=list(model_list_llamacpp.keys()), value=list(model_list_llamacpp.keys())[0], label="Model", allow_custom_value=True, info="Choose model to use for inference or copy/paste a HF repo id (TheBloke GGUF models only). Manually set prompt and system templates according to model page.")
                            with gr.Column():
                                max_tokens_llamacpp = gr.Slider(0, 131072, step=16, value=1024, label="Max tokens", info="Maximum number of tokens to generate")
                            with gr.Column():
                                seed_llamacpp = gr.Slider(0, 10000000000, step=1, value=1337, label="Seed(0 for random)", info="Seed to use for generation.")    
                        with gr.Row():
                            with gr.Column():
                                stream_llamacpp = gr.Checkbox(value=False, label="Stream", info="Stream results", interactive=False)                            
                            with gr.Column():
                                n_ctx_llamacpp = gr.Slider(0, 131072, step=128, value=8192, label="n_ctx", info="Maximum context size")
                            with gr.Column():
                                repeat_penalty_llamacpp = gr.Slider(0.0, 10.0, step=0.1, value=1.1, label="Repeat penalty", info="The penalty to apply to repeated tokens")
                        with gr.Row():
                            with gr.Column():
                                temperature_llamacpp = gr.Slider(0.0, 10.0, step=0.1, value=0.8, label="Temperature", info="Temperature to use for sampling")
                            with gr.Column():
                                top_p_llamacpp = gr.Slider(0.0, 10.0, step=0.05, value=0.95, label="top_p", info="The top-p value to use for sampling")
                            with gr.Column():
                                top_k_llamacpp = gr.Slider(0, 500, step=1, value=40, label="top_k", info="The top-k value to use for sampling")
                        with gr.Row():
                            with gr.Column():
                                force_prompt_template_llamacpp = gr.Dropdown(choices=list(prompt_template_list_llamacpp.keys()), value=list(prompt_template_list_llamacpp.keys())[0], label="Force prompt template", info="Choose prompt template to use for inference")
                            with gr.Column():
                                gr.Number(visible=False)
                            with gr.Column():
                                gr.Number(visible=False)
                        with gr.Row():
                            with gr.Column():
                                prompt_template_llamacpp = gr.Textbox(label="Prompt template", value=model_list_llamacpp[model_llamacpp.value][1], lines=4, max_lines=4, info="Place your custom prompt template here. Keep the {prompt} and {system} tags, they will be replaced by your prompt and system template.")
                        with gr.Row():
                            with gr.Column():
                                system_template_llamacpp = gr.Textbox(label="System template", value=model_list_llamacpp[model_llamacpp.value][2], lines=4, max_lines=4, info="Place your custom system template here.")
                                model_llamacpp.change(fn=change_model_type_llamacpp, inputs=model_llamacpp, outputs=[prompt_template_llamacpp, system_template_llamacpp])
                                force_prompt_template_llamacpp.change(fn=change_prompt_template_llamacpp, inputs=force_prompt_template_llamacpp, outputs=[prompt_template_llamacpp, system_template_llamacpp])
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_llamacpp = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_llamacpp = gr.Textbox(value="llamacpp", visible=False, interactive=False)
                                del_ini_btn_llamacpp = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_llamacpp.value) else False)
                                save_ini_btn_llamacpp.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_llamacpp, 
                                        model_llamacpp, 
                                        max_tokens_llamacpp, 
                                        seed_llamacpp, 
                                        stream_llamacpp, 
                                        n_ctx_llamacpp, 
                                        repeat_penalty_llamacpp, 
                                        temperature_llamacpp, 
                                        top_p_llamacpp, 
                                        top_k_llamacpp, 
                                        prompt_template_llamacpp,
                                        system_template_llamacpp,
                                        ]
                                    )
                                save_ini_btn_llamacpp.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_llamacpp.click(fn=lambda: del_ini_btn_llamacpp.update(interactive=True), outputs=del_ini_btn_llamacpp)
                                del_ini_btn_llamacpp.click(fn=lambda: del_ini(module_name_llamacpp.value))
                                del_ini_btn_llamacpp.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_llamacpp.click(fn=lambda: del_ini_btn_llamacpp.update(interactive=False), outputs=del_ini_btn_llamacpp)
                        if test_cfg_exist(module_name_llamacpp.value) :
                            readcfg_llamacpp = read_ini_llamacpp(module_name_llamacpp.value) 
                            model_llamacpp.value = readcfg_llamacpp[0] 
                            max_tokens_llamacpp.value = readcfg_llamacpp[1]
                            seed_llamacpp.value = readcfg_llamacpp[2]  
                            stream_llamacpp.value = readcfg_llamacpp[3] 
                            n_ctx_llamacpp.value = readcfg_llamacpp[4]  
                            repeat_penalty_llamacpp.value = readcfg_llamacpp[5] 
                            temperature_llamacpp.value = readcfg_llamacpp[6] 
                            top_p_llamacpp.value = readcfg_llamacpp[7]  
                            top_k_llamacpp.value = readcfg_llamacpp[8] 
                            prompt_template_llamacpp.value = readcfg_llamacpp[9] 
#                           To remove : dirty temporary workaround
                            if len(readcfg_llamacpp)>10:
                                system_template_llamacpp.value = readcfg_llamacpp[10]
                    with gr.Row():
                        history_llamacpp = gr.Chatbot(
                            label="Chatbot history", 
                            height=400,
                            autoscroll=True, 
                            show_copy_button=True, 
                            interactive=True,
                            bubble_full_width = False,
                            avatar_images = ("./images/user_64.png", "./images/pixify_64.png"),
                        )
                        last_reply_llamacpp = gr.Textbox(value="", visible=False)                        
                    with gr.Row():
                            prompt_llamacpp = gr.Textbox(label="Input", lines=1, max_lines=3, placeholder="Type your request here ...", autofocus=True)
                            hidden_prompt_llamacpp = gr.Textbox(value="", visible=False)
                    with gr.Row():
                        with gr.Column():
                            btn_llamacpp = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_llamacpp_continue = gr.Button("Continue ➕")
                        with gr.Column():                      
                            btn_llamacpp_clear_output = gr.ClearButton(components=[history_llamacpp], value="Clear outputs 🧹") 
                        with gr.Column():
                            btn_download_file_llamacpp = gr.ClearButton(value="Download full conversation 💾", visible=True) 
                            download_file_llamacpp = gr.File(label="Download full conversation", value=blankfile_common, height=30, interactive=False, visible=False)
                            download_file_llamacpp_hidden = gr.Textbox(value=blankfile_common, interactive=False, visible=False)
                            btn_download_file_llamacpp.click(fn=show_download_llamacpp, outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                            download_file_llamacpp_hidden.change(fn=lambda x:x, inputs=download_file_llamacpp_hidden, outputs=download_file_llamacpp)
                        btn_llamacpp.click(
                            fn=text_llamacpp,
                            inputs=[
                                model_llamacpp, 
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
                        btn_llamacpp.click(fn=lambda x:x, inputs=hidden_prompt_llamacpp, outputs=prompt_llamacpp)
                        prompt_llamacpp.submit(fn=lambda x:x, inputs=hidden_prompt_llamacpp, outputs=prompt_llamacpp)
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... last chatbot reply to ...')
                                        gr.HTML(value='... text module ...')
                                        llamacpp_nllb = gr.Button(" >> Nllb translation")
                                        gr.HTML(value='... image module ...')                                        
                                        llamacpp_txt2img_sd = gr.Button(" >> Stable Diffusion")
                                        llamacpp_txt2img_kd = gr.Button(" >> Kandinsky") 
                                        llamacpp_txt2img_lcm = gr.Button(" >> LCM") 
                                        llamacpp_txt2img_mjm = gr.Button(" >> Midjourney-mini") 
                                        llamacpp_txt2img_paa = gr.Button(" >> PixArt-Alpha") 
                                        llamacpp_img2img = gr.Button(" >> img2img")
                                        llamacpp_img2img_ip = gr.Button(" >>  IP-Adapter")
                                        llamacpp_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        llamacpp_inpaint = gr.Button(" >> inpaint")
                                        llamacpp_controlnet = gr.Button(" >> ControlNet")
                                        llamacpp_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... audio module ...')
                                        llamacpp_musicgen = gr.Button(" >> Musicgen")                                        
                                        llamacpp_audiogen = gr.Button(" >> Audiogen")
                                        llamacpp_bark = gr.Button(" >> Bark")                                        
                                        gr.HTML(value='... video module ...')                                               
                                        llamacpp_txt2vid_ms = gr.Button(" >> Modelscope")
                                        llamacpp_txt2vid_ze = gr.Button(" >> Text2Video-Zero")
                                        llamacpp_animatediff_lcm = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# llava
                with gr.TabItem("Llava (gguf)", id=12) as tab_llava:
                    with gr.Accordion("About", open=False):
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Llava (gguf)</br>
                                <b>Function : </b>Interrogate a chatbot about an input image using <a href='https://github.com/abetlen/llama-cpp-python' target='_blank'>llama-cpp-python</a>, <a href='https://llava-vl.github.io/' target='_blank'>Llava</a> and <a href='https://github.com/SkunkworksAI/BakLLaVA' target='_blank'>BakLLaVA</a></br>
                                <b>Input(s) : </b>Input image, Input text</br>
                                <b>Output(s) : </b>Output text</br>
                                <b>HF models pages : </b>
                                <a href='https://huggingface.co/mys/ggml_bakllava-1' target='_blank'>mys/ggml_bakllava-1</a>, 
                                <a href='https://huggingface.co/cmp-nct/llava-1.6-gguf' target='_blank'>cmp-nct/llava-1.6-gguf</a>, 
                                <a href='https://huggingface.co/mys/ggml_llava-v1.5-7b' target='_blank'>mys/ggml_llava-v1.5-7b</a>, 
                                <a href='https://huggingface.co/mys/ggml_llava-v1.5-13b' target='_blank'>mys/ggml_llava-v1.5-13b</a>
                           </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an <b>Input image</b></br>
                                - Type your request in the <b>Input</b> textbox field</br>
                                - (optional) modify settings to use another model, change context size or modify maximum number of tokens generated.</br>
                                - Click the <b>Generate</b> button to generate a response to your input, using the chatbot history to keep a context.</br>
                                - Click the <b>Continue</b> button to complete the last reply.
                                </br>
                                <b>Models :</b></br>
                                - You could place llama-cpp compatible .gguf models in the directory ./biniou/models/llava. Restart Pixify to see them in the models list.
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_llava = gr.Dropdown(choices=model_list_llava, value=model_list_llava[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                max_tokens_llava = gr.Slider(0, 131072, step=16, value=512, label="Max tokens", info="Maximum number of tokens to generate")
                            with gr.Column():
                                seed_llava = gr.Slider(0, 10000000000, step=1, value=1337, label="Seed(0 for random)", info="Seed to use for generation.")
                        with gr.Row():
                            with gr.Column():
                                stream_llava = gr.Checkbox(value=False, label="Stream", info="Stream results", interactive=False)
                            with gr.Column():
                                n_ctx_llava = gr.Slider(0, 131072, step=128, value=8192, label="n_ctx", info="Maximum context size")
                            with gr.Column():
                                repeat_penalty_llava = gr.Slider(0.0, 10.0, step=0.1, value=1.1, label="Repeat penalty", info="The penalty to apply to repeated tokens")
                        with gr.Row():
                            with gr.Column():
                                temperature_llava = gr.Slider(0.0, 10.0, step=0.1, value=0.8, label="Temperature", info="Temperature to use for sampling")
                            with gr.Column():
                                top_p_llava = gr.Slider(0.0, 10.0, step=0.05, value=0.95, label="top_p", info="The top-p value to use for sampling")
                            with gr.Column():
                                top_k_llava = gr.Slider(0, 500, step=1, value=40, label="top_k", info="The top-k value to use for sampling")
                        with gr.Row():
                            with gr.Column():
                                prompt_template_llava = gr.Textbox(label="Prompt template", value="{prompt}", lines=4, max_lines=4, info="Place your custom prompt template here. Keep the {prompt} tag, that will be replaced by your prompt.")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_llava = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_llava = gr.Textbox(value="llava", visible=False, interactive=False)
                                del_ini_btn_llava = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_llava.value) else False)
                                save_ini_btn_llava.click(
                                    fn=write_ini,
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
                                        ]
                                    )
                                save_ini_btn_llava.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_llava.click(fn=lambda: del_ini_btn_llava.update(interactive=True), outputs=del_ini_btn_llava)
                                del_ini_btn_llava.click(fn=lambda: del_ini(module_name_llava.value))
                                del_ini_btn_llava.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_llava.click(fn=lambda: del_ini_btn_llava.update(interactive=False), outputs=del_ini_btn_llava)
                        if test_cfg_exist(module_name_llava.value) :
                            readcfg_llava = read_ini_llava(module_name_llava.value)
                            model_llava.value = readcfg_llava[0]
                            max_tokens_llava.value = readcfg_llava[1]
                            seed_llava.value = readcfg_llava[2]
                            stream_llava.value = readcfg_llava[3]
                            n_ctx_llava.value = readcfg_llava[4]
                            repeat_penalty_llava.value = readcfg_llava[5]
                            temperature_llava.value = readcfg_llava[6]
                            top_p_llava.value = readcfg_llava[7]
                            top_k_llava.value = readcfg_llava[8]
                            prompt_template_llava.value = readcfg_llava[9]
                    with gr.Row():
                        with gr.Column(scale=1):
                            img_llava = gr.Image(label="Input image", type="filepath", height=400)
                        with gr.Column(scale=3):
                            history_llava = gr.Chatbot(
                                label="Chatbot history",
                                height=400,
                                autoscroll=True,
                                show_copy_button=True,
                                interactive=True,
                                bubble_full_width = False,
                                avatar_images = ("./images/user_64.png", "./images/pixify_64.png"),
                            )
                            last_reply_llava = gr.Textbox(value="", visible=False)
                    with gr.Row():
                            prompt_llava = gr.Textbox(label="Input", lines=1, max_lines=3, placeholder="Type your request here ...", autofocus=True)
                            hidden_prompt_llava = gr.Textbox(value="", visible=False)
                    with gr.Row():
                        with gr.Column():
                            btn_llava = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_llava_clear_input = gr.ClearButton(components=[img_llava, prompt_llava], value="Clear inputs 🧹")
                            btn_llava_continue = gr.Button("Continue ➕", visible=False)
                        with gr.Column():
                            btn_llava_clear_output = gr.ClearButton(components=[history_llava], value="Clear outputs 🧹")
                        with gr.Column():
                            btn_download_file_llava = gr.ClearButton(value="Download full conversation 💾", visible=True)
                            download_file_llava = gr.File(label="Download full conversation", value=blankfile_common, height=30, interactive=False, visible=False)
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
                        btn_llava.click(fn=lambda x:x, inputs=hidden_prompt_llava, outputs=prompt_llava)
                        prompt_llava.submit(fn=lambda x:x, inputs=hidden_prompt_llava, outputs=prompt_llava)
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... last chatbot reply to ...')
                                        gr.HTML(value='... text module ...')
                                        llava_nllb = gr.Button(" >> Nllb translation")
                                        gr.HTML(value='... image module ...')
                                        llava_txt2img_sd = gr.Button(" >> Stable Diffusion")
                                        llava_txt2img_kd = gr.Button(" >> Kandinsky") 
                                        llava_txt2img_lcm = gr.Button(" >> LCM") 
                                        llava_txt2img_mjm = gr.Button(" >> Midjourney-mini") 
                                        llava_txt2img_paa = gr.Button(" >> PixArt-Alpha") 
                                        llava_img2img = gr.Button(" >> img2img")
                                        llava_img2img_ip = gr.Button(" >>  IP-Adapter")
                                        llava_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        llava_inpaint = gr.Button(" >> inpaint")
                                        llava_controlnet = gr.Button(" >> ControlNet")
                                        llava_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... audio module ...')
                                        llava_musicgen = gr.Button(" >> Musicgen")
                                        llava_audiogen = gr.Button(" >> Audiogen")
                                        llava_bark = gr.Button(" >> Bark")
                                        gr.HTML(value='... video module ...')
                                        llava_txt2vid_ms = gr.Button(" >> Modelscope")
                                        llava_txt2vid_ze = gr.Button(" >> Text2Video-Zero")
                                        llava_animatediff_lcm = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
# Image captioning
                with gr.TabItem("Image captioning", id=13) as tab_img2txt_git:
                    with gr.Accordion("About", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Image Captioning</br>
                                <b>Function : </b>Caption an image by a simple description of it using GIT</br>
                                <b>Input(s) : </b>Input image</br>
                                <b>Output(s) : </b>Caption text</br>
                                <b>HF model page : </b><a href='https://huggingface.co/microsoft/git-large-coco' target='_blank'>microsoft/git-large-coco</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload an input image by clicking on the <b>Input image</b> field</br>
                                - (optional) modify settings to use another model, change min. and/or max. number of tokens generated.</br>
                                - Click the <b>Generate button</b></br>
                                - After generation, captions of the image are displayed in the Generated captions field                                 
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2txt_git = gr.Dropdown(choices=model_list_img2txt_git, value=model_list_img2txt_git[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                min_tokens_img2txt_git = gr.Slider(0, 128, step=1, value=0, label="Min tokens", info="Minimum number of tokens in output")
                            with gr.Column():
                                max_tokens_img2txt_git = gr.Slider(0, 256, step=1, value=20, label="Max tokens", info="Maximum number of tokens in output")
                        with gr.Row():
                            with gr.Column():
                                num_beams_img2txt_git = gr.Slider(1, 16, step=1, value=1, label="Num beams", info="Number of total beams")
                            with gr.Column():
                                num_beam_groups_img2txt_git = gr.Slider(1, 8, step=1, value=1, label="Num beam groups", info="Number of beam groups")
                                num_beams_img2txt_git.change(set_num_beam_groups_img2txt_git, inputs=[num_beams_img2txt_git, num_beam_groups_img2txt_git], outputs=num_beam_groups_img2txt_git)
                            with gr.Column():
                                diversity_penalty_img2txt_git = gr.Slider(0.0, 5.0, step=0.01, value=0.5, label="Diversity penalty", info="Penalty score value for a beam")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2txt_git = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_img2txt_git = gr.Textbox(value="img2txt_git", visible=False, interactive=False)
                                del_ini_btn_img2txt_git = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_img2txt_git.value) else False)
                                save_ini_btn_img2txt_git.click(
                                    fn=write_ini, 
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
                                save_ini_btn_img2txt_git.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_img2txt_git.click(fn=lambda: del_ini_btn_img2txt_git.update(interactive=True), outputs=del_ini_btn_img2txt_git)
                                del_ini_btn_img2txt_git.click(fn=lambda: del_ini(module_name_img2txt_git.value))
                                del_ini_btn_img2txt_git.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_img2txt_git.click(fn=lambda: del_ini_btn_img2txt_git.update(interactive=False), outputs=del_ini_btn_img2txt_git)
                        if test_cfg_exist(module_name_img2txt_git.value) :
                            readcfg_img2txt_git = read_ini_img2txt_git(module_name_img2txt_git.value)
                            model_img2txt_git.value = readcfg_img2txt_git[0]
                            min_tokens_img2txt_git.value = readcfg_img2txt_git[1]
                            max_tokens_img2txt_git.value = readcfg_img2txt_git[2]
                            num_beams_img2txt_git.value = readcfg_img2txt_git[3]
                            num_beam_groups_img2txt_git.value = readcfg_img2txt_git[4]
                            diversity_penalty_img2txt_git.value = readcfg_img2txt_git[5]
                    with gr.Row():
                        with gr.Column():
                            img_img2txt_git = gr.Image(label="Input image", type="pil", height=400)
                        with gr.Column():
                            out_img2txt_git = gr.Textbox(label="Generated captions", lines=15, show_copy_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_img2txt_git = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_img2txt_git_clear_input = gr.ClearButton(components=[img_img2txt_git], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_img2txt_git_clear_output = gr.ClearButton(components=[out_img2txt_git], value="Clear outputs 🧹") 
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        img2txt_git_nllb = gr.Button(" >> Nllb translation")
                                        gr.HTML(value='... image module ...')                                        
                                        img2txt_git_txt2img_sd = gr.Button(" >> Stable Diffusion")
                                        img2txt_git_txt2img_kd = gr.Button(" >> Kandinsky")                                        
                                        img2txt_git_txt2img_lcm = gr.Button(" >> LCM") 
                                        img2txt_git_txt2img_mjm = gr.Button(" >> Midjourney-mini") 
                                        img2txt_git_txt2img_paa = gr.Button(" >> PixArt-Alpha") 
                                        img2txt_git_img2img = gr.Button(" >> img2img")
                                        img2txt_git_img2img_ip = gr.Button(" >> IP-Adapter")
                                        img2txt_git_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        img2txt_git_inpaint = gr.Button(" >> inpaint")
                                        img2txt_git_controlnet = gr.Button(" >> ControlNet")                                        
                                        img2txt_git_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... audio module ...')
                                        img2txt_git_musicgen = gr.Button(" >> Musicgen")                                        
                                        img2txt_git_audiogen = gr.Button(" >> Audiogen")
                                        gr.HTML(value='... video module ...')                                               
                                        img2txt_git_txt2vid_ms = gr.Button(" >> Modelscope")
                                        img2txt_git_txt2vid_ze = gr.Button(" >> Text2Video-Zero")
                                        img2txt_git_animatediff_lcm = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        img2txt_git_img2img_both = gr.Button("+ >> img2img")
                                        img2txt_git_img2img_ip_both = gr.Button("+ >> IP-Adapter")
                                        img2txt_git_pix2pix_both = gr.Button("+ >> Instruct pix2pix")
                                        img2txt_git_inpaint_both = gr.Button("+ >> inpaint")
                                        img2txt_git_controlnet_both = gr.Button("+ >> ControlNet")
                                        img2txt_git_faceid_ip_both = gr.Button("+ >> IP-Adapter FaceID")

# Whisper 
                with gr.TabItem("Whisper", id=14) as tab_whisper:
                    with gr.Accordion("About", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Whisper</br>
                                <b>Function : </b>Transcribe/translate audio to text with <a href='https://openai.com/research/whisper' target='_blank'>whisper</a></br>
                                <b>Input(s) : </b>Input audio</br>
                                <b>Output(s) : </b>Transcribed/translated text</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/openai/whisper-tiny' target='_blank'>openai/whisper-tiny</a>, 
                                <a href='https://huggingface.co/openai/whisper-base' target='_blank'>openai/whisper-base</a>, 
                                <a href='https://huggingface.co/openai/whisper-medium' target='_blank'>openai/whisper-medium</a>,
                                <a href='https://huggingface.co/openai/whisper-large' target='_blank'>openai/whisper-large</a>,
                                <a href='https://huggingface.co/openai/whisper-large-v3' target='_blank'>openai/whisper-large-v3</a>,
                                <a href='https://huggingface.co/distil-whisper/distil-large-v2' target='_blank'>distil-whisper/distil-large-v2</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload an input audio file by clicking on the <b>Source audio</b> field or select the <b>micro</b> input type and record your voice</br>
                                - Select the source language of the audio</br>
                                - Select the task to execute : transcribe in source language or translate to english</br>
                                - (optional) modify settings to use another model, or generate SRT-formated subtitles</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, audio transcription is displayed in the <b>Output text</b> field
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_whisper = gr.Dropdown(choices=list(model_list_whisper.keys()), value=list(model_list_whisper.keys())[4], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                srt_output_whisper = gr.Checkbox(value=False, label=".srt format output", info="Generate an output in .srt format")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_whisper = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_whisper = gr.Textbox(value="whisper", visible=False, interactive=False)
                                del_ini_btn_whisper = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_whisper.value) else False)
                                save_ini_btn_whisper.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_whisper, 
                                        model_whisper, 
                                        srt_output_whisper,
                                        ]
                                    )
                                save_ini_btn_whisper.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_whisper.click(fn=lambda: del_ini_btn_whisper.update(interactive=True), outputs=del_ini_btn_whisper)
                                del_ini_btn_whisper.click(fn=lambda: del_ini(module_name_whisper.value))
                                del_ini_btn_whisper.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_whisper.click(fn=lambda: del_ini_btn_whisper.update(interactive=False), outputs=del_ini_btn_whisper)
                        if test_cfg_exist(module_name_whisper.value) :
                            readcfg_whisper = read_ini_whisper(module_name_whisper.value)
                            model_whisper.value = readcfg_whisper[0]
                            srt_output_whisper.value = readcfg_whisper[1]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    source_type_whisper = gr.Radio(choices=["audio", "micro"], value="audio", label="Input type", info="Choose input type")
                                with gr.Column():
                                    source_language_whisper = gr.Dropdown(choices=language_list_whisper, value=language_list_whisper[14], label="Input language", info="Select input language")    
                            with gr.Row():
                                source_audio_whisper = gr.Audio(label="Source audio", source="upload", type="filepath")
                                source_type_whisper.change(fn=change_source_type_whisper, inputs=source_type_whisper, outputs=source_audio_whisper)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    output_type_whisper = gr.Radio(choices=["transcribe", "translate"], value="transcribe", label="Task", info="Choose task to execute")
                                with gr.Column():
                                    output_language_whisper = gr.Dropdown(choices=language_list_whisper, value=language_list_whisper[14], label="Output language", info="Select output language", visible=False, interactive=False)
                            with gr.Row():
                                out_whisper = gr.Textbox(label="Output text", lines=9, max_lines=9, show_copy_button=True, interactive=False)
                                output_type_whisper.change(fn=change_output_type_whisper, inputs=output_type_whisper, outputs=output_language_whisper)
                    with gr.Row():
                        with gr.Column():
                            btn_whisper = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_whisper_clear_input = gr.ClearButton(components=[source_audio_whisper], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_whisper_clear_output = gr.ClearButton(components=[out_whisper], value="Clear outputs 🧹") 
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... output text to ...')
                                        gr.HTML(value='... text module ...')
                                        whisper_nllb = gr.Button(" >> Nllb translation")
                                        gr.HTML(value='... image module ...')
                                        whisper_txt2img_sd = gr.Button(" >> Stable Diffusion")
                                        whisper_txt2img_kd = gr.Button(" >> Kandinsky")
                                        whisper_txt2img_lcm = gr.Button(" >> LCM") 
                                        whisper_txt2img_mjm = gr.Button(" >> Midjourney-mini") 
                                        whisper_txt2img_paa = gr.Button(" >> PixArt-Alpha") 
                                        whisper_img2img = gr.Button(" >> img2img")
                                        whisper_img2img_ip = gr.Button(" >> IP-Adapter")
                                        whisper_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        whisper_inpaint = gr.Button(" >> inpaint")
                                        whisper_controlnet = gr.Button(" >> ControlNet")
                                        whisper_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... audio module ...')
                                        whisper_musicgen = gr.Button(" >> Musicgen")
                                        whisper_audiogen = gr.Button(" >> Audiogen")
                                        whisper_bark = gr.Button(" >> Bark")
                                        gr.HTML(value='... video module ...')
                                        whisper_txt2vid_ms = gr.Button(" >> Modelscope")
                                        whisper_txt2vid_ze = gr.Button(" >> Text2Video-Zero")
                                        whisper_animatediff_lcm = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# nllb 
                with gr.TabItem("nllb translation", id=15) as tab_nllb:
                    with gr.Accordion("About", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>nllb translation</br>
                                <b>Function : </b>Translate text with <a href='https://ai.meta.com/research/no-language-left-behind/' target='_blank'>nllb</a></br>
                                <b>Input(s) : </b>Input text</br>
                                <b>Output(s) : </b>Translated text</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/facebook/nllb-200-distilled-600M' target='_blank'>facebook/nllb-200-distilled-600M</a>
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Select an <b>input language</b></br>
                                - Type or copy/paste the text to translate in the <b>source text</b> field</br>
                                - Select an <b>output language</b></br>
                                - (optional) modify settings to use another model, or reduce the maximum number of tokens in the output</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, translation is displayed in the <b>Output text</b> field
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_nllb = gr.Dropdown(choices=model_list_nllb, value=model_list_nllb[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                max_tokens_nllb = gr.Slider(0, 1024, step=1, value=1024, label="Max tokens", info="Maximum number of tokens in output")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_nllb = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_nllb = gr.Textbox(value="nllb", visible=False, interactive=False)
                                del_ini_btn_nllb = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_nllb.value) else False)
                                save_ini_btn_nllb.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_nllb, 
                                        model_nllb, 
                                        max_tokens_nllb,
                                        ]
                                    )
                                save_ini_btn_nllb.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_nllb.click(fn=lambda: del_ini_btn_nllb.update(interactive=True), outputs=del_ini_btn_nllb)
                                del_ini_btn_nllb.click(fn=lambda: del_ini(module_name_nllb.value))
                                del_ini_btn_nllb.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_nllb.click(fn=lambda: del_ini_btn_nllb.update(interactive=False), outputs=del_ini_btn_nllb)
                        if test_cfg_exist(module_name_nllb.value) :
                            readcfg_nllb = read_ini_nllb(module_name_nllb.value)
                            model_nllb.value = readcfg_nllb[0]
                            max_tokens_nllb.value = readcfg_nllb[1]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                source_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()), value=list(language_list_nllb.keys())[47], label="Input language", info="Select input language")    
                            with gr.Row():
                                prompt_nllb = gr.Textbox(label="Source text", lines=9, max_lines=9, placeholder="Type or paste here the text to translate")
                        with gr.Column():
                            with gr.Row():
                                output_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()), value=list(language_list_nllb.keys())[47], label="Output language", info="Select output language")
                            with gr.Row():
                                out_nllb = gr.Textbox(label="Output text", lines=9, max_lines=9, show_copy_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_nllb = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_nllb_clear_input = gr.ClearButton(components=[prompt_nllb], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_nllb_clear_output = gr.ClearButton(components=[out_nllb], value="Clear outputs 🧹") 
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... output text to ...')
                                        nllb_llamacpp = gr.Button(" >> Chatbot Llama-cpp")
                                        gr.HTML(value='... image module ...')                                        
                                        nllb_txt2img_sd = gr.Button(" >> Stable Diffusion")
                                        nllb_txt2img_kd = gr.Button(" >> Kandinsky")                                        
                                        nllb_txt2img_lcm = gr.Button(" >> LCM") 
                                        nllb_txt2img_mjm = gr.Button(" >> Midjourney-mini") 
                                        nllb_txt2img_paa = gr.Button(" >> PixArt-Alpha") 
                                        nllb_img2img = gr.Button(" >> img2img")
                                        nllb_img2img_ip = gr.Button(" >> IP-Adapter")
                                        nllb_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        nllb_inpaint = gr.Button(" >> inpaint")
                                        nllb_controlnet = gr.Button(" >> ControlNet")                                        
                                        nllb_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... audio module ...')
                                        nllb_musicgen = gr.Button(" >> Musicgen")                                        
                                        nllb_audiogen = gr.Button(" >> Audiogen")
                                        nllb_bark = gr.Button(" >> Bark")                                        
                                        gr.HTML(value='... video module ...')                                               
                                        nllb_txt2vid_ms = gr.Button(" >> Modelscope")
                                        nllb_txt2vid_ze = gr.Button(" >> Text2Video-Zero")                                        
                                        nllb_animatediff_lcm = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# txt2prompt
                if ram_size() >= 16 :
                    titletab_txt2prompt = "Prompt generator 📝"
                else :
                    titletab_txt2prompt = "Prompt generator ⛔"

                with gr.TabItem(titletab_txt2prompt, id=16) as tab_txt2prompt:
                    with gr.Accordion("About", open=False):
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Prompt generator</br>
                                <b>Function : </b>Create complex prompt from a simple instruction.</br>
                                <b>Input(s) : </b>Prompt</br>
                                <b>Output(s) : </b>Enhanced output prompt</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/PulsarAI/prompt-generator' target='_blank'>PulsarAI/prompt-generator</a>, 
                                <a href='https://huggingface.co/RamAnanth1/distilgpt2-sd-prompts' target='_blank'>RamAnanth1/distilgpt2-sd-prompts</a>, 
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Define a <b>prompt</b></br>
                                - Choose the type of output to produce : ChatGPT will produce a persona for the chatbot from your input, SD will generate a prompt usable for image and video modules</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, output is displayed in the <b>Output text</b> field. Send them to the desired module (chatbot or media modules).
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2prompt = gr.Dropdown(choices=model_list_txt2prompt, value=model_list_txt2prompt[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                max_tokens_txt2prompt = gr.Slider(0, 2048, step=1, value=128, label="Max tokens", info="Maximum number of tokens in output")
                            with gr.Column():
                                repetition_penalty_txt2prompt = gr.Slider(0.0, 10.0, step=0.01, value=1.05, label="Repetition penalty", info="The penalty to apply to repeated tokens")
                        with gr.Row():                                
                            with gr.Column():
                                seed_txt2prompt = gr.Slider(0, 4294967295, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Permit reproducibility") 
                            with gr.Column():
                                num_prompt_txt2prompt = gr.Slider(1, 64, step=1, value=1, label="Batch size", info="Number of prompts to generate") 
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2prompt = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2prompt = gr.Textbox(value="txt2prompt", visible=False, interactive=False)
                                del_ini_btn_txt2prompt = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2prompt.value) else False)
                                save_ini_btn_txt2prompt.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_txt2prompt, 
                                        model_txt2prompt, 
                                        max_tokens_txt2prompt,
                                        repetition_penalty_txt2prompt,
                                        seed_txt2prompt,
                                        num_prompt_txt2prompt,
                                        ]
                                    )
                                save_ini_btn_txt2prompt.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2prompt.click(fn=lambda: del_ini_btn_txt2prompt.update(interactive=True), outputs=del_ini_btn_txt2prompt)
                                del_ini_btn_txt2prompt.click(fn=lambda: del_ini(module_name_txt2prompt.value))
                                del_ini_btn_txt2prompt.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2prompt.click(fn=lambda: del_ini_btn_txt2prompt.update(interactive=False), outputs=del_ini_btn_txt2prompt)
                        if test_cfg_exist(module_name_txt2prompt.value) :
                            readcfg_txt2prompt = read_ini_txt2prompt(module_name_txt2prompt.value)
                            model_txt2prompt.value = readcfg_txt2prompt[0]
                            max_tokens_txt2prompt.value = readcfg_txt2prompt[1]
                            repetition_penalty_txt2prompt.value = readcfg_txt2prompt[2]
                            seed_txt2prompt.value = readcfg_txt2prompt[3]
                            num_prompt_txt2prompt.value = readcfg_txt2prompt[4]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                prompt_txt2prompt = gr.Textbox(label="Prompt", lines=9, max_lines=9, placeholder="a doctor")
                            with gr.Row():
                                output_type_txt2prompt = gr.Radio(choices=["ChatGPT", "SD"], value="ChatGPT", label="Output type", info="Choose type of prompt to generate")
                                output_type_txt2prompt.change(fn=change_output_type_txt2prompt, inputs=output_type_txt2prompt, outputs=[model_txt2prompt, max_tokens_txt2prompt])
                        with gr.Column():
                            with gr.Row():
                                out_txt2prompt = gr.Textbox(label="Output prompt", lines=16, max_lines=16, show_copy_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_txt2prompt = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_txt2prompt_clear_input = gr.ClearButton(components=[prompt_txt2prompt], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2prompt_clear_output = gr.ClearButton(components=[out_txt2prompt], value="Clear outputs 🧹") 
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... output text to ...')
                                        gr.HTML(value='... text module ...') 
                                        txt2prompt_nllb = gr.Button(" >> Nllb translation")
                                        txt2prompt_llamacpp = gr.Button(" >> Chatbot llama-cpp")
                                        gr.HTML(value='... image module ...')                                        
                                        txt2prompt_txt2img_sd = gr.Button(" >> Stable Diffusion")
                                        txt2prompt_txt2img_kd = gr.Button(" >> Kandinsky")                                        
                                        txt2prompt_txt2img_lcm = gr.Button(" >> LCM") 
                                        txt2prompt_txt2img_mjm = gr.Button(" >> Midjourney-mini") 
                                        txt2prompt_txt2img_paa = gr.Button(" >> PixArt-Alpha") 
                                        txt2prompt_img2img = gr.Button(" >> img2img")
                                        txt2prompt_img2img_ip = gr.Button(" >> IP-Adapter")
                                        txt2prompt_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        txt2prompt_inpaint = gr.Button(" >> inpaint")
                                        txt2prompt_controlnet = gr.Button(" >> ControlNet")                                        
                                        txt2prompt_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                               
                                        txt2prompt_txt2vid_ms = gr.Button(" >> Modelscope")
                                        txt2prompt_txt2vid_ze = gr.Button(" >> Text2Video-Zero")                                        
                                        txt2prompt_animatediff_lcm = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                        

# Image
        with gr.TabItem("Img Gen ", id=2) as tab_image:
            with gr.Tabs() as tabs_image:
# Stable Diffusion
                with gr.TabItem("Stable Diffusion ", id=21) as tab_txt2img_sd:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Stable diffusion</br>
                                <b>Function : </b>Generate images from a prompt and a negative prompt using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/stabilityai/sd-turbo' target='_blank'>stabilityai/sd-turbo</a>, 
                                <a href='https://huggingface.co/stabilityai/sdxl-turbo' target='_blank'>stabilityai/sdxl-turbo</a>, 
                                <a href='https://huggingface.co/thibaud/sdxl_dpo_turbo' target='_blank'>thibaud/sdxl_dpo_turbo</a>, 
                                <a href='https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B' target='_blank'>IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B</a>, 
                                <a href='https://huggingface.co/dataautogpt3/OpenDalleV1.1' target='_blank'>dataautogpt3/OpenDalleV1.1</a>, 
                                <a href='https://huggingface.co/dataautogpt3/ProteusV0.4' target='_blank'>dataautogpt3/ProteusV0.4</a>, 
                                <a href='https://huggingface.co/etri-vilab/koala-1b' target='_blank'>etri-vilab/koala-1b</a>, 
                                <a href='https://huggingface.co/etri-vilab/koala-700m' target='_blank'>etri-vilab/koala-700m</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/segmind/Segmind-Vega' target='_blank'>segmind/Segmind-Vega</a>, 
                                <a href='https://huggingface.co/segmind/SSD-1B' target='_blank'>segmind/SSD-1B</a>, 
                                <a href='https://huggingface.co/gsdf/Counterfeit-V2.5' target='_blank'>gsdf/Counterfeit-V2.5</a>, 
                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0' target='_blank'>stabilityai/stable-diffusion-xl-base-1.0</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a>, 
                                """
#                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model, generate several images in a single run or change dimensions of the outputs</br>
                                - (optional) Select a LoRA model and set its weight</br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory ./biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.</br>
                                <b>LoRA models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors LoRA models in the directory ./biniou/models/lora/SD or ./biniou/models/lora/SDXL (depending on the LoRA model type : SD 1.5 or SDXL). Restart Pixify to see them in the models list.
                                </div>                                
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_sd = gr.Dropdown(choices=model_list_txt2img_sd, value=model_list_txt2img_sd[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2img_sd = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_sd = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_sd = gr.Slider(0.0, 20.0, step=0.1, value=7.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2img_sd = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_sd = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_sd = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_sd = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_sd = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")    
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_sd = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_txt2img_sd = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_sd = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2img_sd = gr.Textbox(value="txt2img_sd", visible=False, interactive=False)
                                del_ini_btn_txt2img_sd = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2img_sd.value) else False)
                                save_ini_btn_txt2img_sd.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_txt2img_sd.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2img_sd.click(fn=lambda: del_ini_btn_txt2img_sd.update(interactive=True), outputs=del_ini_btn_txt2img_sd)
                                del_ini_btn_txt2img_sd.click(fn=lambda: del_ini(module_name_txt2img_sd.value))
                                del_ini_btn_txt2img_sd.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2img_sd.click(fn=lambda: del_ini_btn_txt2img_sd.update(interactive=False), outputs=del_ini_btn_txt2img_sd)
                        if test_cfg_exist(module_name_txt2img_sd.value) :
                            readcfg_txt2img_sd = read_ini_txt2img_sd(module_name_txt2img_sd.value)
                            model_txt2img_sd.value = readcfg_txt2img_sd[0]
                            num_inference_step_txt2img_sd.value = readcfg_txt2img_sd[1]
                            sampler_txt2img_sd.value = readcfg_txt2img_sd[2]
                            guidance_scale_txt2img_sd.value = readcfg_txt2img_sd[3]
                            num_images_per_prompt_txt2img_sd.value = readcfg_txt2img_sd[4]
                            num_prompt_txt2img_sd.value = readcfg_txt2img_sd[5]
                            width_txt2img_sd.value = readcfg_txt2img_sd[6]
                            height_txt2img_sd.value = readcfg_txt2img_sd[7]
                            seed_txt2img_sd.value = readcfg_txt2img_sd[8]
                            use_gfpgan_txt2img_sd.value = readcfg_txt2img_sd[9]
                            tkme_txt2img_sd.value = readcfg_txt2img_sd[10]
                        with gr.Accordion("LoRA models", open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_txt2img_sd = gr.Dropdown(choices=list(lora_model_list(model_txt2img_sd.value).keys()), value="", label="LoRA model", info="Choose LoRA model to use for inference")
                                with gr.Column():
                                    lora_weight_txt2img_sd = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="LoRA weight", info="Weight of the LoRA model in the final result")
                        with gr.Accordion("Textual inversion", open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_txt2img_sd = gr.Dropdown(choices=list(txtinv_list(model_txt2img_sd.value).keys()), value="", label="Textual inversion", info="Choose textual inversion to use for inference")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2img_sd = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_txt2img_sd = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
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
                        lora_model_txt2img_sd.change(fn=change_lora_model_txt2img_sd, inputs=[model_txt2img_sd, lora_model_txt2img_sd, prompt_txt2img_sd], outputs=[prompt_txt2img_sd])
                        txtinv_txt2img_sd.change(fn=change_txtinv_txt2img_sd, inputs=[model_txt2img_sd, txtinv_txt2img_sd, prompt_txt2img_sd, negative_prompt_txt2img_sd], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd])
                        with gr.Column(scale=2):
                            out_txt2img_sd = gr.Gallery(
                                label="Generated images",
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
                                    download_btn_txt2img_sd = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_txt2img_sd = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_txt2img_sd.click(fn=zip_download_file_txt2img_sd, inputs=out_txt2img_sd, outputs=[download_file_txt2img_sd, download_file_txt2img_sd])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_sd = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_sd_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2img_sd_cancel.click(fn=initiate_stop_txt2img_sd, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2img_sd_clear_input = gr.ClearButton(components=[prompt_txt2img_sd, negative_prompt_txt2img_sd], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2img_sd_clear_output = gr.ClearButton(components=[out_txt2img_sd, gs_out_txt2img_sd], value="Clear outputs 🧹")   
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
                                lora_model_txt2img_sd,
                                lora_weight_txt2img_sd,
                                txtinv_txt2img_sd,
                            ],
                                outputs=[out_txt2img_sd, gs_out_txt2img_sd],
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')                                        
                                        txt2img_sd_llava = gr.Button(" >> Llava")
                                        txt2img_sd_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        txt2img_sd_img2img = gr.Button(" >> img2img")
                                        txt2img_sd_img2img_ip = gr.Button(" >> IP-Adapter")
                                        txt2img_sd_img2var = gr.Button(" >> Image variation")
                                        txt2img_sd_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        txt2img_sd_magicmix = gr.Button(" >> MagicMix")
                                        txt2img_sd_inpaint = gr.Button(" >> inpaint")
                                        txt2img_sd_paintbyex = gr.Button(" >> Paint by example") 
                                        txt2img_sd_outpaint = gr.Button(" >> outpaint")
                                        txt2img_sd_controlnet = gr.Button(" >> ControlNet")
                                        txt2img_sd_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        txt2img_sd_faceswap = gr.Button(" >> Faceswap target")
                                        txt2img_sd_resrgan = gr.Button(" >> Real ESRGAN")
                                        txt2img_sd_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        txt2img_sd_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...')
                                        txt2img_sd_img2shape = gr.Button(" >> Shap-E img2shape")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_sd_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        txt2img_sd_txt2img_lcm_input = gr.Button(" >> LCM")
                                        txt2img_sd_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        txt2img_sd_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        txt2img_sd_img2img_input = gr.Button(" >> img2img")
                                        txt2img_sd_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        txt2img_sd_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        txt2img_sd_inpaint_input = gr.Button(" >> inpaint")
                                        txt2img_sd_controlnet_input = gr.Button(" >> ControlNet")
                                        txt2img_sd_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                        
                                        txt2img_sd_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        txt2img_sd_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")                                        
                                        txt2img_sd_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...') 
                                        txt2img_sd_img2img_both = gr.Button(" +  >> img2img")
                                        txt2img_sd_img2img_ip_both = gr.Button(" +  >> IP-Adapter")                                        
                                        txt2img_sd_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        txt2img_sd_inpaint_both = gr.Button(" +  >> inpaint")
                                        txt2img_sd_controlnet_both = gr.Button(" + ️ >> ControlNet")
                                        txt2img_sd_faceid_ip_both = gr.Button(" + ️ >> IP-Adapter FaceID")

# Kandinsky
                if ram_size() >= 16 :
                    titletab_txt2img_kd = "Kandinsky"
                else :
                    titletab_txt2img_kd = "Kandinsky ⛔"

                with gr.TabItem(titletab_txt2img_kd, id=22) as tab_txt2img_kd:                    
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left; text-decoration: underline;'>Informations</h1>
                                <b>Module : </b>Kandinsky</br>
                                <b>Function : </b>Generate images from a prompt and a negative prompt using <a href='https://github.com/ai-forever/Kandinsky-2' target='_blank'>Kandinsky</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder' target='_blank'>kandinsky-community/kandinsky-2-2-decoder</a>, 
                                <a href='https://huggingface.co/kandinsky-community/kandinsky-3' target='_blank'>kandinsky-community/kandinsky-3</a>, 
                                <a href='https://huggingface.co/kandinsky-community/kandinsky-2-1' target='_blank'>kandinsky-community/kandinsky-2-1</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left; text-decoration: underline;'>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to use another model, generate several images in a single run or change dimensions of the outputs</br>
                                - Click the <b>generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.                                
                                </div>
                                """
                            )                                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_kd = gr.Dropdown(choices=model_list_txt2img_kd, value=model_list_txt2img_kd[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2img_kd = gr.Slider(1, biniou_global_steps_max, step=1, value=25, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_kd = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[5], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_kd = gr.Slider(0.1, 20.0, step=0.1, value=4.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2img_kd = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_kd = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_kd = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_kd = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_kd = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_kd = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_kd = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2img_kd = gr.Textbox(value="txt2img_kd", visible=False, interactive=False)
                                del_ini_btn_txt2img_kd = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2img_kd.value) else False)
                                save_ini_btn_txt2img_kd.click(
                                    fn=write_ini, 
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
                                save_ini_btn_txt2img_kd.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2img_kd.click(fn=lambda: del_ini_btn_txt2img_kd.update(interactive=True), outputs=del_ini_btn_txt2img_kd)
                                del_ini_btn_txt2img_kd.click(fn=lambda: del_ini(module_name_txt2img_kd.value))
                                del_ini_btn_txt2img_kd.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2img_kd.click(fn=lambda: del_ini_btn_txt2img_kd.update(interactive=False), outputs=del_ini_btn_txt2img_kd)
                        if test_cfg_exist(module_name_txt2img_kd.value) :
                            readcfg_txt2img_kd = read_ini_txt2img_kd(module_name_txt2img_kd.value)
                            model_txt2img_kd.value = readcfg_txt2img_kd[0]
                            num_inference_step_txt2img_kd.value = readcfg_txt2img_kd[1]
                            sampler_txt2img_kd.value = readcfg_txt2img_kd[2]
                            guidance_scale_txt2img_kd.value = readcfg_txt2img_kd[3]
                            num_images_per_prompt_txt2img_kd.value = readcfg_txt2img_kd[4]
                            num_prompt_txt2img_kd.value = readcfg_txt2img_kd[5]
                            width_txt2img_kd.value = readcfg_txt2img_kd[6]
                            height_txt2img_kd.value = readcfg_txt2img_kd[7]
                            seed_txt2img_kd.value = readcfg_txt2img_kd[8]
                            use_gfpgan_txt2img_kd.value = readcfg_txt2img_kd[9]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():   
                                    prompt_txt2img_kd = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="An alien cheeseburger creature eating itself, claymation, cinematic, moody lighting")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2img_kd = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="low quality, bad quality")
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
                                label="Generated images",
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
                                    download_btn_txt2img_kd = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_txt2img_kd = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_txt2img_kd.click(fn=zip_download_file_txt2img_kd, inputs=out_txt2img_kd, outputs=[download_file_txt2img_kd, download_file_txt2img_kd])                            
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_kd = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_kd_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2img_kd_cancel.click(fn=initiate_stop_txt2img_kd, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2img_kd_clear_input = gr.ClearButton(components=[prompt_txt2img_kd, negative_prompt_txt2img_kd], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2img_kd_clear_output = gr.ClearButton(components=[out_txt2img_kd, gs_out_txt2img_kd], value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        txt2img_kd_llava = gr.Button(" >> Llava")
                                        txt2img_kd_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        txt2img_kd_img2img = gr.Button(" >> img2img")
                                        txt2img_kd_img2img_ip = gr.Button(" >> IP-Adapter")
                                        txt2img_kd_img2var = gr.Button(" >> Image variation")
                                        txt2img_kd_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        txt2img_kd_magicmix = gr.Button(" >> MagicMix")
                                        txt2img_kd_inpaint = gr.Button(" >> inpaint")
                                        txt2img_kd_paintbyex = gr.Button(" >> Paint by example") 
                                        txt2img_kd_outpaint = gr.Button(" >> outpaint")
                                        txt2img_kd_controlnet = gr.Button(" >> ControlNet")
                                        txt2img_kd_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        txt2img_kd_faceswap = gr.Button(" >> Faceswap target")
                                        txt2img_kd_resrgan = gr.Button(" >> Real ESRGAN")
                                        txt2img_kd_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        txt2img_kd_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        txt2img_kd_img2shape = gr.Button(" >> Shap-E img2shape")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_kd_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        txt2img_kd_txt2img_lcm_input = gr.Button(" >> LCM")
                                        txt2img_kd_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        txt2img_kd_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        txt2img_kd_img2img_input = gr.Button(" >> img2img")
                                        txt2img_kd_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        txt2img_kd_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        txt2img_kd_inpaint_input = gr.Button(" >> inpaint")
                                        txt2img_kd_controlnet_input = gr.Button(" >> ControlNet")                                        
                                        txt2img_kd_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                                                                
                                        txt2img_kd_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        txt2img_kd_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")
                                        txt2img_kd_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_kd_img2img_both = gr.Button(" +  >> img2img")
                                        txt2img_kd_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        txt2img_kd_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        txt2img_kd_inpaint_both = gr.Button(" +  >> inpaint")
                                        txt2img_kd_controlnet_both = gr.Button(" +  >> ControlNet")
                                        txt2img_kd_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")

# LCM
                with gr.TabItem("LCM", id=23) as tab_txt2img_lcm:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>LCM</br>
                                <b>Function : </b>Generate images from a prompt using <a href='https://github.com/luosiallen/latent-consistency-model' target='_blank'>LCM (Latent Consistency Model)</a></br>
                                <b>Input(s) : </b>Prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7' target='_blank'>SimianLuo/LCM_Dreamshaper_v7</a>, 
                                <a href='https://huggingface.co/segmind/Segmind-VegaRT' target='_blank'>segmind/Segmind-VegaRT</a>, 
                                <a href='https://huggingface.co/latent-consistency/lcm-ssd-1b' target='_blank'>latent-consistency/lcm-ssd-1b</a>, 
                                <a href='https://huggingface.co/latent-consistency/lcm-sdxl' target='_blank'>latent-consistency/lcm-sdxl</a>, 
                                <a href='https://huggingface.co/latent-consistency/lcm-lora-sdv1-5' target='_blank'>latent-consistency/lcm-lora-sdv1-5</a>, 
                                <a href='https://huggingface.co/latent-consistency/lcm-lora-sdxl' target='_blank'>latent-consistency/lcm-lora-sdxl</a>, 
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to generate several images in a single run or change dimensions of the outputs</br>
                                - (optional) Select a LoRA model and set its weight</br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.</br>
                                <b>LoRA models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors LoRA models in the directory ./biniou/models/lora/SD or ./biniou/models/lora/SDXL (depending on the LoRA model type : SD 1.5 or SDXL). Restart Pixify to see them in the models list.</br>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_lcm = gr.Dropdown(choices=model_list_txt2img_lcm, value=model_list_txt2img_lcm[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2img_lcm = gr.Slider(1, biniou_global_steps_max, step=1, value=4, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_lcm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[13], label="Sampler", info="Sampler to use for inference", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_lcm = gr.Slider(0.1, 20.0, step=0.1, value=8.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                lcm_origin_steps_txt2img_lcm = gr.Slider(1, biniou_global_steps_max, step=1, value=50, label="LCM origin steps", info="LCM origin steps")
                            with gr.Column():
                                num_images_per_prompt_txt2img_lcm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_lcm = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_lcm = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_lcm = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_lcm = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility") 
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_lcm = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_txt2img_lcm = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_lcm = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2img_lcm = gr.Textbox(value="txt2img_lcm", visible=False, interactive=False)
                                del_ini_btn_txt2img_lcm = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2img_lcm.value) else False)
                                save_ini_btn_txt2img_lcm.click(
                                    fn=write_ini, 
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
                                save_ini_btn_txt2img_lcm.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2img_lcm.click(fn=lambda: del_ini_btn_txt2img_lcm.update(interactive=True), outputs=del_ini_btn_txt2img_lcm)
                                del_ini_btn_txt2img_lcm.click(fn=lambda: del_ini(module_name_txt2img_lcm.value))
                                del_ini_btn_txt2img_lcm.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2img_lcm.click(fn=lambda: del_ini_btn_txt2img_lcm.update(interactive=False), outputs=del_ini_btn_txt2img_lcm)
                        if test_cfg_exist(module_name_txt2img_lcm.value) :
                            readcfg_txt2img_lcm = read_ini_txt2img_lcm(module_name_txt2img_lcm.value)
                            model_txt2img_lcm.value = readcfg_txt2img_lcm[0]
                            num_inference_step_txt2img_lcm.value = readcfg_txt2img_lcm[1]
                            sampler_txt2img_lcm.value = readcfg_txt2img_lcm[2]
                            guidance_scale_txt2img_lcm.value = readcfg_txt2img_lcm[3]
                            lcm_origin_steps_txt2img_lcm.value = readcfg_txt2img_lcm[4]
                            num_images_per_prompt_txt2img_lcm.value = readcfg_txt2img_lcm[5]
                            num_prompt_txt2img_lcm.value = readcfg_txt2img_lcm[6]
                            width_txt2img_lcm.value = readcfg_txt2img_lcm[7]
                            height_txt2img_lcm.value = readcfg_txt2img_lcm[8]
                            seed_txt2img_lcm.value = readcfg_txt2img_lcm[9]
                            use_gfpgan_txt2img_lcm.value = readcfg_txt2img_lcm[10]
                            tkme_txt2img_lcm.value = readcfg_txt2img_lcm[11]
                        with gr.Accordion("LoRA Model", open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_txt2img_lcm = gr.Dropdown(choices=list(lora_model_list(model_txt2img_lcm.value).keys()), value="", label="LoRA model", info="Choose LoRA model to use for inference")
                                with gr.Column():
                                    lora_weight_txt2img_lcm = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="LoRA weight", info="Weight of the LoRA model in the final result")
                        with gr.Accordion("Textual inversion", open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_txt2img_lcm = gr.Dropdown(choices=list(txtinv_list(model_txt2img_lcm.value).keys()), value="", label="Textual inversion", info="Choose textual inversion to use for inference")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2img_lcm = gr.Textbox(lines=18, max_lines=18, label="Prompt", info="Describe what you want in your image", placeholder="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
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
                        lora_model_txt2img_lcm.change(fn=change_lora_model_txt2img_lcm, inputs=[model_txt2img_lcm, lora_model_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        txtinv_txt2img_lcm.change(fn=change_txtinv_txt2img_lcm, inputs=[model_txt2img_lcm, txtinv_txt2img_lcm, prompt_txt2img_lcm], outputs=[prompt_txt2img_lcm])
                        with gr.Column(scale=2):
                            out_txt2img_lcm = gr.Gallery(
                                label="Generated images",
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
                                    download_btn_txt2img_lcm = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_txt2img_lcm = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_txt2img_lcm.click(fn=zip_download_file_txt2img_lcm, inputs=out_txt2img_lcm, outputs=[download_file_txt2img_lcm, download_file_txt2img_lcm])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_lcm = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_txt2img_lcm_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2img_lcm_cancel.click(fn=initiate_stop_txt2img_lcm, inputs=None, outputs=None)
                        with gr.Column():
                            btn_txt2img_lcm_clear_input = gr.ClearButton(components=[prompt_txt2img_lcm], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2img_lcm_clear_output = gr.ClearButton(components=[out_txt2img_lcm, gs_out_txt2img_lcm], value="Clear outputs 🧹")   
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
                                txtinv_txt2img_lcm,
                            ],
                                outputs=[out_txt2img_lcm, gs_out_txt2img_lcm],
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')                                        
                                        txt2img_lcm_llava = gr.Button(" >> Llava")
                                        txt2img_lcm_img2txt_git = gr.Button(" >> GIT Captioning")      
                                        gr.HTML(value='... image module ...')
                                        txt2img_lcm_img2img = gr.Button(" >> img2img")
                                        txt2img_lcm_img2img_ip = gr.Button(" >> IP-Adapter")
                                        txt2img_lcm_img2var = gr.Button(" >> Image variation")
                                        txt2img_lcm_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        txt2img_lcm_magicmix = gr.Button(" >> MagicMix")
                                        txt2img_lcm_inpaint = gr.Button(" >> inpaint")
                                        txt2img_lcm_paintbyex = gr.Button(" >> Paint by example") 
                                        txt2img_lcm_outpaint = gr.Button(" >> outpaint")
                                        txt2img_lcm_controlnet = gr.Button(" >> ControlNet")
                                        txt2img_lcm_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        txt2img_lcm_faceswap = gr.Button(" >> Faceswap target")
                                        txt2img_lcm_resrgan = gr.Button(" >> Real ESRGAN")
                                        txt2img_lcm_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        txt2img_lcm_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        txt2img_lcm_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_lcm_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        txt2img_lcm_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        txt2img_lcm_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        txt2img_lcm_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        txt2img_lcm_img2img_input = gr.Button(" >> img2img")
                                        txt2img_lcm_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        txt2img_lcm_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        txt2img_lcm_inpaint_input = gr.Button(" >> inpaint")
                                        txt2img_lcm_controlnet_input = gr.Button(" >> ControlNet")
                                        txt2img_lcm_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')
                                        txt2img_lcm_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        txt2img_lcm_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")
                                        txt2img_lcm_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_lcm_img2img_both = gr.Button(" +  >> img2img")
                                        txt2img_lcm_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        txt2img_lcm_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        txt2img_lcm_inpaint_both = gr.Button(" +  >> inpaint")
                                        txt2img_lcm_controlnet_both = gr.Button(" + ️ >> ControlNet")
                                        txt2img_lcm_faceid_ip_both = gr.Button(" + ️ >> IP-Adapter FaceID")

# txt2img_mjm
                with gr.TabItem("Midjourney-mini", id=24) as tab_txt2img_mjm:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Midjourney-mini</br>
                                <b>Function : </b>Generate images from a prompt and a negative prompt using <a href='https://huggingface.co/openskyml/midjourney-mini' target='_blank'>Midjourney-mini</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/openskyml/midjourney-mini' target='_blank'>openskyml/midjourney-mini</a>
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to generate several images in a single run or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_mjm = gr.Dropdown(choices=model_list_txt2img_mjm, value=model_list_txt2img_mjm[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2img_mjm = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_mjm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[4], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_mjm = gr.Slider(0.1, 20.0, step=0.1, value=7.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2img_mjm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_mjm = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_mjm = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_mjm = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_mjm = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")    
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_mjm = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_txt2img_mjm = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_mjm = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2img_mjm = gr.Textbox(value="txt2img_mjm", visible=False, interactive=False)
                                del_ini_btn_txt2img_mjm = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2img_mjm.value) else False)
                                save_ini_btn_txt2img_mjm.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_txt2img_mjm.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2img_mjm.click(fn=lambda: del_ini_btn_txt2img_mjm.update(interactive=True), outputs=del_ini_btn_txt2img_mjm)
                                del_ini_btn_txt2img_mjm.click(fn=lambda: del_ini(module_name_txt2img_mjm.value))
                                del_ini_btn_txt2img_mjm.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2img_mjm.click(fn=lambda: del_ini_btn_txt2img_mjm.update(interactive=False), outputs=del_ini_btn_txt2img_mjm)
                        if test_cfg_exist(module_name_txt2img_mjm.value) :
                            readcfg_txt2img_mjm = read_ini_txt2img_mjm(module_name_txt2img_mjm.value)
                            model_txt2img_mjm.value = readcfg_txt2img_mjm[0]
                            num_inference_step_txt2img_mjm.value = readcfg_txt2img_mjm[1]
                            sampler_txt2img_mjm.value = readcfg_txt2img_mjm[2]
                            guidance_scale_txt2img_mjm.value = readcfg_txt2img_mjm[3]
                            num_images_per_prompt_txt2img_mjm.value = readcfg_txt2img_mjm[4]
                            num_prompt_txt2img_mjm.value = readcfg_txt2img_mjm[5]
                            width_txt2img_mjm.value = readcfg_txt2img_mjm[6]
                            height_txt2img_mjm.value = readcfg_txt2img_mjm[7]
                            seed_txt2img_mjm.value = readcfg_txt2img_mjm[8]
                            use_gfpgan_txt2img_mjm.value = readcfg_txt2img_mjm[9]
                            tkme_txt2img_mjm.value = readcfg_txt2img_mjm[10]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2img_mjm = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_txt2img_mjm = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
                        with gr.Column(scale=2):
                            out_txt2img_mjm = gr.Gallery(
                                label="Generated images",
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
                                    download_btn_txt2img_mjm = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_txt2img_mjm = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_txt2img_mjm.click(fn=zip_download_file_txt2img_mjm, inputs=out_txt2img_mjm, outputs=[download_file_txt2img_mjm, download_file_txt2img_mjm])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_mjm = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_mjm_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2img_mjm_cancel.click(fn=initiate_stop_txt2img_mjm, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2img_mjm_clear_input = gr.ClearButton(components=[prompt_txt2img_mjm, negative_prompt_txt2img_mjm], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2img_mjm_clear_output = gr.ClearButton(components=[out_txt2img_mjm, gs_out_txt2img_mjm], value="Clear outputs 🧹")   
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
                            ],
                                outputs=[out_txt2img_mjm, gs_out_txt2img_mjm],
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        txt2img_mjm_llava = gr.Button(" >> Llava")
                                        txt2img_mjm_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        txt2img_mjm_img2img = gr.Button(" >> img2img")
                                        txt2img_mjm_img2img_ip = gr.Button(" >> IP-Adapter")
                                        txt2img_mjm_img2var = gr.Button(" >> Image variation")
                                        txt2img_mjm_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        txt2img_mjm_magicmix = gr.Button(" >> MagicMix")
                                        txt2img_mjm_inpaint = gr.Button(" >> inpaint")
                                        txt2img_mjm_paintbyex = gr.Button(" >> Paint by example") 
                                        txt2img_mjm_outpaint = gr.Button(" >> outpaint")
                                        txt2img_mjm_controlnet = gr.Button(" >> ControlNet")
                                        txt2img_mjm_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        txt2img_mjm_faceswap = gr.Button(" >> Faceswap target")
                                        txt2img_mjm_resrgan = gr.Button(" >> Real ESRGAN")
                                        txt2img_mjm_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...') 
                                        txt2img_mjm_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        txt2img_mjm_img2shape = gr.Button(" >> Shap-E img2shape")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_mjm_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        txt2img_mjm_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        txt2img_mjm_txt2img_lcm_input = gr.Button(" >> LCM")
                                        txt2img_mjm_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        txt2img_mjm_img2img_input = gr.Button(" >> img2img")
                                        txt2img_mjm_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        txt2img_mjm_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        txt2img_mjm_inpaint_input = gr.Button(" >> inpaint")
                                        txt2img_mjm_controlnet_input = gr.Button(" >> ControlNet")
                                        txt2img_mjm_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                        
                                        txt2img_mjm_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        txt2img_mjm_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")                                        
                                        txt2img_mjm_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...') 
                                        txt2img_mjm_img2img_both = gr.Button(" +  >> img2img")
                                        txt2img_mjm_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        txt2img_mjm_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        txt2img_mjm_inpaint_both = gr.Button(" +  >> inpaint")
                                        txt2img_mjm_controlnet_both = gr.Button(" + ️ >> ControlNet") 
                                        txt2img_mjm_faceid_ip_both = gr.Button(" + ️ >> IP-Adapter FaceID") 

# txt2img_paa
                with gr.TabItem("PixArt-Alpha", id=25) as tab_txt2img_paa:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>PixArt-Alpha</br>
                                <b>Function : </b>Generate images from a prompt and a negative prompt using <a href='https://pixart-alpha.github.io/' target='_blank'>PixArt-Alpha</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512' target='_blank'>PixArt-alpha/PixArt-XL-2-512x512</a>,
                                <a href='https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS' target='_blank'>PixArt-alpha/PixArt-XL-2-1024-MS</a>,
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to use another model, generate several images in a single run or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_paa = gr.Dropdown(choices=model_list_txt2img_paa, value=model_list_txt2img_paa[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2img_paa = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_paa = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_paa = gr.Slider(0.1, 20.0, step=0.1, value=7.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2img_paa = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_paa = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_paa = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_paa = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_paa = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")    
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_paa = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_txt2img_paa = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality", visible=False, interactive=False)
                        model_txt2img_paa.change(fn=change_model_type_txt2img_paa, inputs=model_txt2img_paa, outputs=[width_txt2img_paa, height_txt2img_paa])
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2img_paa = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2img_paa = gr.Textbox(value="txt2img_paa", visible=False, interactive=False)
                                del_ini_btn_txt2img_paa = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2img_paa.value) else False)
                                save_ini_btn_txt2img_paa.click(
                                    fn=write_ini, 
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
                                save_ini_btn_txt2img_paa.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2img_paa.click(fn=lambda: del_ini_btn_txt2img_paa.update(interactive=True), outputs=del_ini_btn_txt2img_paa)
                                del_ini_btn_txt2img_paa.click(fn=lambda: del_ini(module_name_txt2img_paa.value))
                                del_ini_btn_txt2img_paa.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2img_paa.click(fn=lambda: del_ini_btn_txt2img_paa.update(interactive=False), outputs=del_ini_btn_txt2img_paa)
                        if test_cfg_exist(module_name_txt2img_paa.value) :
                            readcfg_txt2img_paa = read_ini_txt2img_paa(module_name_txt2img_paa.value)
                            model_txt2img_paa.value = readcfg_txt2img_paa[0]
                            num_inference_step_txt2img_paa.value = readcfg_txt2img_paa[1]
                            sampler_txt2img_paa.value = readcfg_txt2img_paa[2]
                            guidance_scale_txt2img_paa.value = readcfg_txt2img_paa[3]
                            num_images_per_prompt_txt2img_paa.value = readcfg_txt2img_paa[4]
                            num_prompt_txt2img_paa.value = readcfg_txt2img_paa[5]
                            width_txt2img_paa.value = readcfg_txt2img_paa[6]
                            height_txt2img_paa.value = readcfg_txt2img_paa[7]
                            seed_txt2img_paa.value = readcfg_txt2img_paa[8]
                            use_gfpgan_txt2img_paa.value = readcfg_txt2img_paa[9]
                            tkme_txt2img_paa.value = readcfg_txt2img_paa[10]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2img_paa = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="A small cactus with a happy face in the Sahara desert.")
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_txt2img_paa = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
                        with gr.Column(scale=2):
                            out_txt2img_paa = gr.Gallery(
                                label="Generated images",
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
                                    download_btn_txt2img_paa = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_txt2img_paa = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_txt2img_paa.click(fn=zip_download_file_txt2img_paa, inputs=out_txt2img_paa, outputs=[download_file_txt2img_paa, download_file_txt2img_paa])
                    with gr.Row():
                        with gr.Column():
                            btn_txt2img_paa = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2img_paa_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2img_paa_cancel.click(fn=initiate_stop_txt2img_paa, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2img_paa_clear_input = gr.ClearButton(components=[prompt_txt2img_paa, negative_prompt_txt2img_paa], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2img_paa_clear_output = gr.ClearButton(components=[out_txt2img_paa, gs_out_txt2img_paa], value="Clear outputs 🧹")   
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        txt2img_paa_llava = gr.Button(" >> Llava")
                                        txt2img_paa_img2txt_git = gr.Button(" >> GIT Captioning")      
                                        gr.HTML(value='... image module ...')
                                        txt2img_paa_img2img = gr.Button(" >> img2img")
                                        txt2img_paa_img2img_ip = gr.Button(" >> IP-Adapter")
                                        txt2img_paa_img2var = gr.Button(" >> Image variation")
                                        txt2img_paa_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        txt2img_paa_magicmix = gr.Button(" >> MagicMix")
                                        txt2img_paa_inpaint = gr.Button(" >> inpaint")
                                        txt2img_paa_paintbyex = gr.Button(" >> Paint by example") 
                                        txt2img_paa_outpaint = gr.Button(" >> outpaint")
                                        txt2img_paa_controlnet = gr.Button(" >> ControlNet")
                                        txt2img_paa_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        txt2img_paa_faceswap = gr.Button(" >> Faceswap target")
                                        txt2img_paa_resrgan = gr.Button(" >> Real ESRGAN")
                                        txt2img_paa_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        txt2img_paa_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        txt2img_paa_img2shape = gr.Button(" >> Shap-E img2shape")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_paa_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        txt2img_paa_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        txt2img_paa_txt2img_lcm_input = gr.Button(" >> LCM")
                                        txt2img_paa_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        txt2img_paa_img2img_input = gr.Button(" >> img2img")
                                        txt2img_paa_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        txt2img_paa_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        txt2img_paa_inpaint_input = gr.Button(" >> inpaint")
                                        txt2img_paa_controlnet_input = gr.Button(" >> ControlNet")
                                        txt2img_paa_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                        
                                        txt2img_paa_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        txt2img_paa_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")                                        
                                        txt2img_paa_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...') 
                                        txt2img_paa_img2img_both = gr.Button(" +  >> img2img")
                                        txt2img_paa_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        txt2img_paa_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        txt2img_paa_inpaint_both = gr.Button(" +  >> inpaint")
                                        txt2img_paa_controlnet_both = gr.Button(" + ️ >> IP-Adapter FaceID") 
                                        txt2img_paa_faceid_ip_both = gr.Button(" + ️ >> ControlNet") 
                                        
# img2img    
                with gr.TabItem("img2img", id=26) as tab_img2img:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Img2img</br>
                                <b>Function : </b>Generate images variations of an input image, from a prompt and a negative prompt using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                You could use this module to refine an image produced by another module.</br>
                                <b>Input(s) : </b>Input image, prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/stabilityai/sd-turbo' target='_blank'>stabilityai/sd-turbo</a>, 
                                <a href='https://huggingface.co/stabilityai/sdxl-turbo' target='_blank'>stabilityai/sdxl-turbo</a>, 
                                <a href='https://huggingface.co/thibaud/sdxl_dpo_turbo' target='_blank'>thibaud/sdxl_dpo_turbo</a>
                                <a href='https://huggingface.co/dataautogpt3/OpenDalleV1.1' target='_blank'>dataautogpt3/OpenDalleV1.1</a>, 
                                <a href='https://huggingface.co/dataautogpt3/ProteusV0.4' target='_blank'>dataautogpt3/ProteusV0.4</a>, 
                                <a href='https://huggingface.co/etri-vilab/koala-1b' target='_blank'>etri-vilab/koala-1b</a>, 
                                <a href='https://huggingface.co/etri-vilab/koala-700m' target='_blank'>etri-vilab/koala-700m</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/segmind/Segmind-Vega' target='_blank'>segmind/Segmind-Vega</a>, 
                                <a href='https://huggingface.co/segmind/SSD-1B' target='_blank'>segmind/SSD-1B</a>, 
                                <a href='https://huggingface.co/gsdf/Counterfeit-V2.5' target='_blank'>gsdf/Counterfeit-V2.5</a>, 
                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0' target='_blank'>stabilityai/stable-diffusion-xl-refiner-1.0</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a>
                                """
#                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model, generate several images in a single run</br>
                                - (optional) Select a LoRA model and set its weight</br>
                                - Upload, import an image or draw a sketch as an <b>Input image</b></br>
                                - Set the balance between the input image and the prompt (<b>denoising strength</b>) to a value between 0 and 1 : 0 will completely ignore the prompt, 1 will completely ignore the input image</br>                                
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.</br>
                                <b>LoRA models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors LoRA models in the directory ./biniou/models/lora/SD or ./biniou/models/lora/SDXL (depending on the LoRA model type : SD 1.5 or SDXL). Restart Pixify to see them in the models list.</br>
                                </div>
                                """
                            )               
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2img = gr.Dropdown(choices=model_list_img2img, value=model_list_img2img[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_img2img = gr.Slider(2, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_img2img = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2img = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_img2img = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_img2img = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_img2img = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_img2img = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_img2img = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_img2img = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_img2img = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")    
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2img = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_img2img = gr.Textbox(value="img2img", visible=False, interactive=False)
                                del_ini_btn_img2img = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_img2img.value) else False)
                                save_ini_btn_img2img.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_img2img.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_img2img.click(fn=lambda: del_ini_btn_img2img.update(interactive=True), outputs=del_ini_btn_img2img)
                                del_ini_btn_img2img.click(fn=lambda: del_ini(module_name_img2img.value))
                                del_ini_btn_img2img.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_img2img.click(fn=lambda: del_ini_btn_img2img.update(interactive=False), outputs=del_ini_btn_img2img)
                        if test_cfg_exist(module_name_img2img.value) :
                            readcfg_img2img = read_ini_img2img(module_name_img2img.value)
                            model_img2img.value = readcfg_img2img[0]
                            num_inference_step_img2img.value = readcfg_img2img[1]
                            sampler_img2img.value = readcfg_img2img[2]
                            guidance_scale_img2img.value = readcfg_img2img[3]
                            num_images_per_prompt_img2img.value = readcfg_img2img[4]
                            num_prompt_img2img.value = readcfg_img2img[5]
                            width_img2img.value = readcfg_img2img[6]
                            height_img2img.value = readcfg_img2img[7]
                            seed_img2img.value = readcfg_img2img[8]
                            use_gfpgan_img2img.value = readcfg_img2img[9]
                            tkme_img2img.value = readcfg_img2img[10]
                        with gr.Accordion("LoRA Model", open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_img2img = gr.Dropdown(choices=list(lora_model_list(model_img2img.value).keys()), value="", label="LoRA model", info="Choose LoRA model to use for inference")
                                with gr.Column():
                                    lora_weight_img2img = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="LoRA weight", info="Weight of the LoRA model in the final result")
                        with gr.Accordion("Textual inversion", open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_img2img = gr.Dropdown(choices=list(txtinv_list(model_img2img.value).keys()), value="", label="Textual inversion", info="Choose textual inversion to use for inference")
                    with gr.Row():
                        with gr.Column():
                            img_img2img = gr.Image(label="Input image", height=400, type="filepath")
                            with gr.Row():
                                source_type_img2img = gr.Radio(choices=["image", "sketch"], value="image", label="Input type", info="Choose input type")
                                img_img2img.change(image_upload_event, inputs=img_img2img, outputs=[width_img2img, height_img2img])
                                source_type_img2img.change(fn=change_source_type_img2img, inputs=source_type_img2img, outputs=img_img2img)                                
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_img2img = gr.Slider(0.01, 1.0, step=0.01, value=0.75, label="Denoising strength", info="Balance between input image (0) and prompts (1)")  
                            with gr.Row():
                                with gr.Column():
                                    prompt_img2img = gr.Textbox(lines=5, max_lines=5, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():                                    
                                with gr.Column():
                                    negative_prompt_img2img = gr.Textbox(lines=5, max_lines=5, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
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
                        lora_model_img2img.change(fn=change_lora_model_img2img, inputs=[model_img2img, lora_model_img2img, prompt_img2img], outputs=[prompt_img2img])
                        txtinv_img2img.change(fn=change_txtinv_img2img, inputs=[model_img2img, txtinv_img2img, prompt_img2img, negative_prompt_img2img], outputs=[prompt_img2img, negative_prompt_img2img])
                        denoising_strength_img2img.change(check_steps_strength, [num_inference_step_img2img, denoising_strength_img2img, model_img2img], [num_inference_step_img2img])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                            
                                    out_img2img = gr.Gallery(
                                        label="Generated images",
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
                                        download_btn_img2img = gr.Button("Zip gallery 💾")
                                    with gr.Column():
                                        download_file_img2img = gr.File(label="Output", height=30, interactive=False, visible=False)
                                        download_btn_img2img.click(fn=zip_download_file_img2img, inputs=out_img2img, outputs=[download_file_img2img, download_file_img2img])
                    with gr.Row():
                        with gr.Column():
                            btn_img2img = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_img2img_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_img2img_cancel.click(fn=initiate_stop_img2img, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_img2img_clear_input = gr.ClearButton(components=[img_img2img, prompt_img2img, negative_prompt_img2img], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_img2img_clear_output = gr.ClearButton(components=[out_img2img, gs_out_img2img], value="Clear outputs 🧹")
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
                                    lora_model_img2img,
                                    lora_weight_img2img,
                                    txtinv_img2img,
                                ],
                                outputs=[out_img2img, gs_out_img2img], 
                                show_progress="full",
                            )  
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        img2img_llava = gr.Button(" >> Llava")
                                        img2img_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        img2img_img2img = gr.Button(" >> img2img")
                                        img2img_img2img_ip = gr.Button(" >> IP-Adapter")
                                        img2img_img2var = gr.Button(" >> Image variation")
                                        img2img_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        img2img_inpaint = gr.Button(" >> inpaint")
                                        img2img_magicmix = gr.Button(" >> MagicMix")
                                        img2img_paintbyex = gr.Button(" >> Paint by example") 
                                        img2img_outpaint = gr.Button(" >> outpaint")
                                        img2img_controlnet = gr.Button(" >> ControlNet")
                                        img2img_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        img2img_faceswap = gr.Button(" >> Faceswap target")
                                        img2img_resrgan = gr.Button(" >> Real ESRGAN")
                                        img2img_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        img2img_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...')
                                        img2img_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        img2img_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        img2img_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        img2img_txt2img_lcm_input = gr.Button(" >> LCM")
                                        img2img_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        img2img_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        img2img_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        img2img_inpaint_input = gr.Button(" >> inpaint")
                                        img2img_controlnet_input = gr.Button(" >> ControlNet")
                                        img2img_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        img2img_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        img2img_inpaint_both = gr.Button(" +  >> inpaint")
                                        img2img_controlnet_both = gr.Button(" +  >> ControlNet")
                                        img2img_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")

# img2img_ip    
                with gr.TabItem("IP-Adapter", id=27) as tab_img2img_ip:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>IP-Adapter</br>
                                <b>Function : </b>Transform an input image, with a conditional IP-Adapter image, a prompt and a negative prompt using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> and  <a href='https://ip-adapter.github.io/' target='_blank'>IP-Adapter</a></br>
                                <b>Input(s) : </b>Input image, conditional IP-Adapter image, prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/stabilityai/sdxl-turbo' target='_blank'>stabilityai/sdxl-turbo</a>, 
                                <a href='https://huggingface.co/dataautogpt3/OpenDalleV1.1' target='_blank'>dataautogpt3/OpenDalleV1.1</a>, 
                                <a href='https://huggingface.co/dataautogpt3/ProteusV0.4' target='_blank'>dataautogpt3/ProteusV0.4</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/gsdf/Counterfeit-V2.5' target='_blank'>gsdf/Counterfeit-V2.5</a>, 
                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0' target='_blank'>stabilityai/stable-diffusion-xl-base-1.0</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a>
                                """
#                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0' target='_blank'>stabilityai/stable-diffusion-xl-refiner-1.0</a>,
#                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model or generate several images in a single run</br>
                                - (optional) Select a LoRA model and set its weight</br>
                                - Upload or import an image as an <b>Input image</b></br>
                                - Upload an image as an <b>IP-Adapter image</b></br>
                                - Set the balance between the input image and the prompts (Ip-Adapter image, prompts, negative prompt) by choosing a <b>denoising strength</b> value between 0.01 and 1 : 0.01 will mostly ignore the prompts, 1 will completely ignore the input image</br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.</br>
                                <b>LoRA models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors LoRA models in the directory ./biniou/models/lora/SD or ./biniou/models/lora/SDXL (depending on the LoRA model type : SD 1.5 or SDXL). Restart Pixify to see them in the models list.</br>
                                </br>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2img_ip = gr.Dropdown(choices=model_list_img2img_ip, value=model_list_img2img_ip[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_img2img_ip = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_img2img_ip = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2img_ip = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_img2img_ip = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_img2img_ip = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_img2img_ip = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_img2img_ip = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_img2img_ip = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_img2img_ip = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_img2img_ip = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")    
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2img_ip = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_img2img_ip = gr.Textbox(value="img2img_ip", visible=False, interactive=False)
                                del_ini_btn_img2img_ip = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_img2img_ip.value) else False)
                                save_ini_btn_img2img_ip.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_img2img_ip.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_img2img_ip.click(fn=lambda: del_ini_btn_img2img_ip.update(interactive=True), outputs=del_ini_btn_img2img_ip)
                                del_ini_btn_img2img_ip.click(fn=lambda: del_ini(module_name_img2img_ip.value))
                                del_ini_btn_img2img_ip.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_img2img_ip.click(fn=lambda: del_ini_btn_img2img_ip.update(interactive=False), outputs=del_ini_btn_img2img_ip)
                        if test_cfg_exist(module_name_img2img_ip.value) :
                            readcfg_img2img_ip = read_ini_img2img_ip(module_name_img2img_ip.value)
                            model_img2img_ip.value = readcfg_img2img_ip[0]
                            num_inference_step_img2img_ip.value = readcfg_img2img_ip[1]
                            sampler_img2img_ip.value = readcfg_img2img_ip[2]
                            guidance_scale_img2img_ip.value = readcfg_img2img_ip[3]
                            num_images_per_prompt_img2img_ip.value = readcfg_img2img_ip[4]
                            num_prompt_img2img_ip.value = readcfg_img2img_ip[5]
                            width_img2img_ip.value = readcfg_img2img_ip[6]
                            height_img2img_ip.value = readcfg_img2img_ip[7]
                            seed_img2img_ip.value = readcfg_img2img_ip[8]
                            use_gfpgan_img2img_ip.value = readcfg_img2img_ip[9]
                            tkme_img2img_ip.value = readcfg_img2img_ip[10]
                        with gr.Accordion("LoRA Model", open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_img2img_ip = gr.Dropdown(choices=list(lora_model_list(model_img2img_ip.value).keys()), value="", label="LoRA model", info="Choose LoRA model to use for inference")
                                with gr.Column():
                                    lora_weight_img2img_ip = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="LoRA weight", info="Weight of the LoRA model in the final result")
                        with gr.Accordion("Textual inversion", open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_img2img_ip = gr.Dropdown(choices=list(txtinv_list(model_img2img_ip.value).keys()), value="", label="Textual inversion", info="Choose textual inversion to use for inference")
                    with gr.Row():
                        with gr.Column():
                            img_img2img_ip = gr.Image(label="Input image", height=400, type="filepath")
                            img_img2img_ip.change(image_upload_event, inputs=img_img2img_ip, outputs=[width_img2img_ip, height_img2img_ip])
                        with gr.Column():
                            img_ipa_img2img_ip = gr.Image(label="IP-Adapter image", height=400, type="filepath")
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_img2img_ip = gr.Slider(0.01, 1.0, step=0.01, value=0.6, label="Denoising strength", info="Balance between input image (0) and prompts (1)")  
                            with gr.Row():
                                with gr.Column():
                                    prompt_img2img_ip = gr.Textbox(lines=1, max_lines=1, label="Prompt", info="Describe what you want in your image", placeholder="wearing sunglasses, high quality")
                            with gr.Row():                                    
                                with gr.Column():
                                    negative_prompt_img2img_ip = gr.Textbox(lines=1, max_lines=1, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="low quality, medium quality, blurry")
                        denoising_strength_img2img_ip.change(check_steps_strength, [num_inference_step_img2img_ip, denoising_strength_img2img_ip, model_img2img_ip], [num_inference_step_img2img_ip])
                        model_img2img_ip.change(
                            fn=change_model_type_img2img_ip,
                            inputs=[model_img2img_ip],
                            outputs=[
                                sampler_img2img_ip,
                                width_img2img_ip,
                                height_img2img_ip,
                                num_inference_step_img2img_ip,
                                guidance_scale_img2img_ip,
                                lora_model_img2img_ip,
                                txtinv_img2img_ip,
                                negative_prompt_img2img_ip,
                            ]
                        )
                        lora_model_img2img_ip.change(fn=change_lora_model_img2img_ip, inputs=[model_img2img_ip, lora_model_img2img_ip, prompt_img2img_ip], outputs=[prompt_img2img_ip])
                        txtinv_img2img_ip.change(fn=change_txtinv_img2img_ip, inputs=[model_img2img_ip, txtinv_img2img_ip, prompt_img2img_ip, negative_prompt_img2img_ip], outputs=[prompt_img2img_ip, negative_prompt_img2img_ip])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                            
                                    out_img2img_ip = gr.Gallery(
                                        label="Generated images",
                                        show_label=True,
                                        elem_id="gallery_i2i",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                )
                                gs_out_img2img_ip = gr.State()
                                sel_out_img2img_ip = gr.Number(precision=0, visible=False)                              
                                out_img2img_ip.select(get_select_index, None, sel_out_img2img_ip)
                                with gr.Row():
                                    with gr.Column():
                                        download_btn_img2img_ip = gr.Button("Zip gallery 💾")
                                    with gr.Column():
                                        download_file_img2img_ip = gr.File(label="Output", height=30, interactive=False, visible=False)
                                        download_btn_img2img_ip.click(fn=zip_download_file_img2img_ip, inputs=out_img2img_ip, outputs=[download_file_img2img_ip, download_file_img2img_ip])                                
                    with gr.Row():
                        with gr.Column():
                            btn_img2img_ip = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_img2img_ip_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_img2img_ip_cancel.click(fn=initiate_stop_img2img_ip, inputs=None, outputs=None)
                        with gr.Column():
                            btn_img2img_ip_clear_input = gr.ClearButton(components=[img_img2img_ip, img_ipa_img2img_ip, prompt_img2img_ip, negative_prompt_img2img_ip], value="Clear inputs 🧹")
                        with gr.Column():
                            btn_img2img_ip_clear_output = gr.ClearButton(components=[out_img2img_ip, gs_out_img2img_ip], value="Clear outputs 🧹")
                            btn_img2img_ip.click(fn=hide_download_file_img2img_ip, inputs=None, outputs=download_file_img2img_ip)
                            btn_img2img_ip.click(
                                fn=image_img2img_ip,
                                inputs=[
                                    model_img2img_ip,
                                    sampler_img2img_ip,
                                    img_img2img_ip,
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
                                    lora_model_img2img_ip,
                                    lora_weight_img2img_ip,
                                    txtinv_img2img_ip,
                                ],
                                outputs=[out_img2img_ip, gs_out_img2img_ip],
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        img2img_ip_llava = gr.Button(" >> Llava")
                                        img2img_ip_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        img2img_ip_img2img = gr.Button(" >> img2img")
                                        img2img_ip_img2img_ip = gr.Button(" >> IP-Adapter")
                                        img2img_ip_img2var = gr.Button(" >> Image variation")
                                        img2img_ip_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        img2img_ip_inpaint = gr.Button(" >> inpaint")
                                        img2img_ip_magicmix = gr.Button(" >> MagicMix")
                                        img2img_ip_paintbyex = gr.Button(" >> Paint by example") 
                                        img2img_ip_outpaint = gr.Button(" >> outpaint")
                                        img2img_ip_controlnet = gr.Button(" >> ControlNet")
                                        img2img_ip_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        img2img_ip_faceswap = gr.Button(" >> Faceswap target")
                                        img2img_ip_resrgan = gr.Button(" >> Real ESRGAN")
                                        img2img_ip_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        img2img_ip_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...')
                                        img2img_ip_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        img2img_ip_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        img2img_ip_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        img2img_ip_txt2img_lcm_input = gr.Button(" >> LCM")
                                        img2img_ip_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        img2img_ip_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        img2img_ip_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        img2img_ip_inpaint_input = gr.Button(" >> inpaint")
                                        img2img_ip_controlnet_input = gr.Button(" >> ControlNet")
                                        img2img_ip_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        img2img_ip_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        img2img_ip_inpaint_both = gr.Button(" +  >> inpaint")
                                        img2img_ip_controlnet_both = gr.Button(" +  >> ControlNet")
                                        img2img_ip_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")

# img2var    
                if ram_size() >= 16 :
                    titletab_img2var = "Image variation"
                else :
                    titletab_img2var = "Image variation ⛔"

                with gr.TabItem(titletab_img2var, id=28) as tab_img2var: 
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Image variation</br>
                                <b>Function : </b>Generate variations of an input image using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>Input(s) : </b>Input image</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/lambdalabs/sd-image-variations-diffusers' target='_blank'>lambdalabs/sd-image-variations-diffusers</a>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image as an <b>Input image</b></br>
                                - (optional) Modify the settings to generate several images in a single run</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                """
                            )               
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2var = gr.Dropdown(choices=model_list_img2var, value=model_list_img2var[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_img2var = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_img2var = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2var = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_img2var = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_img2var = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_img2var = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_img2var = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_img2var = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_img2var = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_img2var = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2var = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_img2var = gr.Textbox(value="img2var", visible=False, interactive=False)
                                del_ini_btn_img2var = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_img2var.value) else False)
                                save_ini_btn_img2var.click(
                                    fn=write_ini, 
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
                                save_ini_btn_img2var.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_img2var.click(fn=lambda: del_ini_btn_img2var.update(interactive=True), outputs=del_ini_btn_img2var)
                                del_ini_btn_img2var.click(fn=lambda: del_ini(module_name_img2var.value))
                                del_ini_btn_img2var.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_img2var.click(fn=lambda: del_ini_btn_img2var.update(interactive=False), outputs=del_ini_btn_img2var)
                        if test_cfg_exist(module_name_img2var.value) :
                            readcfg_img2var = read_ini_img2var(module_name_img2var.value)
                            model_img2var.value = readcfg_img2var[0]
                            num_inference_step_img2var.value = readcfg_img2var[1]
                            sampler_img2var.value = readcfg_img2var[2]
                            guidance_scale_img2var.value = readcfg_img2var[3]
                            num_images_per_prompt_img2var.value = readcfg_img2var[4]
                            num_prompt_img2var.value = readcfg_img2var[5]
                            width_img2var.value = readcfg_img2var[6]
                            height_img2var.value = readcfg_img2var[7]
                            seed_img2var.value = readcfg_img2var[8]
                            use_gfpgan_img2var.value = readcfg_img2var[9]
                            tkme_img2var.value = readcfg_img2var[10]
                    with gr.Row():
                        with gr.Column():
                            img_img2var = gr.Image(label="Input image", height=400, type="filepath")
                            img_img2var.change(image_upload_event, inputs=img_img2var, outputs=[width_img2var, height_img2var])
                        with gr.Column():
                            with gr.Row():
                                out_img2var = gr.Gallery(
                                    label="Generated images",
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
                                    download_btn_img2var = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_img2var = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_img2var.click(fn=zip_download_file_img2var, inputs=out_img2var, outputs=[download_file_img2var, download_file_img2var])                                
                    with gr.Row():
                        with gr.Column():
                            btn_img2var = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_img2var_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_img2var_cancel.click(fn=initiate_stop_img2var, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_img2var_clear_input = gr.ClearButton(components=[img_img2var], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_img2var_clear_output = gr.ClearButton(components=[out_img2var, gs_out_img2var], value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        img2var_llava = gr.Button(" >> Llava")
                                        img2var_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        img2var_img2img = gr.Button(" >> img2img")
                                        img2var_img2img_ip = gr.Button(" >> IP-Adapter")
                                        img2var_img2var = gr.Button(" >> Image variation")
                                        img2var_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        img2var_magicmix = gr.Button(" >> MagicMix")
                                        img2var_inpaint = gr.Button(" >> inpaint")
                                        img2var_paintbyex = gr.Button(" >> Paint by example") 
                                        img2var_outpaint = gr.Button(" >> outpaint")
                                        img2var_controlnet = gr.Button(" >> ControlNet")
                                        img2var_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        img2var_faceswap = gr.Button(" >> Faceswap target")
                                        img2var_resrgan = gr.Button(" >> Real ESRGAN")
                                        img2var_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        img2var_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        img2var_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                       

# pix2pix    
                with gr.TabItem("Instruct pix2pix", id=29) as tab_pix2pix:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Instruct pix2pix</br>
                                <b>Function : </b>Edit an input image with instructions from a prompt and a negative prompt using <a href='https://github.com/timothybrooks/instruct-pix2pix' target='_blank'>Instructpix2pix</a></br>
                                <b>Input(s) : </b>Input image, prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/timbrooks/instruct-pix2pix' target='_blank'>timbrooks/instruct-pix2pix</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image using the <b>Input image</b> field</br>
                                - Fill the <b>prompt</b> with the instructions for modifying your input image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to change image CFG scale or generate several images in a single run</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery</br></br>
                                <b>Examples : </b><a href='https://www.timothybrooks.com/instruct-pix2pix/' target='_blank'>InstructPix2Pix : Learning to Follow Image Editing Instructions</a>
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_pix2pix = gr.Dropdown(choices=model_list_pix2pix, value=model_list_pix2pix[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_pix2pix = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_pix2pix = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_pix2pix = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                image_guidance_scale_pix2pix = gr.Slider(0.0, 10.0, step=0.1, value=1.5, label="Img CFG Scale", info="Low values : more creativity. High values : more fidelity to the input image")
                            with gr.Column():
                                num_images_per_prompt_pix2pix = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_pix2pix = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_pix2pix = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_pix2pix = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_pix2pix = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_pix2pix = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_pix2pix = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_pix2pix = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_pix2pix = gr.Textbox(value="pix2pix", visible=False, interactive=False)
                                del_ini_btn_pix2pix = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_pix2pix.value) else False)
                                save_ini_btn_pix2pix.click(
                                    fn=write_ini, 
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
                                save_ini_btn_pix2pix.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_pix2pix.click(fn=lambda: del_ini_btn_pix2pix.update(interactive=True), outputs=del_ini_btn_pix2pix)
                                del_ini_btn_pix2pix.click(fn=lambda: del_ini(module_name_pix2pix.value))
                                del_ini_btn_pix2pix.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_pix2pix.click(fn=lambda: del_ini_btn_pix2pix.update(interactive=False), outputs=del_ini_btn_pix2pix)
                        if test_cfg_exist(module_name_pix2pix.value) :
                            readcfg_pix2pix = read_ini_pix2pix(module_name_pix2pix.value)
                            model_pix2pix.value = readcfg_pix2pix[0]
                            num_inference_step_pix2pix.value = readcfg_pix2pix[1]
                            sampler_pix2pix.value = readcfg_pix2pix[2]
                            guidance_scale_pix2pix.value = readcfg_pix2pix[3]
                            image_guidance_scale_pix2pix.value = readcfg_pix2pix[4]
                            num_images_per_prompt_pix2pix.value = readcfg_pix2pix[5]
                            num_prompt_pix2pix.value = readcfg_pix2pix[6]
                            width_pix2pix.value = readcfg_pix2pix[7]
                            height_pix2pix.value = readcfg_pix2pix[8]
                            seed_pix2pix.value = readcfg_pix2pix[9]
                            use_gfpgan_pix2pix.value = readcfg_pix2pix[10]
                            tkme_pix2pix.value = readcfg_pix2pix[11]
                    with gr.Row():
                        with gr.Column():
                             img_pix2pix = gr.Image(label="Input image", height=400, type="filepath")
                             img_pix2pix.change(image_upload_event, inputs=img_pix2pix, outputs=[width_pix2pix, height_pix2pix])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_pix2pix = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want to modify in your input image", placeholder="make it a Rembrandt painting")
                                with gr.Column():
                                    negative_prompt_pix2pix = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your output image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_pix2pix = gr.Gallery(
                                        label="Generated images",
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
                                            download_btn_pix2pix = gr.Button("Zip gallery 💾")
                                        with gr.Column():
                                            download_file_pix2pix = gr.File(label="Output", height=30, interactive=False, visible=False)
                                            download_btn_pix2pix.click(fn=zip_download_file_pix2pix, inputs=out_pix2pix, outputs=[download_file_pix2pix, download_file_pix2pix])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_pix2pix = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_pix2pix_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_pix2pix_cancel.click(fn=initiate_stop_pix2pix, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_pix2pix_clear_input = gr.ClearButton(components=[img_pix2pix, prompt_pix2pix, negative_prompt_pix2pix], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_pix2pix_clear_output = gr.ClearButton(components=[out_pix2pix, gs_out_pix2pix], value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        pix2pix_llava = gr.Button(" >> Llava")
                                        pix2pix_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        pix2pix_img2img = gr.Button(" >> img2img")
                                        pix2pix_img2img_ip = gr.Button(" >> IP-Adapter")
                                        pix2pix_img2var = gr.Button(" >> Image variation")
                                        pix2pix_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        pix2pix_magicmix = gr.Button(" >> MagicMix")
                                        pix2pix_inpaint = gr.Button(" >> inpaint")
                                        pix2pix_paintbyex = gr.Button(" >> Paint by example") 
                                        pix2pix_outpaint = gr.Button(" >> outpaint")
                                        pix2pix_controlnet = gr.Button(" >> ControlNet")
                                        pix2pix_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        pix2pix_faceswap = gr.Button(" >> Faceswap target")
                                        pix2pix_resrgan = gr.Button(" >> Real ESRGAN")
                                        pix2pix_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        pix2pix_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        pix2pix_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')                                        
                                        pix2pix_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        pix2pix_txt2img_kd_input = gr.Button(" >> Kandinsky")                                        
                                        pix2pix_txt2img_lcm_input = gr.Button(" >> LCM")
                                        pix2pix_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        pix2pix_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        pix2pix_img2img_input = gr.Button(" >> img2img")
                                        pix2pix_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        pix2pix_inpaint_input = gr.Button(" >> inpaint")
                                        pix2pix_controlnet_input = gr.Button(" >> ControlNet")
                                        pix2pix_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                        
                                        pix2pix_vid2vid_ze_input = gr.Button(" >> Video Instruct-pix2pix")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')                                        
                                        pix2pix_img2img_both = gr.Button(" +  >> img2img")
                                        pix2pix_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        pix2pix_inpaint_both = gr.Button(" +  >> inpaint")
                                        pix2pix_controlnet_both = gr.Button(" +  >> ControlNet")
                                        pix2pix_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")
# magicmix    
                with gr.TabItem("MagicMix", id=291) as tab_magicmix:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>MagicMix</br>
                                <b>Function : </b>Edit an input image with instructions from a prompt using <a href='https://magicmix.github.io/' target='_blank'>MagicMix</a> and <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>Input(s) : </b>Input image, prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>,
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>,
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image using the <b>Input image</b> field</br>
                                - Set the <b>Mix Factor</b> field to create a balance between input image and prompt</br>
                                - Fill the <b>prompt</b> with the instructions for modifying your input image. Use simple prompt instruction (e.g. "a dog")</br>
                                - (optional) Modify the settings to generate several images in a single run or generate several images in a single run</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery</br></br>
                                <b>Examples : </b><a href='https://magicmix.github.io/' target='_blank'>MagicMix: Semantic Mixing with Diffusion Models</a>
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_magicmix = gr.Dropdown(choices=model_list_magicmix, value=model_list_magicmix[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_magicmix = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_magicmix = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[1], label="Sampler", info="Sampler to use for inference", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_magicmix = gr.Slider(0.0, 20.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                kmin_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=0.3, label="Kmin", info="Controls the number of steps during the content generation process")
                            with gr.Column():
                                kmax_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Kmax", info="Determines how much information is kept in the layout of the original image")
                        with gr.Row():
                            with gr.Column():
                                num_prompt_magicmix = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                            with gr.Column():
                                seed_magicmix = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_magicmix = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_magicmix = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_magicmix = gr.Textbox(value="magicmix", visible=False, interactive=False)
                                del_ini_btn_magicmix = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_magicmix.value) else False)
                                save_ini_btn_magicmix.click(
                                    fn=write_ini, 
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
                                save_ini_btn_magicmix.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_magicmix.click(fn=lambda: del_ini_btn_magicmix.update(interactive=True), outputs=del_ini_btn_magicmix)
                                del_ini_btn_magicmix.click(fn=lambda: del_ini(module_name_magicmix.value))
                                del_ini_btn_magicmix.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_magicmix.click(fn=lambda: del_ini_btn_magicmix.update(interactive=False), outputs=del_ini_btn_magicmix)
                        if test_cfg_exist(module_name_magicmix.value) :
                            readcfg_magicmix = read_ini_magicmix(module_name_magicmix.value)
                            model_magicmix.value = readcfg_magicmix[0]
                            num_inference_step_magicmix.value = readcfg_magicmix[1]
                            sampler_magicmix.value = readcfg_magicmix[2]
                            guidance_scale_magicmix.value = readcfg_magicmix[3]
                            kmin_magicmix.value = readcfg_magicmix[4]
                            kmax_magicmix.value = readcfg_magicmix[5]
                            num_prompt_magicmix.value = readcfg_magicmix[6]
                            seed_magicmix.value = readcfg_magicmix[7]
                            use_gfpgan_magicmix.value = readcfg_magicmix[8]
                            tkme_magicmix.value = readcfg_magicmix[9]
                    with gr.Row():
                        with gr.Column():
                             img_magicmix = gr.Image(label="Input image", height=400, type="filepath")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    mix_factor_magicmix = gr.Slider(0.0, 1.0, step=0.01, value=0.5, label="Mix Factor", info="Determines how much influence the prompt has on the layout generation")
                            with gr.Row(): 
                                with gr.Column():
                                    prompt_magicmix = gr.Textbox(lines=9, max_lines=9, label="Prompt", info="Describe how you want to modify your input image", placeholder="a bed")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_magicmix = gr.Gallery(
                                        label="Generated images",
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
                                            download_btn_magicmix = gr.Button("Zip gallery 💾")
                                        with gr.Column():
                                            download_file_magicmix = gr.File(label="Output", height=30, interactive=False, visible=False)
                                            download_btn_magicmix.click(fn=zip_download_file_magicmix, inputs=out_magicmix, outputs=[download_file_magicmix, download_file_magicmix])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_magicmix = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_magicmix_clear_input = gr.ClearButton(components=[img_magicmix, prompt_magicmix], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_magicmix_clear_output = gr.ClearButton(components=[out_magicmix, gs_out_magicmix], value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        magicmix_llava = gr.Button(" >> Llava")
                                        magicmix_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        magicmix_img2img = gr.Button(" >> img2img")
                                        magicmix_img2img_ip = gr.Button(" >> IP-Adapter")
                                        magicmix_img2var = gr.Button(" >> Image variation")
                                        magicmix_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        magicmix_magicmix = gr.Button(" >> MagicMix")
                                        magicmix_inpaint = gr.Button(" >> inpaint")
                                        magicmix_paintbyex = gr.Button(" >> Paint by example") 
                                        magicmix_outpaint = gr.Button(" >> outpaint")
                                        magicmix_controlnet = gr.Button(" >> ControlNet")
                                        magicmix_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        magicmix_faceswap = gr.Button(" >> Faceswap target")
                                        magicmix_resrgan = gr.Button(" >> Real ESRGAN")
                                        magicmix_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        magicmix_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        magicmix_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                       
# inpaint    
                with gr.TabItem("inpaint", id=292) as tab_inpaint:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Inpaint</br>
                                <b>Function : </b>Inpaint the masked area of an input image, from a prompt and a negative prompt using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>Input(s) : </b>Input image, inpaint masked area, prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/Uminosachi/realisticVisionV30_v30VAE-inpainting' target='_blank'>Uminosachi/realisticVisionV30_v30VAE-inpainting</a> ,
                                <a href='https://huggingface.co/Uminosachi/diffusers/stable-diffusion-xl-1.0-inpainting-0.1' target='_blank'>diffusers/stable-diffusion-xl-1.0-inpainting-0.1</a> ,
                                <a href='https://huggingface.co/runwayml/stable-diffusion-inpainting' target='_blank'>runwayml/stable-diffusion-inpainting</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image using the <b>Input image</b> field</br>
                                - Using the sketch tool of the <b>inpaint field</b>, mask the area to be modified</br>
                                - Modify the <b>denoising strength of the inpainted area</b> : 0 will keep the original content, 1 will ignore it</br>
                                - Fill <b>the prompt</b> with what you want to see in your WHOLE (not only the inpaint area) output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to use another model or generate several images in a single run</br>
                                - Click the <b>Generate button</b></br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.
                                </div>
                                """
                            )                   
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_inpaint = gr.Dropdown(choices=model_list_inpaint, value=model_list_inpaint[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_inpaint = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_inpaint = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_inpaint = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_inpaint= gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_inpaint = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_inpaint = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_inpaint = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_inpaint = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_inpaint = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_inpaint = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_inpaint = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_inpaint = gr.Textbox(value="inpaint", visible=False, interactive=False)
                                del_ini_btn_inpaint = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_inpaint.value) else False)
                                save_ini_btn_inpaint.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_inpaint.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_inpaint.click(fn=lambda: del_ini_btn_inpaint.update(interactive=True), outputs=del_ini_btn_inpaint)
                                del_ini_btn_inpaint.click(fn=lambda: del_ini(module_name_inpaint.value))
                                del_ini_btn_inpaint.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_inpaint.click(fn=lambda: del_ini_btn_inpaint.update(interactive=False), outputs=del_ini_btn_inpaint)
                        if test_cfg_exist(module_name_inpaint.value) :
                            readcfg_inpaint = read_ini_inpaint(module_name_inpaint.value)
                            model_inpaint.value = readcfg_inpaint[0]
                            num_inference_step_inpaint.value = readcfg_inpaint[1]
                            sampler_inpaint.value = readcfg_inpaint[2]
                            guidance_scale_inpaint.value = readcfg_inpaint[3]
                            num_images_per_prompt_inpaint.value = readcfg_inpaint[4]
                            num_prompt_inpaint.value = readcfg_inpaint[5]
                            width_inpaint.value = readcfg_inpaint[6]
                            height_inpaint.value = readcfg_inpaint[7]
                            seed_inpaint.value = readcfg_inpaint[8]
                            use_gfpgan_inpaint.value = readcfg_inpaint[9]
                            tkme_inpaint.value = readcfg_inpaint[10]
                    with gr.Row():
                        with gr.Column(scale=2):
                             rotation_img_inpaint = gr.Number(value=0, visible=False)
                             img_inpaint = gr.Image(label="Input image", type="pil", height=400, tool="sketch")
                             img_inpaint.upload(image_upload_event_inpaint_c, inputs=[img_inpaint, model_inpaint], outputs=[width_inpaint, height_inpaint, img_inpaint, rotation_img_inpaint], preprocess=False)
                             gs_img_inpaint = gr.Image(type="pil", visible=False)
                             gs_img_inpaint.change(image_upload_event_inpaint_b, inputs=gs_img_inpaint, outputs=[width_inpaint, height_inpaint], preprocess=False)
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_inpaint = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label="Denoising strength", info="Balance between input image (0) and prompts (1)")                                
                            with gr.Row():
                                with gr.Column():
                                    prompt_inpaint = gr.Textbox(lines=3, max_lines=3, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                                with gr.Column():
                                    negative_prompt_inpaint = gr.Textbox(lines=3, max_lines=3, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    out_inpaint = gr.Gallery(
                                        label="Generated images",
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
                                            download_btn_inpaint = gr.Button("Zip gallery 💾")
                                        with gr.Column():
                                            download_file_inpaint = gr.File(label="Output", height=30, interactive=False, visible=False)
                                            download_btn_inpaint.click(fn=zip_download_file_inpaint, inputs=out_inpaint, outputs=[download_file_inpaint, download_file_inpaint])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_inpaint = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_inpaint_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_inpaint_cancel.click(fn=initiate_stop_inpaint, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_inpaint_clear_input = gr.ClearButton(components=[img_inpaint, gs_img_inpaint, prompt_inpaint, negative_prompt_inpaint], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_inpaint_clear_output = gr.ClearButton(components=[out_inpaint, gs_out_inpaint], value="Clear outputs 🧹")  
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
                                ],
                                outputs=[out_inpaint, gs_out_inpaint], 
                                show_progress="full",
                            )  
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        inpaint_llava = gr.Button(" >> Llava")
                                        inpaint_img2txt_git = gr.Button(" >> GIT Captioning")      
                                        gr.HTML(value='... image module ...')
                                        inpaint_img2img = gr.Button(" >> img2img")
                                        inpaint_img2img_ip = gr.Button(" >> IP-Adapter")
                                        inpaint_img2var = gr.Button(" >> Image variation")
                                        inpaint_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        inpaint_magicmix = gr.Button(" >> MagicMix")
                                        inpaint_inpaint = gr.Button(" >> inpaint")
                                        inpaint_paintbyex = gr.Button(" >> Paint by example") 
                                        inpaint_outpaint = gr.Button(" >> outpaint")
                                        inpaint_controlnet = gr.Button(" >> ControlNet")
                                        inpaint_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        inpaint_faceswap = gr.Button(" >> Faceswap target")
                                        inpaint_resrgan = gr.Button(" >> Real ESRGAN")
                                        inpaint_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        inpaint_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        inpaint_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        inpaint_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        inpaint_txt2img_kd_input = gr.Button(" >> Kandinsky") 
                                        inpaint_txt2img_lcm_input = gr.Button(" >> LCM") 
                                        inpaint_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        inpaint_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        inpaint_img2img_input = gr.Button(" >> img2img")
                                        inpaint_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        inpaint_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        inpaint_controlnet_input = gr.Button(" >> ControlNet")
                                        inpaint_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    
                                        gr.HTML(value='... image module ...')                                        
                                        inpaint_img2img_both = gr.Button(" +  >> img2img")
                                        inpaint_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        inpaint_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        inpaint_controlnet_both = gr.Button(" +  >> ControlNet")
                                        inpaint_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")

# paintbyex    
                if ram_size() >= 16 :
                    titletab_paintbyex = "Paint by example"
                else :
                    titletab_paintbyex = "Paint by example ⛔"

                with gr.TabItem(titletab_paintbyex, id=293) as tab_paintbyex: 
                    with gr.Accordion("About", open=False): 
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Paint by example</br>
                                <b>Function : </b>Paint the masked area of an input image, from an example image using  <a href='https://github.com/Fantasy-Studio/Paint-by-Example' target='_blank'>Paint by example</a>  and <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>Input(s) : </b>Input image, masked area, example image</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/Fantasy-Studio/Paint-by-Example' target='_blank'>Fantasy-Studio/Paint-by-Example</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image using the <b>Input image</b> field</br>
                                - Using the sketch tool of the <b>Input image field</b>, mask the area to be modified</br>
                                - Upload or import an example image using the <b>Example image</b> field. This image will be used as an example on how to modify the masked area of the input image</br>
                                - (optional) Modify the settings to generate several images in a single run</br>
                                - Click the <b>Generate button</b></br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                </div>
                                """
                            )                   
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_paintbyex = gr.Dropdown(choices=model_list_paintbyex, value=model_list_paintbyex[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_paintbyex = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_paintbyex = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_paintbyex = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_paintbyex= gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_paintbyex = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_paintbyex = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_paintbyex = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_paintbyex = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_paintbyex = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_paintbyex = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_paintbyex = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_paintbyex = gr.Textbox(value="paintbyex", visible=False, interactive=False)
                                del_ini_btn_paintbyex = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_paintbyex.value) else False)
                                save_ini_btn_paintbyex.click(
                                    fn=write_ini, 
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
                                save_ini_btn_paintbyex.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_paintbyex.click(fn=lambda: del_ini_btn_paintbyex.update(interactive=True), outputs=del_ini_btn_paintbyex)
                                del_ini_btn_paintbyex.click(fn=lambda: del_ini(module_name_paintbyex.value))
                                del_ini_btn_paintbyex.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_paintbyex.click(fn=lambda: del_ini_btn_paintbyex.update(interactive=False), outputs=del_ini_btn_paintbyex)
                        if test_cfg_exist(module_name_paintbyex.value) :
                            readcfg_paintbyex = read_ini_paintbyex(module_name_paintbyex.value)
                            model_paintbyex.value = readcfg_paintbyex[0]
                            num_inference_step_paintbyex.value = readcfg_paintbyex[1]
                            sampler_paintbyex.value = readcfg_paintbyex[2]
                            guidance_scale_paintbyex.value = readcfg_paintbyex[3]
                            num_images_per_prompt_paintbyex.value = readcfg_paintbyex[4]
                            num_prompt_paintbyex.value = readcfg_paintbyex[5]
                            width_paintbyex.value = readcfg_paintbyex[6]
                            height_paintbyex.value = readcfg_paintbyex[7]
                            seed_paintbyex.value = readcfg_paintbyex[8]
                            use_gfpgan_paintbyex.value = readcfg_paintbyex[9]
                            tkme_paintbyex.value = readcfg_paintbyex[10]
                    with gr.Row():
                        with gr.Column(scale=2):
                             rotation_img_paintbyex = gr.Number(value=0, visible=False)
                             img_paintbyex = gr.Image(label="Input image", type="pil", height=400, tool="sketch")
                             img_paintbyex.upload(image_upload_event_inpaint, inputs=img_paintbyex, outputs=[width_paintbyex, height_paintbyex, img_paintbyex, rotation_img_paintbyex], preprocess=False)
                             gs_img_paintbyex = gr.Image(type="pil", visible=False)
                             gs_img_paintbyex.change(image_upload_event_inpaint_b, inputs=gs_img_paintbyex, outputs=[width_paintbyex, height_paintbyex], preprocess=False)
                        with gr.Column():
                             example_img_paintbyex = gr.Image(label="Example image", type="pil", height=400)
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    out_paintbyex = gr.Gallery(
                                        label="Generated images",
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
                                            download_btn_paintbyex = gr.Button("Zip gallery 💾")
                                        with gr.Column():
                                            download_file_paintbyex = gr.File(label="Output", height=30, interactive=False, visible=False)
                                            download_btn_paintbyex.click(fn=zip_download_file_paintbyex, inputs=out_paintbyex, outputs=[download_file_paintbyex, download_file_paintbyex])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_paintbyex = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_paintbyex_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_paintbyex_cancel.click(fn=initiate_stop_paintbyex, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_paintbyex_clear_input = gr.ClearButton(components=[img_paintbyex, gs_img_paintbyex, example_img_paintbyex], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_paintbyex_clear_output = gr.ClearButton(components=[out_paintbyex, gs_out_paintbyex], value="Clear outputs 🧹")  
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        paintbyex_llava = gr.Button(" >> Llava")
                                        paintbyex_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        paintbyex_img2img = gr.Button(" >> img2img")
                                        paintbyex_img2img_ip = gr.Button(" >> IP-Adapter")
                                        paintbyex_img2var = gr.Button(" >> Image variation") 
                                        paintbyex_pix2pix = gr.Button(" >> Instruct pix2pix") 
                                        paintbyex_magicmix = gr.Button(" >> MagicMix")
                                        paintbyex_inpaint = gr.Button(" >> inpaint") 
                                        paintbyex_paintbyex = gr.Button(" >> Paint by example") 
                                        paintbyex_outpaint = gr.Button(" >> outpaint")
                                        paintbyex_controlnet = gr.Button(" >> ControlNet")
                                        paintbyex_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        paintbyex_faceswap = gr.Button(" >> Faceswap target")
                                        paintbyex_resrgan = gr.Button(" >> Real ESRGAN")
                                        paintbyex_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        paintbyex_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        paintbyex_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box(): 
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    
# outpaint    
                if ram_size() >= 16 :
                    titletab_outpaint = "outpaint"
                else :
                    titletab_outpaint = "outpaint ⛔"

                with gr.TabItem(titletab_outpaint, id=294) as tab_outpaint:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>outpaint</br>
                                <b>Function : </b>Outpaint an input image, by defining borders and using a prompt and a negative prompt, with <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a></br>
                                <b>Input(s) : </b>Input image, outpaint mask, prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/Uminosachi/realisticVisionV30_v30VAE-inpainting' target='_blank'>Uminosachi/realisticVisionV30_v30VAE-inpainting</a> ,
                                <a href='https://huggingface.co/Uminosachi/diffusers/stable-diffusion-xl-1.0-inpainting-0.1' target='_blank'>diffusers/stable-diffusion-xl-1.0-inpainting-0.1</a> ,
                                <a href='https://huggingface.co/runwayml/stable-diffusion-inpainting' target='_blank'>runwayml/stable-diffusion-inpainting</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image using the <b>Input image</b> field</br>
                                - Define the size in pixels of the borders to add for top, bottom, left and right sides 
                                - Click the <b>Create mask</b> button to add borders to your image and generate a mask</br>
                                - Modify the <b>denoising strength of the outpainted area</b> : 0 will keep the original content, 1 will ignore it</br>
                                - Fill <b>the prompt</b> with what you want to see in your WHOLE (not only the outpaint area) output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to use another model or generate several images in a single run</br>
                                - Click the <b>Generate button</b></br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.
                                </div>
                                """
                            )                   
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_outpaint = gr.Dropdown(choices=model_list_outpaint, value=model_list_outpaint[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_outpaint = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_outpaint = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_outpaint = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_outpaint= gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_outpaint = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_outpaint = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_outpaint = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_outpaint = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_outpaint = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_outpaint = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_outpaint = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_outpaint = gr.Textbox(value="outpaint", visible=False, interactive=False)
                                del_ini_btn_outpaint = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_outpaint.value) else False)
                                save_ini_btn_outpaint.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_outpaint.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_outpaint.click(fn=lambda: del_ini_btn_outpaint.update(interactive=True), outputs=del_ini_btn_outpaint)
                                del_ini_btn_outpaint.click(fn=lambda: del_ini(module_name_outpaint.value))
                                del_ini_btn_outpaint.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_outpaint.click(fn=lambda: del_ini_btn_outpaint.update(interactive=False), outputs=del_ini_btn_outpaint)
                        if test_cfg_exist(module_name_outpaint.value) :
                            readcfg_outpaint = read_ini_outpaint(module_name_outpaint.value)
                            model_outpaint.value = readcfg_outpaint[0]
                            num_inference_step_outpaint.value = readcfg_outpaint[1]
                            sampler_outpaint.value = readcfg_outpaint[2]
                            guidance_scale_outpaint.value = readcfg_outpaint[3]
                            num_images_per_prompt_outpaint.value = readcfg_outpaint[4]
                            num_prompt_outpaint.value = readcfg_outpaint[5]
                            width_outpaint.value = readcfg_outpaint[6]
                            height_outpaint.value = readcfg_outpaint[7]
                            seed_outpaint.value = readcfg_outpaint[8]
                            use_gfpgan_outpaint.value = readcfg_outpaint[9]
                            tkme_outpaint.value = readcfg_outpaint[10]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                rotation_img_outpaint = gr.Number(value=0, visible=False)
                                img_outpaint = gr.Image(label="Input image", type="pil", height=350)
                                gs_img_outpaint = gr.Image(type="pil", visible=False)
                                gs_img_outpaint.change(image_upload_event_inpaint_b, inputs=gs_img_outpaint, outputs=[width_outpaint, height_outpaint], preprocess=False)
                            with gr.Column():
                                with gr.Row():									
                                    top_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label="Top", info="Pixels to add on top")
                                    bottom_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label="Bottom", info="Pixels to add on bottom")
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column(): 
                                        btn_outpaint_preview = gr.Button("Create mask")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                                                            
                                    mask_outpaint = gr.Image(label="Mask preview", height=350, type="pil")
                                    gs_mask_outpaint = gr.Image(type="pil", visible=False)
                                    scale_preview_outpaint = gr.Number(value=2048, visible=False)
                                    mask_outpaint.upload(fn=scale_image, inputs=[mask_outpaint, scale_preview_outpaint], outputs=[width_outpaint, height_outpaint, mask_outpaint])
                                    gs_mask_outpaint.change(image_upload_event_inpaint_b, inputs=gs_mask_outpaint, outputs=[width_outpaint, height_outpaint], preprocess=False)
                                with gr.Column():
                                    with gr.Row():
                                        left_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label="Left", info="Pixels to add on left")
                                        right_outpaint = gr.Number(minimum=0, maximum=1024, step=1, value=256, label="Right", info="Pixels to add on right")
                                        btn_outpaint_preview.click(fn=prepare_outpaint, inputs=[img_outpaint, top_outpaint, bottom_outpaint, left_outpaint, right_outpaint], outputs=[img_outpaint, gs_img_outpaint, mask_outpaint, gs_mask_outpaint], show_progress="full")
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_outpaint = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label="Denoising strength", info="Balance between input image (0) and prompts (1)")                                
                            with gr.Row():
                                with gr.Column():
                                    prompt_outpaint = gr.Textbox(lines=3, max_lines=3, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                                with gr.Column():
                                    negative_prompt_outpaint = gr.Textbox(lines=3, max_lines=3, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    out_outpaint = gr.Gallery(
                                        label="Generated images",
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
                                            download_btn_outpaint = gr.Button("Zip gallery 💾")
                                        with gr.Column():
                                            download_file_outpaint = gr.File(label="Output", height=30, interactive=False, visible=False)
                                            download_btn_outpaint.click(fn=zip_download_file_outpaint, inputs=out_outpaint, outputs=[download_file_outpaint, download_file_outpaint])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_outpaint = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_outpaint_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_outpaint_cancel.click(fn=initiate_stop_outpaint, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_outpaint_clear_input = gr.ClearButton(components=[img_outpaint, gs_img_outpaint, prompt_outpaint, negative_prompt_outpaint], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_outpaint_clear_output = gr.ClearButton(components=[out_outpaint, gs_out_outpaint], value="Clear outputs 🧹")  
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
                                ],
                                outputs=[out_outpaint, gs_out_outpaint], 
                                show_progress="full",
                            )  
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        outpaint_llava = gr.Button(" >> Llava")
                                        outpaint_img2txt_git = gr.Button(" >> GIT Captioning")      
                                        gr.HTML(value='... image module ...')
                                        outpaint_img2img = gr.Button(" >> img2img")
                                        outpaint_img2img_ip = gr.Button(" >> IP-Adapter")
                                        outpaint_img2var = gr.Button(" >> Image variation")
                                        outpaint_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        outpaint_magicmix = gr.Button(" >> MagicMix")
                                        outpaint_inpaint = gr.Button(" >> inpaint")
                                        outpaint_paintbyex = gr.Button(" >> Paint by example") 
                                        outpaint_outpaint = gr.Button(" >> outpaint")
                                        outpaint_controlnet = gr.Button(" >> ControlNet")
                                        outpaint_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        outpaint_faceswap = gr.Button(" >> Faceswap target")
                                        outpaint_resrgan = gr.Button(" >> Real ESRGAN")
                                        outpaint_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        outpaint_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        outpaint_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        outpaint_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        outpaint_txt2img_kd_input = gr.Button(" >> Kandinsky")                                        
                                        outpaint_txt2img_lcm_input = gr.Button(" >> LCM")
                                        outpaint_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        outpaint_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        outpaint_img2img_input = gr.Button(" >> img2img")
                                        outpaint_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        outpaint_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        outpaint_controlnet_input = gr.Button(" >> ControlNet")
                                        outpaint_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    
                                        gr.HTML(value='... image module ...')                                        
                                        outpaint_img2img_both = gr.Button(" +  >> img2img")
                                        outpaint_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        outpaint_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        outpaint_controlnet_both = gr.Button(" +  >> ControlNet")
                                        outpaint_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")
# ControlNet
                with gr.TabItem("ControlNet", id=295) as tab_controlnet:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>ControlNet</br>
                                <b>Function : </b>Generate images from a prompt, a negative prompt and a control image using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> and <a href='https://stablediffusionweb.com/ControlNet' target='_blank'>ControlNet</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt, ControlNet input</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF Stable Diffusion models pages : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/stabilityai/sd-turbo' target='_blank'>stabilityai/sd-turbo</a>, 
                                <a href='https://huggingface.co/stabilityai/sdxl-turbo' target='_blank'>stabilityai/sdxl-turbo</a>, 
                                <a href='https://huggingface.co/thibaud/sdxl_dpo_turbo' target='_blank'>thibaud/sdxl_dpo_turbo</a>, 
                                <a href='https://huggingface.co/dataautogpt3/OpenDalleV1.1' target='_blank'>dataautogpt3/OpenDalleV1.1</a>, 
                                <a href='https://huggingface.co/dataautogpt3/ProteusV0.4' target='_blank'>dataautogpt3/ProteusV0.4</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/segmind/Segmind-Vega' target='_blank'>segmind/Segmind-Vega</a>, 
                                <a href='https://huggingface.co/segmind/SSD-1B' target='_blank'>segmind/SSD-1B</a>, 
                                <a href='https://huggingface.co/gsdf/Counterfeit-V2.5' target='_blank'>gsdf/Counterfeit-V2.5</a>, 
                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0' target='_blank'>stabilityai/stable-diffusion-xl-refiner-1.0</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></br>
                                <b>HF ControlNet models pages : </b>
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_canny' target='_blank'>lllyasviel/control_v11p_sd15_canny</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth' target='_blank'>lllyasviel/control_v11f1p_sd15_depth</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime' target='_blank'>lllyasviel/control_v11p_sd15s2_lineart_anime</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_lineart' target='_blank'>lllyasviel/control_v11p_sd15_lineart</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_mlsd' target='_blank'>lllyasviel/control_v11p_sd15_mlsd</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae' target='_blank'>lllyasviel/control_v11p_sd15_normalbae</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_openpose' target='_blank'>lllyasviel/control_v11p_sd15_openpose</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_scribble' target='_blank'>lllyasviel/control_v11p_sd15_scribble</a>, 
                                <a href='https://huggingface.co/lllyasviel/control_v11p_sd15_softedge' target='_blank'>lllyasviel/control_v11p_sd15_softedge</a>
                                </br>
                                """
#                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>, 
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model, change the settings for ControlNet or adjust threshold on canny</br>
                                - (optional) Select a LoRA model and set its weight</br>                                     
                                - Select a <b>Source image</b> that will be used to generate the control image</br>
                                - Select a <b>pre-processor</b> for the control image</br> 
                                - Click the <b>Preview</b> button</br>
                                - If the <b>Control image</b> generated suits your needs, continue. Else, you could modify the settings and generate a new one</br> 
                                - You should not modifiy the value in the <b>ControlNet Model</b> field, as it is automatically selected from the used pre-processor</br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - Click the <b>Generate button</b></br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery</br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory ./biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.</br>
                                <b>LoRA models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors LoRA models in the directory ./biniou/models/lora/SD or ./biniou/models/lora/SDXL (depending on the LoRA model type : SD 1.5 or SDXL). Restart Pixify to see them in the models list.</br>
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_controlnet = gr.Dropdown(choices=model_list_controlnet, value=model_list_controlnet[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_controlnet = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_controlnet = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_controlnet = gr.Slider(0.0, 20.0, step=0.1, value=7.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_controlnet = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_controlnet = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_controlnet = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_controlnet = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_controlnet = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")    
                        with gr.Row():
                            with gr.Column():
                                low_threshold_controlnet = gr.Slider(0, 255, step=1, value=100, label="Canny low threshold", info="ControlNet Low threshold")  
                            with gr.Column():
                                high_threshold_controlnet = gr.Slider(0, 255, step=1, value=200, label="Canny high threshold", info="ControlNet high threshold")
                            with gr.Column():                                    
                                strength_controlnet = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="ControlNet strength", info="ControlNet strength")
                        with gr.Row():
                            with gr.Column():
                                start_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=0.0, label="Start ControlNet", info="Start ControlNet at % step")
                            with gr.Column():
                                stop_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label="Stop ControlNet", info="Stop ControlNet at % step")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_controlnet = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_controlnet = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_controlnet = gr.Textbox(value="controlnet", visible=False, interactive=False)
                                del_ini_btn_controlnet = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_controlnet.value) else False)
                                save_ini_btn_controlnet.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_controlnet.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_controlnet.click(fn=lambda: del_ini_btn_controlnet.update(interactive=True), outputs=del_ini_btn_controlnet)
                                del_ini_btn_controlnet.click(fn=lambda: del_ini(module_name_controlnet.value))
                                del_ini_btn_controlnet.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_controlnet.click(fn=lambda: del_ini_btn_controlnet.update(interactive=False), outputs=del_ini_btn_controlnet)
                        if test_cfg_exist(module_name_controlnet.value) :
                            readcfg_controlnet = read_ini_controlnet(module_name_controlnet.value)
                            model_controlnet.value = readcfg_controlnet[0]
                            num_inference_step_controlnet.value = readcfg_controlnet[1]
                            sampler_controlnet.value = readcfg_controlnet[2]
                            guidance_scale_controlnet.value = readcfg_controlnet[3]
                            num_images_per_prompt_controlnet.value = readcfg_controlnet[4]
                            num_prompt_controlnet.value = readcfg_controlnet[5]
                            width_controlnet.value = readcfg_controlnet[6]
                            height_controlnet.value = readcfg_controlnet[7]
                            seed_controlnet.value = readcfg_controlnet[8]
                            low_threshold_controlnet.value = readcfg_controlnet[9]
                            high_threshold_controlnet.value = readcfg_controlnet[10]
                            strength_controlnet.value = readcfg_controlnet[11]
                            start_controlnet.value = readcfg_controlnet[12]
                            stop_controlnet.value = readcfg_controlnet[13]
                            use_gfpgan_controlnet.value = readcfg_controlnet[14]
                            tkme_controlnet.value = readcfg_controlnet[15]
                        with gr.Accordion("LoRA Model", open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_controlnet = gr.Dropdown(choices=list(lora_model_list(model_controlnet.value).keys()), value="", label="LoRA model", info="Choose LoRA model to use for inference")
                                with gr.Column():
                                    lora_weight_controlnet = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="LoRA weight", info="Weight of the LoRA model in the final result")                            
                        with gr.Accordion("Textual inversion", open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_controlnet = gr.Dropdown(choices=list(txtinv_list(model_controlnet.value).keys()), value="", label="Textual inversion", info="Choose textual inversion to use for inference")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    img_source_controlnet = gr.Image(label="Source image", height=250, type="filepath")
                                    img_source_controlnet.change(fn=image_upload_event, inputs=img_source_controlnet, outputs=[width_controlnet, height_controlnet])                                            
                                    gs_img_source_controlnet = gr.Image(type="pil", visible=False)
                                    gs_img_source_controlnet.change(fn=image_upload_event, inputs=gs_img_source_controlnet, outputs=[width_controlnet, height_controlnet], preprocess=False)                                    
                            with gr.Row():
                                with gr.Column(): 
                                    preprocessor_controlnet = gr.Dropdown(choices=preprocessor_list_controlnet, value=preprocessor_list_controlnet[0], label="Pre-processor", info="Choose pre-processor to use")
                                    btn_controlnet_preview = gr.Button("Preview")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                                                            
                                    img_preview_controlnet = gr.Image(label="Control image preview", height=250, type="filepath")
                                    gs_img_preview_controlnet = gr.Image(type="pil", visible=False)
                            with gr.Row():
                                with gr.Column():
                                    variant_controlnet = gr.Dropdown(choices=variant_list_controlnet, value=variant_list_controlnet[0], label="ControlNet Model", info="Choose ControlNet model to use")                                    
                                    gs_variant_controlnet = gr.Textbox(visible=False)
                                    gs_variant_controlnet.change(fn=in_and_out, inputs=gs_variant_controlnet, outputs=variant_controlnet) 
                                    scale_preview_controlnet = gr.Number(value=2048, visible=False)
                                    img_preview_controlnet.upload(fn=scale_image, inputs=[img_preview_controlnet, scale_preview_controlnet], outputs=[width_controlnet, height_controlnet, img_preview_controlnet])
                                    gs_img_preview_controlnet.change(image_upload_event_inpaint_b, inputs=gs_img_preview_controlnet, outputs=[width_controlnet, height_controlnet], preprocess=False)
                                    btn_controlnet_preview.click(fn=dispatch_controlnet_preview, inputs=[model_controlnet, low_threshold_controlnet, high_threshold_controlnet, img_source_controlnet, preprocessor_controlnet], outputs=[img_preview_controlnet, gs_img_preview_controlnet, gs_variant_controlnet], show_progress="full")
                            with gr.Row():
                                with gr.Column(): 
                                    btn_controlnet_clear_preview = gr.ClearButton(components=[img_preview_controlnet, gs_img_preview_controlnet], value="Clear preview 🧹")  
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_controlnet = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_controlnet = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
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
                            lora_model_controlnet.change(fn=change_lora_model_controlnet, inputs=[model_controlnet, lora_model_controlnet, prompt_controlnet], outputs=[prompt_controlnet])
                            txtinv_controlnet.change(fn=change_txtinv_controlnet, inputs=[model_controlnet, txtinv_controlnet, prompt_controlnet, negative_prompt_controlnet], outputs=[prompt_controlnet, negative_prompt_controlnet])
                        with gr.Column():
                            out_controlnet = gr.Gallery(
                                label="Generated images",
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
                                    download_btn_controlnet = gr.Button("Zip gallery 💾")
                                with gr.Column():
                                    download_file_controlnet = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_controlnet.click(fn=zip_download_file_controlnet, inputs=out_controlnet, outputs=[download_file_controlnet, download_file_controlnet])
                    with gr.Row():
                        with gr.Column():
                            btn_controlnet = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_controlnet_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_controlnet_cancel.click(fn=initiate_stop_controlnet, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_controlnet_clear_input = gr.ClearButton(components=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, img_preview_controlnet], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_controlnet_clear_output = gr.ClearButton(components=[out_controlnet, gs_out_controlnet], value="Clear outputs 🧹")   
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
                                variant_controlnet, 
                                img_preview_controlnet,
                                nsfw_filter,
                                tkme_controlnet,
                                lora_model_controlnet,
                                lora_weight_controlnet,
                                txtinv_controlnet,
                            ],
                                outputs=[out_controlnet, gs_out_controlnet],
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')                                        
                                        controlnet_llava = gr.Button(" >> Llava")
                                        controlnet_img2txt_git = gr.Button(" >> GIT Captioning")         
                                        gr.HTML(value='... image module ...')
                                        controlnet_img2img = gr.Button(" >> img2img")
                                        controlnet_img2img_ip = gr.Button(" >> IP-Adapter")
                                        controlnet_img2var = gr.Button(" >> Image variation")
                                        controlnet_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        controlnet_magicmix = gr.Button(" >> MagicMix")
                                        controlnet_inpaint = gr.Button(" >> inpaint")
                                        controlnet_paintbyex = gr.Button(" >> Paint by example") 
                                        controlnet_outpaint = gr.Button(" >> outpaint")
                                        controlnet_controlnet = gr.Button(" >> ControlNet")
                                        controlnet_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        controlnet_faceswap = gr.Button(" >> Faceswap target")
                                        controlnet_resrgan = gr.Button(" >> Real ESRGAN")
                                        controlnet_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        controlnet_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        controlnet_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        controlnet_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        controlnet_txt2img_kd_input = gr.Button(" >> Kandinsky")                                        
                                        controlnet_txt2img_lcm_input = gr.Button(" >> LCM") 
                                        controlnet_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        controlnet_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        controlnet_img2img_input = gr.Button(" >> img2img")
                                        controlnet_img2img_ip_input = gr.Button(" >> IP-Adapter")
                                        controlnet_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        controlnet_inpaint_input = gr.Button(" >> inpaint")
                                        controlnet_faceid_ip_input = gr.Button(" >> IP-Adapter FaceID")
                                        gr.HTML(value='... video module ...')                                        
                                        controlnet_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        controlnet_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")
                                        controlnet_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')                                                                            
                                        controlnet_img2img_both = gr.Button(" +  >> img2img")
                                        controlnet_img2img_ip_both = gr.Button(" +  >> IP-Adapter")
                                        controlnet_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        controlnet_inpaint_both = gr.Button(" +  >> inpaint")
                                        controlnet_faceid_ip_both = gr.Button(" +  >> IP-Adapter FaceID")


# faceid_ip
                with gr.TabItem("IP-Adapter FaceID", id=296) as tab_faceid_ip:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>IP-Adapter Faceid</br>
                                <b>Function : </b>Generate portraits using the face taken from the input image, a prompt and a negative prompt using <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a>, <a href='https://huggingface.co/h94/IP-Adapter-FaceID' target='_blank'>IP-Adapter FaceID</a> and <a href='https://github.com/deepinsight/insightface' target='_blank'>Insight face</a>.</br>
                                <b>Input(s) : </b>Input image, prompt, negative prompt</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/gsdf/Counterfeit-V2.5' target='_blank'>gsdf/Counterfeit-V2.5</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a>
                                """
#                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model, generate several images in a single run</br>
                                - (optional) Select a LoRA model and set its weight</br>
                                - Upload or import an image using the <b>Input image</b> field</br>
                                - Set the the FaceID strength : lower values give more creativity to the portrait, higher values more fidelity to the input image.</br>
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.</br>
                                <b>LoRA models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors LoRA models in the directory ./biniou/models/lora/SD or ./biniou/models/lora/SDXL (depending on the LoRA model type : SD 1.5 or SDXL). Restart Biniou to see them in the models list.</br>
                                </div>
                                """
                            )               
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_faceid_ip = gr.Dropdown(choices=model_list_faceid_ip, value=model_list_faceid_ip[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_faceid_ip = gr.Slider(2, biniou_global_steps_max, step=1, value=25, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
#                                sampler_faceid_ip = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                                sampler_faceid_ip = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value="DDIM", label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_faceid_ip = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_faceid_ip = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_faceid_ip = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_faceid_ip = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_faceid_ip = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_faceid_ip = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_faceid_ip = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_faceid_ip = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")    
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_faceid_ip = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_faceid_ip = gr.Textbox(value="faceid_ip", visible=False, interactive=False)
                                del_ini_btn_faceid_ip = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_faceid_ip.value) else False)
                                save_ini_btn_faceid_ip.click(
                                    fn=write_ini, 
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
                                        ]
                                    )
                                save_ini_btn_faceid_ip.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_faceid_ip.click(fn=lambda: del_ini_btn_faceid_ip.update(interactive=True), outputs=del_ini_btn_faceid_ip)
                                del_ini_btn_faceid_ip.click(fn=lambda: del_ini(module_name_faceid_ip.value))
                                del_ini_btn_faceid_ip.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_faceid_ip.click(fn=lambda: del_ini_btn_faceid_ip.update(interactive=False), outputs=del_ini_btn_faceid_ip)
                        if test_cfg_exist(module_name_faceid_ip.value) :
                            readcfg_faceid_ip = read_ini_faceid_ip(module_name_faceid_ip.value)
                            model_faceid_ip.value = readcfg_faceid_ip[0]
                            num_inference_step_faceid_ip.value = readcfg_faceid_ip[1]
                            sampler_faceid_ip.value = readcfg_faceid_ip[2]
                            guidance_scale_faceid_ip.value = readcfg_faceid_ip[3]
                            num_images_per_prompt_faceid_ip.value = readcfg_faceid_ip[4]
                            num_prompt_faceid_ip.value = readcfg_faceid_ip[5]
                            width_faceid_ip.value = readcfg_faceid_ip[6]
                            height_faceid_ip.value = readcfg_faceid_ip[7]
                            seed_faceid_ip.value = readcfg_faceid_ip[8]
                            use_gfpgan_faceid_ip.value = readcfg_faceid_ip[9]
                            tkme_faceid_ip.value = readcfg_faceid_ip[10]
                        with gr.Accordion("LoRA Model", open=True):
                            with gr.Row():
                                with gr.Column():
                                    lora_model_faceid_ip = gr.Dropdown(choices=list(lora_model_list(model_faceid_ip.value).keys()), value="", label="LoRA model", info="Choose LoRA model to use for inference")
                                with gr.Column():
                                    lora_weight_faceid_ip = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="LoRA weight", info="Weight of the LoRA model in the final result")
                        with gr.Accordion("Textual inversion", open=True):
                            with gr.Row():
                                with gr.Column():
                                    txtinv_faceid_ip = gr.Dropdown(choices=list(txtinv_list(model_faceid_ip.value).keys()), value="", label="Textual inversion", info="Choose textual inversion to use for inference")
                    with gr.Row():
                        with gr.Column():
                            img_faceid_ip = gr.Image(label="Input image", height=400, type="filepath")
                            scale_preview_faceid_ip = gr.Number(value=512, visible=False)
                            img_faceid_ip.upload(fn=scale_image_any, inputs=[img_faceid_ip, scale_preview_faceid_ip], outputs=[img_faceid_ip])
#                            img_faceid_ip.change(image_upload_event, inputs=img_faceid_ip, outputs=[width_faceid_ip, height_faceid_ip])
                        with gr.Column():
                            with gr.Row(): 
                                with gr.Column():
                                    denoising_strength_faceid_ip = gr.Slider(0.01, 2.0, step=0.01, value=1.0, label="FaceID strength", info="Weight of the FaceID in the generated image")  
                            with gr.Row():
                                with gr.Column():
                                    prompt_faceid_ip = gr.Textbox(lines=5, max_lines=5, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():                                    
                                with gr.Column():
                                    negative_prompt_faceid_ip = gr.Textbox(lines=5, max_lines=5, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
                        model_faceid_ip.change(
                            fn=change_model_type_faceid_ip, 
                            inputs=[model_faceid_ip],
                            outputs=[
                                sampler_faceid_ip,
                                width_faceid_ip,
                                height_faceid_ip,
                                num_inference_step_faceid_ip,
                                guidance_scale_faceid_ip,
                                lora_model_faceid_ip,
                                txtinv_faceid_ip,
                                negative_prompt_faceid_ip,
                            ]
                        )
                        lora_model_faceid_ip.change(fn=change_lora_model_faceid_ip, inputs=[model_faceid_ip, lora_model_faceid_ip, prompt_faceid_ip], outputs=[prompt_faceid_ip])
                        txtinv_faceid_ip.change(fn=change_txtinv_faceid_ip, inputs=[model_faceid_ip, txtinv_faceid_ip, prompt_faceid_ip, negative_prompt_faceid_ip], outputs=[prompt_faceid_ip, negative_prompt_faceid_ip])
#                        denoising_strength_faceid_ip.change(check_steps_strength, [num_inference_step_faceid_ip, denoising_strength_faceid_ip, model_faceid_ip], [num_inference_step_faceid_ip])
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                            
                                    out_faceid_ip = gr.Gallery(
                                        label="Generated images",
                                        show_label=True,
                                        elem_id="gallery_i2i",
                                        columns=2,
                                        height=400,
                                        preview=True,
                                )
                                gs_out_faceid_ip = gr.State()
                                sel_out_faceid_ip = gr.Number(precision=0, visible=False)                              
                                out_faceid_ip.select(get_select_index, None, sel_out_faceid_ip)
                                with gr.Row():
                                    with gr.Column():
                                        download_btn_faceid_ip = gr.Button("Zip gallery 💾")
                                    with gr.Column():
                                        download_file_faceid_ip = gr.File(label="Output", height=30, interactive=False, visible=False)
                                        download_btn_faceid_ip.click(fn=zip_download_file_faceid_ip, inputs=out_faceid_ip, outputs=[download_file_faceid_ip, download_file_faceid_ip])
                    with gr.Row():
                        with gr.Column():
                            btn_faceid_ip = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_faceid_ip_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_faceid_ip_cancel.click(fn=initiate_stop_faceid_ip, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_faceid_ip_clear_input = gr.ClearButton(components=[img_faceid_ip, prompt_faceid_ip, negative_prompt_faceid_ip], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_faceid_ip_clear_output = gr.ClearButton(components=[out_faceid_ip, gs_out_faceid_ip], value="Clear outputs 🧹")
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
                                    lora_model_faceid_ip,
                                    lora_weight_faceid_ip,
                                    txtinv_faceid_ip,
                                ],
                                outputs=[out_faceid_ip, gs_out_faceid_ip], 
                                show_progress="full",
                            )  
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        faceid_ip_llava = gr.Button(" >> Llava")
                                        faceid_ip_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        faceid_ip_img2img = gr.Button(" >> img2img")
                                        faceid_ip_img2img_ip = gr.Button(" >> IP-Adapter")
                                        faceid_ip_img2var = gr.Button(" >> Image variation")
                                        faceid_ip_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        faceid_ip_inpaint = gr.Button(" >> inpaint")
                                        faceid_ip_magicmix = gr.Button(" >> MagicMix")
                                        faceid_ip_paintbyex = gr.Button(" >> Paint by example") 
                                        faceid_ip_outpaint = gr.Button(" >> outpaint")
                                        faceid_ip_controlnet = gr.Button(" >> ControlNet")
                                        faceid_ip_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        faceid_ip_faceswap = gr.Button(" >> Faceswap target")
                                        faceid_ip_resrgan = gr.Button(" >> Real ESRGAN")
                                        faceid_ip_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        faceid_ip_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...')
                                        faceid_ip_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        faceid_ip_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        faceid_ip_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        faceid_ip_txt2img_lcm_input = gr.Button(" >> LCM")
                                        faceid_ip_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        faceid_ip_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        faceid_ip_pix2pix_input = gr.Button(" >> Instruct pix2pix")
                                        faceid_ip_inpaint_input = gr.Button(" >> inpaint")
                                        faceid_ip_controlnet_input = gr.Button(" >> ControlNet")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        faceid_ip_pix2pix_both = gr.Button(" +  >> Instruct pix2pix")
                                        faceid_ip_inpaint_both = gr.Button(" +  >> inpaint")
                                        faceid_ip_controlnet_both = gr.Button(" +  >> ControlNet")

# faceswap    
                with gr.TabItem("Faceswap", id=297) as tab_faceswap:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Faceswap</br>
                                <b>Function : </b>Swap faces between images (source -> target) using <a href='https://github.com/deepinsight/insightface' target='_blank'>Insight Face</a> et <a href='https://github.com/microsoft/onnxruntime' target='_blank'>Onnx runtime</a></br>
                                <b>Input(s) : </b>Source image, target image</br>
                                <b>Output(s) : </b>Image(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/thebiglaskowski/inswapper_128.onnx' target='_blank'>thebiglaskowski/inswapper_128.onnx</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload a <b>Source image</b>. The face(s) in this image will replaces face(s) in the target image.</br>
                                - Upload or import a <b>target image</b>. The face(s) in this image will be replaced with the source one(s)</br>
                                - Set the <b>source index</b> list to choose which face(s) to extract from source image and in which order. From left to right and starting from 0, id comma separated list of faces number. For example, if there is 3 faces in a picture '0,2' will select the face on the left, then on the right, but not on the one in the middle. If set to 0, take only the first face from the left.</br>
                                - Set the <b>target index</b> list to choose which face(s) to replace in target image and in which order. From left to right and starting from 0, id comma separated list of faces number. For example, if there is 3 faces in a picture '2,1' will select the faces on the right, then in the middle, but not the one on the left. The source index list is used to create a mapping between the faces to extract and to replace. If set to 0, replace only the first face from the left.</br>
                                - (optional) Modify the settings to desactivate GFPGAN faces restoration</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_faceswap = gr.Dropdown(choices=list(model_list_faceswap.keys()), value=list(model_list_faceswap.keys())[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                width_faceswap = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_faceswap = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_faceswap = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")    
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_faceswap = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_faceswap = gr.Textbox(value="faceswap", visible=False, interactive=False)
                                del_ini_btn_faceswap = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_faceswap.value) else False)
                                save_ini_btn_faceswap.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_faceswap, 
                                        model_faceswap, 
                                        width_faceswap,
                                        height_faceswap,
                                        use_gfpgan_faceswap,
                                        ]
                                    )
                                save_ini_btn_faceswap.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_faceswap.click(fn=lambda: del_ini_btn_faceswap.update(interactive=True), outputs=del_ini_btn_faceswap)
                                del_ini_btn_faceswap.click(fn=lambda: del_ini(module_name_faceswap.value))
                                del_ini_btn_faceswap.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_faceswap.click(fn=lambda: del_ini_btn_faceswap.update(interactive=False), outputs=del_ini_btn_faceswap)
                        if test_cfg_exist(module_name_faceswap.value) :
                            readcfg_faceswap = read_ini_faceswap(module_name_faceswap.value)
                            model_faceswap.value = readcfg_faceswap[0]
                            width_faceswap.value = readcfg_faceswap[1]
                            height_faceswap.value = readcfg_faceswap[2]
                            use_gfpgan_faceswap.value = readcfg_faceswap[3]
                    with gr.Row():
                        with gr.Column():
                            img_source_faceswap = gr.Image(label="Source image", height=400, type="filepath")
                            scale_preview_faceswap = gr.Number(value=512, visible=False)
                            img_source_faceswap.upload(fn=scale_image_any, inputs=[img_source_faceswap, scale_preview_faceswap], outputs=[img_source_faceswap])
                            with gr.Row():
                                source_index_faceswap = gr.Textbox(value=0, lines=1, label="Source index", info="Use a comma separated list of faces to export to target (numbers from left to right)")
                        with gr.Column():
                             img_target_faceswap = gr.Image(label="Target image", type="filepath", height=400)
                             gs_img_target_faceswap = gr.Image(type="pil", visible=False)
                             img_target_faceswap.change(image_upload_event, inputs=img_target_faceswap, outputs=[width_faceswap, height_faceswap])
                             with gr.Row():
                                 target_index_faceswap = gr.Textbox(value=0, lines=1, label="Target index", info="Use a comma separated list of faces to replace in target (numbers from left to right)")                             
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_faceswap = gr.Gallery(
                                        label="Generated images",
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
                                            download_btn_faceswap = gr.Button("Zip gallery 💾")
                                        with gr.Column():
                                            download_file_faceswap = gr.File(label="Output", height=30, interactive=False, visible=False)
                                            download_btn_faceswap.click(fn=zip_download_file_faceswap, inputs=out_faceswap, outputs=[download_file_faceswap, download_file_faceswap])                                       
                    with gr.Row():
                        with gr.Column():
                            btn_faceswap = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_faceswap_clear_input = gr.ClearButton(components=[img_source_faceswap, img_target_faceswap, source_index_faceswap, target_index_faceswap, gs_img_target_faceswap], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_faceswap_clear_output = gr.ClearButton(components=[out_faceswap, gs_out_faceswap], value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')                                        
                                        faceswap_llava = gr.Button(" >> Llava")
                                        faceswap_img2txt_git = gr.Button(" >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        faceswap_img2img = gr.Button(" >> img2img")
                                        faceswap_img2img_ip = gr.Button(" >> IP-Adapter")
                                        faceswap_img2var = gr.Button(" >> Image variation")
                                        faceswap_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        faceswap_magicmix = gr.Button(" >> MagicMix")
                                        faceswap_inpaint = gr.Button(" >> inpaint")
                                        faceswap_paintbyex = gr.Button(" >> Paint by example") 
                                        faceswap_outpaint = gr.Button(" >> outpaint")
                                        faceswap_controlnet = gr.Button(" >> ControlNet")
                                        faceswap_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        faceswap_faceswap = gr.Button(" >> Faceswap target")
                                        faceswap_resrgan = gr.Button(" >> Real ESRGAN")
                                        faceswap_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        faceswap_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        faceswap_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    

# Real ESRGAN    
                with gr.TabItem("Real ESRGAN", id=298) as tab_resrgan:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Real ESRGAN</br>
                                <b>Function : </b>Upscale x4 using <a href='https://github.com/xinntao/Real-ESRGAN' target='_blank'>Real ESRGAN</a></br>
                                <b>Input(s) : </b>Input image</br>
                                <b>Output(s) : </b>Upscaled image</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/ai-forever/Real-ESRGAN' target='_blank'>ai-forever/Real-ESRGAN</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an <b>Input image</b> </br>
                                - (optional) Modify the settings to change scale factor or use another model</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, upscaled image is displayed in the gallery.
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_resrgan = gr.Dropdown(choices=model_list_resrgan, value=model_list_resrgan[1], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                scale_resrgan = gr.Dropdown(choices=list(RESRGAN_SCALES.keys()), value=list(RESRGAN_SCALES.keys())[1], label="Upscale factor", info="Choose upscale factor")  
                                scale_resrgan.change(scale_resrgan_change, inputs=scale_resrgan, outputs=model_resrgan)                                
                        with gr.Row():
                            with gr.Column():
                                width_resrgan = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of input", interactive=False)
                            with gr.Column():
                                height_resrgan = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of input", interactive=False)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_resrgan = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")     
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_resrgan = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_resrgan = gr.Textbox(value="resrgan", visible=False, interactive=False)
                                del_ini_btn_resrgan = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_resrgan.value) else False)
                                save_ini_btn_resrgan.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_resrgan, 
                                        model_resrgan, 
                                        scale_resrgan,                                         
                                        width_resrgan,
                                        height_resrgan,
                                        use_gfpgan_resrgan,
                                        ]
                                    )
                                save_ini_btn_resrgan.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_resrgan.click(fn=lambda: del_ini_btn_resrgan.update(interactive=True), outputs=del_ini_btn_resrgan)
                                del_ini_btn_resrgan.click(fn=lambda: del_ini(module_name_resrgan.value))
                                del_ini_btn_resrgan.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_resrgan.click(fn=lambda: del_ini_btn_resrgan.update(interactive=False), outputs=del_ini_btn_resrgan)
                        if test_cfg_exist(module_name_resrgan.value) :
                            readcfg_resrgan = read_ini_resrgan(module_name_resrgan.value)
                            model_resrgan.value = readcfg_resrgan[0]
                            scale_resrgan.value = readcfg_resrgan[1]
                            width_resrgan.value = readcfg_resrgan[2]
                            height_resrgan.value = readcfg_resrgan[3]
                            use_gfpgan_resrgan.value = readcfg_resrgan[4]
                    with gr.Row():
                        with gr.Column():
                             img_resrgan = gr.Image(label="Input image", type="filepath", height=400)
                             img_resrgan.change(image_upload_event, inputs=img_resrgan, outputs=[width_resrgan, height_resrgan])
                        with gr.Column():
                            out_resrgan = gr.Gallery(
                                label="Generated image",
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
                            btn_resrgan = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_resrgan_clear_input = gr.ClearButton(components=[img_resrgan], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_resrgan_clear_output = gr.ClearButton(components=[out_resrgan, gs_out_resrgan], value="Clear outputs 🧹") 
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        resrgan_llava = gr.Button(" >> Llava")
                                        resrgan_img2txt_git = gr.Button(" >> GIT Captioning") 
                                        gr.HTML(value='... image module ...')
                                        resrgan_img2img = gr.Button(" >> img2img")
                                        resrgan_img2img_ip = gr.Button(" >> IP-Adapter")
                                        resrgan_img2var = gr.Button(" >> Image variation")
                                        resrgan_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        resrgan_magicmix = gr.Button(" >> MagicMix")
                                        resrgan_inpaint = gr.Button(" >> inpaint")
                                        resrgan_paintbyex = gr.Button(" >> Paint by example") 
                                        resrgan_outpaint = gr.Button(" >> outpaint")
                                        resrgan_controlnet = gr.Button(" >> ControlNet")
                                        resrgan_faceid_ip = gr.Button(" >> IP-Adapter FaceID")
                                        resrgan_faceswap = gr.Button(" >> Faceswap target")
                                        resrgan_gfpgan = gr.Button(" >> GFPGAN")
                                        gr.HTML(value='... Video module ...')
                                        resrgan_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        resrgan_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                         
# GFPGAN    
                with gr.TabItem("GFPGAN", id=299) as tab_gfpgan:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>GFPGAN</br>
                                <b>Function : </b>Restore and enhance faces in an image using <a href='https://github.com/TencentARC/GFPGAN' target='_blank'>GFPGAN</a></br>
                                <b>Input(s) : </b>Input image</br>
                                <b>Output(s) : </b>Enhanced Image</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/leonelhs/gfpgan' target='_blank'>leonelhs/gfpgan</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an <b>Input image</b></br>
                                - (optional) Modify the settings to use another variant of the GFPGAN model</br> 
                                - Click the <b>Generate</b> button</br>
                                - After generation, enhanced image is displayed in the gallery
                                </div>
                                """
                            )                     
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_gfpgan = gr.Dropdown(choices=model_list_gfpgan, value=model_list_gfpgan[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                variant_gfpgan = gr.Dropdown(choices=variant_list_gfpgan, value=variant_list_gfpgan[4], label="Variant", info="Variant of GPFGAN to use")
                        with gr.Row():
                            with gr.Column():
                                width_gfpgan = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_gfpgan = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_gfpgan = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_gfpgan = gr.Textbox(value="gfpgan", visible=False, interactive=False)
                                del_ini_btn_gfpgan = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_gfpgan.value) else False)
                                save_ini_btn_gfpgan.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_gfpgan, 
                                        model_gfpgan, 
                                        variant_gfpgan,
                                        width_gfpgan,
                                        height_gfpgan,
                                        ]
                                    )
                                save_ini_btn_gfpgan.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_gfpgan.click(fn=lambda: del_ini_btn_gfpgan.update(interactive=True), outputs=del_ini_btn_gfpgan)
                                del_ini_btn_gfpgan.click(fn=lambda: del_ini(module_name_gfpgan.value))
                                del_ini_btn_gfpgan.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_gfpgan.click(fn=lambda: del_ini_btn_gfpgan.update(interactive=False), outputs=del_ini_btn_gfpgan)
                        if test_cfg_exist(module_name_gfpgan.value) :
                            readcfg_gfpgan = read_ini_gfpgan(module_name_gfpgan.value)
                            model_gfpgan.value = readcfg_gfpgan[0]
                            variant_gfpgan.value = readcfg_gfpgan[1]
                            width_gfpgan.value = readcfg_gfpgan[2]
                            height_gfpgan.value = readcfg_gfpgan[3]
                    with gr.Row():
                        with gr.Column():
                            img_gfpgan = gr.Image(label="Input image", type="filepath", height=400)
                            img_gfpgan.change(image_upload_event, inputs=img_gfpgan, outputs=[width_gfpgan, height_gfpgan])
                        with gr.Column():
                            out_gfpgan = gr.Gallery(
                                label="Generated image",
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
                            btn_gfpgan = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_gfpgan_clear_input = gr.ClearButton(components=[img_gfpgan], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_gfpgan_clear_output = gr.ClearButton(components=[out_gfpgan, gs_out_gfpgan], value="Clear outputs 🧹") 
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')                                        
                                        gfpgan_llava = gr.Button(" >> Llava")
                                        gfpgan_img2txt_git = gr.Button(" >> GIT Captioning")   
                                        gr.HTML(value='... image module ...')
                                        gfpgan_img2img = gr.Button(" >> img2img")
                                        gfpgan_img2img_ip = gr.Button(" >> IP-Adapter")
                                        gfpgan_img2var = gr.Button(" >> Image variation")
                                        gfpgan_pix2pix = gr.Button(" >> Instruct pix2pix")
                                        gfpgan_magicmix = gr.Button(" >> MagicMix")
                                        gfpgan_inpaint = gr.Button(" >> inpaint")
                                        gfpgan_paintbyex = gr.Button(" >> Paint by example") 
                                        gfpgan_outpaint = gr.Button(" >> outpaint")
                                        gfpgan_controlnet = gr.Button(" >> ControlNet")
                                        gfpgan_faceid_ip = gr.Button(" >> Ip-Adapter FaceID")
                                        gfpgan_faceswap = gr.Button(" >> Faceswap target")
                                        gfpgan_resrgan = gr.Button(" >> Real ESRGAN")
                                        gr.HTML(value='... Video module ...')
                                        gfpgan_img2vid = gr.Button(" >> Stable Video Diffusion")
                                        gr.HTML(value='... 3d module ...') 
                                        gfpgan_img2shape = gr.Button(" >> Shap-E img2shape") 
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                        
# Audio
        with gr.TabItem("Aud Gen", id=3) as tab_audio:
            with gr.Tabs() as tabs_audio:        
# Musicgen
                with gr.TabItem("MusicGen", id=31) as tab_musicgen:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>MusicGen</br>
                                <b>Function : </b>Generate music from a prompt, using <a href='https://github.com/facebookresearch/audiocraft' target='_blank'>MusicGen</a></br>
                                <b>Input(s) : </b>Input prompt</br>
                                <b>Output(s) : </b>Generated music</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/facebook/musicgen-small' target='_blank'>facebook/musicgen-small</a>, 
                                <a href='https://huggingface.co/facebook/musicgen-medium' target='_blank'>facebook/musicgen-medium</a>, 
                                <a href='https://huggingface.co/facebook/musicgen-large' target='_blank'>facebook/musicgen-large</a>, 
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> by describing the music you want to generate</br>
                                - (optional) Modify the settings to use another model or change audio duration</br>                                
                                - Click the <b>Generate<b> button</br>
                                - After generation, generated music is available to listen in the <b>Generated music<b> field.
                                </div>
                                """
                            )                           
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_musicgen= gr.Dropdown(choices=modellist_musicgen, value=modellist_musicgen[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():    
                                duration_musicgen = gr.Slider(1, 160, step=1, value=5, label="Audio length (sec)")
                            with gr.Column():
                                cfg_coef_musicgen = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_batch_musicgen = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")  
                        with gr.Row():
                            with gr.Column():    
                                use_sampling_musicgen = gr.Checkbox(value=True, label="Use sampling")
                            with gr.Column():    
                                temperature_musicgen = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label="temperature")
                            with gr.Column():
                                top_k_musicgen = gr.Slider(0, 500, step=1, value=250, label="top_k")
                            with gr.Column():
                                top_p_musicgen = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label="top_p")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_musicgen = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_musicgen = gr.Textbox(value="musicgen", visible=False, interactive=False)
                                del_ini_btn_musicgen = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_musicgen.value) else False)
                                save_ini_btn_musicgen.click(
                                    fn=write_ini, 
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
                                save_ini_btn_musicgen.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_musicgen.click(fn=lambda: del_ini_btn_musicgen.update(interactive=True), outputs=del_ini_btn_musicgen)
                                del_ini_btn_musicgen.click(fn=lambda: del_ini(module_name_musicgen.value))
                                del_ini_btn_musicgen.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_musicgen.click(fn=lambda: del_ini_btn_musicgen.update(interactive=False), outputs=del_ini_btn_musicgen)
                        if test_cfg_exist(module_name_musicgen.value) :
                            readcfg_musicgen = read_ini_musicgen(module_name_musicgen.value)
                            model_musicgen.value = readcfg_musicgen[0]
                            duration_musicgen.value = readcfg_musicgen[1]
                            cfg_coef_musicgen.value = readcfg_musicgen[2]
                            num_batch_musicgen.value = readcfg_musicgen[3]
                            use_sampling_musicgen.value = readcfg_musicgen[4]
                            temperature_musicgen.value = readcfg_musicgen[5]
                            top_k_musicgen.value = readcfg_musicgen[6]
                            top_p_musicgen.value = readcfg_musicgen[7]
                    with gr.Row():
                        with gr.Column():
                            prompt_musicgen = gr.Textbox(label="Describe your music", lines=2, max_lines=2, placeholder="90s rock song with loud guitars and heavy drums")
                        with gr.Column():
                            out_musicgen = gr.Audio(label="Generated music", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_musicgen = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_musicgen_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_musicgen_cancel.click(fn=initiate_stop_musicgen, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_musicgen_clear_input = gr.ClearButton(components=prompt_musicgen, value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_musicgen_clear_output = gr.ClearButton(components=out_musicgen, value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... audio module ...')
                                        musicgen_musicgen_mel = gr.Button(" >> MusicGen Melody")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... audio module ...')                                        
                                        musicgen_musicgen_mel_input = gr.Button(" >> MusicGen Melody")
                                        musicgen_musicldm_input = gr.Button(" >> MusicLDM")
                                        musicgen_audiogen_input = gr.Button(" >> Audiogen")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    

# Musicgen Melody
                if ram_size() >= 16 :
                    titletab_musicgen_mel = "MusicGen Melody"
                else :
                    titletab_musicgen_mel = "MusicGen Melody ⛔"

                with gr.TabItem(titletab_musicgen_mel, id=32) as tab_musicgen_mel:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>MusicGen Melody</br>
                                <b>Function : </b>Generate music from a prompt with guidance from an input audio, using <a href='https://github.com/facebookresearch/audiocraft' target='_blank'>MusicGen</a></br>
                                <b>Input(s) : </b>Input prompt, Input audio</br>
                                <b>Output(s) : </b>Generated music</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/facebook/musicgen-melody' target='_blank'>facebook/musicgen-melody</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Select an audio source type (file or micro recording)</br>
                                - Select an audio source by choosing a file or recording something</br>
                                - Fill the <b>prompt</b> by describing the music you want to generate from the audio source</br>
                                - (optional) Modify the settings to change audio duration or inferences parameters</br>
                                - Click the <b>Generate<b> button</br>
                                - After generation, generated music is available to listen in the <b>Generated music<b> field.
                                </div>
                                """
                            )                           
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_musicgen_mel= gr.Dropdown(choices=modellist_musicgen_mel, value=modellist_musicgen_mel[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():    
                                duration_musicgen_mel = gr.Slider(1, 160, step=1, value=5, label="Audio length (sec)")
                            with gr.Column():
                                cfg_coef_musicgen_mel = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_batch_musicgen_mel = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")  
                        with gr.Row():
                            with gr.Column():    
                                use_sampling_musicgen_mel = gr.Checkbox(value=True, label="Use sampling")
                            with gr.Column():    
                                temperature_musicgen_mel = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label="temperature")
                            with gr.Column():
                                top_k_musicgen_mel = gr.Slider(0, 500, step=1, value=250, label="top_k")
                            with gr.Column():
                                top_p_musicgen_mel = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label="top_p")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_musicgen_mel = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_musicgen_mel = gr.Textbox(value="musicgen_mel", visible=False, interactive=False)
                                del_ini_btn_musicgen_mel = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_musicgen_mel.value) else False)
                                save_ini_btn_musicgen_mel.click(
                                    fn=write_ini, 
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
                                save_ini_btn_musicgen_mel.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_musicgen_mel.click(fn=lambda: del_ini_btn_musicgen_mel.update(interactive=True), outputs=del_ini_btn_musicgen_mel)
                                del_ini_btn_musicgen_mel.click(fn=lambda: del_ini(module_name_musicgen_mel.value))
                                del_ini_btn_musicgen_mel.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_musicgen_mel.click(fn=lambda: del_ini_btn_musicgen_mel.update(interactive=False), outputs=del_ini_btn_musicgen_mel)
                        if test_cfg_exist(module_name_musicgen_mel.value) :
                            readcfg_musicgen_mel = read_ini_musicgen_mel(module_name_musicgen_mel.value)
                            model_musicgen_mel.value = readcfg_musicgen_mel[0]
                            duration_musicgen_mel.value = readcfg_musicgen_mel[1]
                            cfg_coef_musicgen_mel.value = readcfg_musicgen_mel[2]
                            num_batch_musicgen_mel.value = readcfg_musicgen_mel[3]
                            use_sampling_musicgen_mel.value = readcfg_musicgen_mel[4]
                            temperature_musicgen_mel.value = readcfg_musicgen_mel[5]
                            top_k_musicgen_mel.value = readcfg_musicgen_mel[6]
                            top_p_musicgen_mel.value = readcfg_musicgen_mel[7]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():                                
                                source_type_musicgen_mel = gr.Radio(choices=["audio", "micro"], value="audio", label="Source audio type", info="Choose source audio type")
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            source_audio_musicgen_mel = gr.Audio(label="Source audio", source="upload", type="filepath")
                            source_type_musicgen_mel.change(fn=change_source_type_musicgen_mel, inputs=source_type_musicgen_mel, outputs=source_audio_musicgen_mel)
                        with gr.Column():
                            prompt_musicgen_mel = gr.Textbox(label="Describe your music", lines=8, max_lines=8, placeholder="90s rock song with loud guitars and heavy drums")
                        with gr.Column():
                            out_musicgen_mel = gr.Audio(label="Generated music", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_musicgen_mel = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_musicgen_mel_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_musicgen_mel_cancel.click(fn=initiate_stop_musicgen_mel, inputs=None, outputs=None)
                        with gr.Column():
                            btn_musicgen_mel_clear_input = gr.ClearButton(components=[prompt_musicgen_mel, source_audio_musicgen_mel], value="Clear inputs 🧹")
                        with gr.Column():
                            btn_musicgen_mel_clear_output = gr.ClearButton(components=out_musicgen_mel, value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... audio module ...')
                                        musicgen_mel_musicgen_mel = gr.Button(" >> MusicGen Melody")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... audio module ...')
                                        musicgen_mel_musicgen_input = gr.Button(" >> MusicGen")
                                        musicgen_mel_musicldm_input = gr.Button(" >> MusicLDM")
                                        musicgen_mel_audiogen_input = gr.Button(" >> Audiogen")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# MusicLDM
                with gr.TabItem("MusicLDM", id=33) as tab_musicldm:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>MusicLDM</br>
                                <b>Function : </b>Generate music from a prompt and a negative prompt, using <a href='https://musicldm.github.io' target='_blank'>MusicLDM</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Generated music</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/ucsd-reach/musicldm' target='_blank'>ucsd-reach/musicldm</a>, 
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> by describing the music you want to generate</br>
                                - Fill the <b>negative prompt</b> by describing what you DO NOT want to generate</br>
                                - (optional) Modify the settings to use another model or change audio duration</br>
                                - Click the <b>Generate<b> button</br>
                                - After generation, generated music is available to listen in the <b>Generated music<b> field.
                                </div>
                                """
                            )                           
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_musicldm = gr.Dropdown(choices=model_list_musicldm, value=model_list_musicldm[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_musicldm = gr.Slider(1, 400, step=1, value=50, label="Steps", info="Number of iterations per audio. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_musicldm = gr.Dropdown(choices=list(SCHEDULER_MAPPING_MUSICLDM.keys()), value=list(SCHEDULER_MAPPING_MUSICLDM.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_musicldm = gr.Slider(0.1, 20.0, step=0.1, value=2.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                audio_length_musicldm=gr.Slider(0, 160, step=1, value=10, label="Audio length", info="Duration of audio file to generate")
                            with gr.Column():
                                seed_musicldm = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")    
                        with gr.Row():
                            with gr.Column():
                                num_audio_per_prompt_musicldm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of audios to generate in a single run")
                            with gr.Column():
                                num_prompt_musicldm = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_musicldm = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_musicldm = gr.Textbox(value="musicldm", visible=False, interactive=False)
                                del_ini_btn_musicldm = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_musicldm.value) else False)
                                save_ini_btn_musicldm.click(
                                    fn=write_ini, 
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
                                save_ini_btn_musicldm.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_musicldm.click(fn=lambda: del_ini_btn_musicldm.update(interactive=True), outputs=del_ini_btn_musicldm)
                                del_ini_btn_musicldm.click(fn=lambda: del_ini(module_name_musicldm.value))
                                del_ini_btn_musicldm.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_musicldm.click(fn=lambda: del_ini_btn_musicldm.update(interactive=False), outputs=del_ini_btn_musicldm)
                        if test_cfg_exist(module_name_musicldm.value) :
                            readcfg_musicldm = read_ini_musicldm(module_name_musicldm.value)
                            model_musicldm.value = readcfg_musicldm[0]
                            num_inference_step_musicldm.value = readcfg_musicldm[1]
                            sampler_musicldm.value = readcfg_musicldm[2]
                            guidance_scale_musicldm.value = readcfg_musicldm[3]
                            audio_length_musicldm.value = readcfg_musicldm[4]
                            seed_musicldm.value = readcfg_musicldm[5]
                            num_audio_per_prompt_musicldm.value = readcfg_musicldm[6]
                            num_prompt_musicldm.value = readcfg_musicldm[7]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                prompt_musicldm = gr.Textbox(label="Prompt", lines=2, max_lines=2, info="Describe the content of your output audio file", placeholder="Techno music with a strong, upbeat tempo and high melodic riffs, high quality, clear")
                            with gr.Row():
                                negative_prompt_musicldm = gr.Textbox(label="Negative prompt", info="Describe what you DO NOT want in your output audio file", lines=2, max_lines=2, placeholder="low quality, average quality")
                        with gr.Column():
                            out_musicldm = gr.Audio(label="Generated music", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_musicldm = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_musicldm_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_musicldm_cancel.click(fn=initiate_stop_musicldm, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_musicldm_clear_input = gr.ClearButton(components=prompt_musicldm, value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_musicldm_clear_output = gr.ClearButton(components=out_musicldm, value="Clear outputs 🧹")
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... audio module ...')
                                        musicldm_musicgen_mel = gr.Button(" >> MusicGen Melody")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... audio module ...')                                        
                                        musicldm_musicgen_input = gr.Button(" >> MusicGen")
                                        musicldm_musicgen_mel_input = gr.Button(" >> MusicGen Melody")
                                        musicldm_audiogen_input = gr.Button(" >> Audiogen")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# Audiogen
                if ram_size() >= 16 :
                    titletab_audiogen = "AudioGen"
                else :
                    titletab_audiogen = "AudioGen ⛔"
                
                with gr.TabItem(titletab_audiogen, id=34) as tab_audiogen:

                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>AudioGen</br>
                                <b>Function : </b>Generate sound from a prompt, using <a href='https://github.com/facebookresearch/audiocraft' target='_blank'>Audiogen</a></br>
                                <b>Input(s) : </b>Input prompt</br>
                                <b>Output(s) : </b>Generated sound</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/facebook/audiogen-medium' target='_blank'>facebook/audiogen-medium</a></br>                                
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>Prompt</b> by describing the sound you want to generate</br>
                                - (optional) Modify the settings to change audio duration</br>                                
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated sound is available to listen in the <b>Generated sound</b> field.
                                </div>
                                """
                            )                       
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_audiogen= gr.Dropdown(choices=modellist_audiogen, value=modellist_audiogen[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():    
                                duration_audiogen = gr.Slider(1, 160, step=1, value=5, label="Audio length (sec)")
                            with gr.Column():
                                cfg_coef_audiogen = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_batch_audiogen = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")  
                        with gr.Row():
                            with gr.Column():    
                                use_sampling_audiogen = gr.Checkbox(value=True, label="Use sampling")
                            with gr.Column():    
                                temperature_audiogen = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label="temperature")
                            with gr.Column():
                                top_k_audiogen = gr.Slider(0, 500, step=1, value=250, label="top_k")
                            with gr.Column():
                                top_p_audiogen = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label="top_p")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_audiogen = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_audiogen = gr.Textbox(value="audiogen", visible=False, interactive=False)
                                del_ini_btn_audiogen = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_audiogen.value) else False)
                                save_ini_btn_audiogen.click(
                                    fn=write_ini, 
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
                                save_ini_btn_audiogen.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_audiogen.click(fn=lambda: del_ini_btn_audiogen.update(interactive=True), outputs=del_ini_btn_audiogen)
                                del_ini_btn_audiogen.click(fn=lambda: del_ini(module_name_audiogen.value))
                                del_ini_btn_audiogen.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_audiogen.click(fn=lambda: del_ini_btn_audiogen.update(interactive=False), outputs=del_ini_btn_audiogen)
                        if test_cfg_exist(module_name_audiogen.value) :
                            readcfg_audiogen = read_ini_audiogen(module_name_audiogen.value)
                            model_audiogen.value = readcfg_audiogen[0]
                            duration_audiogen.value = readcfg_audiogen[1]
                            cfg_coef_audiogen.value = readcfg_audiogen[2]
                            num_batch_audiogen.value = readcfg_audiogen[3]
                            use_sampling_audiogen.value = readcfg_audiogen[4]
                            temperature_audiogen.value = readcfg_audiogen[5]
                            top_k_audiogen.value = readcfg_audiogen[6]
                            top_p_audiogen.value = readcfg_audiogen[7]
                    with gr.Row():
                        with gr.Column():
                            prompt_audiogen = gr.Textbox(label="Describe your sound", lines=2, max_lines=2, placeholder="dog barking, sirens of an emergency vehicle, footsteps in a corridor")
                        with gr.Column():
                            out_audiogen = gr.Audio(label="Generated sound", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_audiogen = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_audiogen_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_audiogen_cancel.click(fn=initiate_stop_audiogen, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_audiogen_clear_input = gr.ClearButton(components=prompt_audiogen, value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_audiogen_clear_output = gr.ClearButton(components=out_audiogen, value="Clear outputs 🧹")                        
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... audio module ...')
                                        audiogen_musicgen_mel = gr.Button(" >> MusicGen Melody")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... audio module ...')
                                        audiogen_musicgen_input = gr.Button(" >> Musicgen")
                                        audiogen_musicgen_mel_input = gr.Button(" >> MusicGen Melody")
                                        audiogen_musicldm_input = gr.Button(" >> MusicLDM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    

# Harmonai
                with gr.TabItem("Harmonai", id=35) as tab_harmonai:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Harmonai</br>
                                <b>Function : </b>Generate audio from a specific model using <a href='https://www.harmonai.org/' target='_blank'>Harmonai</a></br>
                                <b>Input(s) : </b>None</br>
                                <b>Output(s) : </b>Generated audio</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/harmonai/glitch-440k' target='_blank'>harmonai/glitch-440k</a> ,
                                <a href='https://huggingface.co/harmonai/honk-140k' target='_blank'>harmonai/honk-140k</a> ,
                                <a href='https://huggingface.co/harmonai/jmann-small-190k' target='_blank'>harmonai/jmann-small-190k</a> ,
                                <a href='https://huggingface.co/harmonai/jmann-large-580k' target='_blank'>harmonai/jmann-large-580k</a> ,
                                <a href='https://huggingface.co/harmonai/maestro-150k' target='_blank'>harmonai/maestro-150k</a> ,
                                <a href='https://huggingface.co/harmonai/unlocked-250k' target='_blank'>harmonai/unlocked-250k</a></br>                                
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to change audio duration</br>                                
                                - Click the <b>Generate<b> button</br>
                                - After generation, generated audio is available to listen in the <b>Output<b> field.
                                </div>
                                """
                            )                                       
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_harmonai = gr.Dropdown(choices=model_list_harmonai, value=model_list_harmonai[4], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                steps_harmonai = gr.Slider(1, biniou_global_steps_max, step=1, value=50, label="Steps", info="Number of iterations per audio. Results and speed depends of sampler")
                            with gr.Column():
                                seed_harmonai = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():
                                length_harmonai = gr.Slider(1, 1200, value=5, step=1, label="Audio length (sec)")
                            with gr.Column():
                                batch_size_harmonai = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of audios to generate in a single run")
                            with gr.Column():
                                batch_repeat_harmonai = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_harmonai = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_harmonai = gr.Textbox(value="harmonai", visible=False, interactive=False)
                                del_ini_btn_harmonai = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_harmonai.value) else False)
                                save_ini_btn_harmonai.click(
                                    fn=write_ini, 
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
                                save_ini_btn_harmonai.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_harmonai.click(fn=lambda: del_ini_btn_harmonai.update(interactive=True), outputs=del_ini_btn_harmonai)
                                del_ini_btn_harmonai.click(fn=lambda: del_ini(module_name_harmonai.value))
                                del_ini_btn_harmonai.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_harmonai.click(fn=lambda: del_ini_btn_harmonai.update(interactive=False), outputs=del_ini_btn_harmonai)
                        if test_cfg_exist(module_name_harmonai.value) :
                            readcfg_harmonai = read_ini_harmonai(module_name_harmonai.value)
                            model_harmonai.value = readcfg_harmonai[0]
                            steps_harmonai.value = readcfg_harmonai[1]
                            seed_harmonai.value = readcfg_harmonai[2]
                            length_harmonai.value = readcfg_harmonai[3]
                            batch_size_harmonai.value = readcfg_harmonai[4]
                            batch_repeat_harmonai.value = readcfg_harmonai[5]
                    with gr.Row():
                        out_harmonai = gr.Audio(label="Output", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_harmonai = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_harmonai_clear_output = gr.ClearButton(components=out_harmonai, value="Clear outputs 🧹")                           
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... audio module ...')
                                        harmonai_musicgen_mel = gr.Button(" >> MusicGen Melody")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                       
# Bark
                with gr.TabItem("Bark 🗣️", id=36) as tab_bark:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Bark</br>
                                <b>Function : </b>Generate high quality text-to-speech in several languages with <a href='https://github.com/suno-ai/bark' target='_blank'>Bark</a></br>
                                <b>Input(s) : </b>Prompt</br>
                                <b>Output(s) : </b>Generated speech</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/suno/bark' target='_blank'>suno/bark</a> ,
                                <a href='https://huggingface.co/suno/bark-small' target='_blank'>suno/bark-small</a></br>               
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with the text you want to hear</br>                                
                                - (optional) Modify the settings to select a model and a voice</br>                                
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated audio is available to listen in the <b>Generated speech</b> field.</br>
                                <b>Tips : </b>You can add modifications to the generated voices, by adding the following in your prompts : 
                                [laughter]</br>
                                [laughs]</br>
                                [sighs]</br>
                                [music]</br>
                                [gasps]</br>
                                [clears throat]</br>
                                — or ... for hesitations</br>
                                ♪ for song lyrics</br>
                                CAPITALIZATION for emphasis of a word</br>
                                [MAN] and [WOMAN] to bias Bark toward male and female speakers, respectively</br>
                                </div>
                                """
                            )                                                       
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_bark = gr.Dropdown(choices=model_list_bark, value=model_list_bark[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                voice_preset_bark = gr.Dropdown(choices=list(voice_preset_list_bark.keys()), value=list(voice_preset_list_bark.keys())[2], label="Voice")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_bark = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_bark = gr.Textbox(value="bark", visible=False, interactive=False)
                                del_ini_btn_bark = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_bark.value) else False)
                                save_ini_btn_bark.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_bark, 
                                        model_bark, 
                                        voice_preset_bark,
                                        ]
                                    )
                                save_ini_btn_bark.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_bark.click(fn=lambda: del_ini_btn_bark.update(interactive=True), outputs=del_ini_btn_bark)
                                del_ini_btn_bark.click(fn=lambda: del_ini(module_name_bark.value))
                                del_ini_btn_bark.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_bark.click(fn=lambda: del_ini_btn_bark.update(interactive=False), outputs=del_ini_btn_bark)
                        if test_cfg_exist(module_name_bark.value):
                            readcfg_bark = read_ini_bark(module_name_bark.value)
                            model_bark.value = readcfg_bark[0]
                            voice_preset_bark.value = readcfg_bark[1]
                    with gr.Row():
                        with gr.Column():                    
                            prompt_bark = gr.Textbox(label="Text to speech", lines=2, max_lines=2, placeholder="Type or past here what you want to hear ...")
                        with gr.Column():
                            out_bark = gr.Audio(label="Generated speech", type="filepath", show_download_button=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_bark = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_bark_clear_input = gr.ClearButton(components=prompt_bark, value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_bark_clear_output = gr.ClearButton(components=out_bark, value="Clear outputs 🧹")                        
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... audio module ...')
                                        bark_musicgen_mel = gr.Button(" >> MusicGen Melody")
                                        gr.HTML(value='... text module ...')
                                        bark_whisper = gr.Button("🗣️ >> Whisper")                                      
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    
# Video
        with gr.TabItem("Vid Gen", id=4) as tab_video:
            with gr.Tabs() as tabs_video:
# Modelscope            
                if ram_size() >= 16 :
                    titletab_txt2vid_ms = "Modelscope "
                else :
                    titletab_txt2vid_ms = "Modelscope ⛔"
                    
                with gr.TabItem(titletab_txt2vid_ms, id=41) as tab_txt2vid_ms:                        
                        
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Modelscope</br>
                                <b>Function : </b>Generate video from a prompt and a negative prompt using <a href='https://github.com/modelscope/modelscope' target='_blank'>Modelscope</a></br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Video</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/cerspense/zeroscope_v2_576w' target='_blank'>cerspense/zeroscope_v2_576w</a>, 
                                <a href='https://huggingface.co/camenduru/potat1' target='_blank'>camenduru/potat1</a>, 
                                <a href='https://huggingface.co/damo-vilab/text-to-video-ms-1.7b' target='_blank'>damo-vilab/text-to-video-ms-1.7b</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with what you want to see in your output video</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output video</br>
                                - (optional) Modify the settings to use another model, modify the number of frames to generate, or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated video is displayed in the <b>Generated video</b> field.
                                </br>
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2vid_ms = gr.Dropdown(choices=model_list_txt2vid_ms, value=model_list_txt2vid_ms[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2vid_ms = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per video. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2vid_ms = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2vid_ms = gr.Slider(0.1, 20.0, step=0.1, value=4.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_frames_txt2vid_ms = gr.Slider(1, 1200, step=1, value=8, label="Video Length (frames)", info="Number of frames in the output video")
                            with gr.Column():
                                num_prompt_txt2vid_ms = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2vid_ms = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=576, label="Video Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2vid_ms = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=320, label="Video Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2vid_ms = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2vid_ms = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2vid_ms = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2vid_ms = gr.Textbox(value="txt2vid_ms", visible=False, interactive=False)
                                del_ini_btn_txt2vid_ms = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2vid_ms.value) else False)
                                save_ini_btn_txt2vid_ms.click(
                                    fn=write_ini, 
                                    inputs=[
                                        module_name_txt2vid_ms, 
                                        model_txt2vid_ms, 
                                        num_inference_step_txt2vid_ms,
                                        sampler_txt2vid_ms,
                                        guidance_scale_txt2vid_ms,
                                        num_frames_txt2vid_ms,
                                        num_prompt_txt2vid_ms,
                                        width_txt2vid_ms,
                                        height_txt2vid_ms,
                                        seed_txt2vid_ms,
                                        use_gfpgan_txt2vid_ms,
                                        ]
                                    )
                                save_ini_btn_txt2vid_ms.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2vid_ms.click(fn=lambda: del_ini_btn_txt2vid_ms.update(interactive=True), outputs=del_ini_btn_txt2vid_ms)
                                del_ini_btn_txt2vid_ms.click(fn=lambda: del_ini(module_name_txt2vid_ms.value))
                                del_ini_btn_txt2vid_ms.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2vid_ms.click(fn=lambda: del_ini_btn_txt2vid_ms.update(interactive=False), outputs=del_ini_btn_txt2vid_ms)
                        if test_cfg_exist(module_name_txt2vid_ms.value) :
                            readcfg_txt2vid_ms = read_ini_txt2vid_ms(module_name_txt2vid_ms.value)
                            model_txt2vid_ms.value = readcfg_txt2vid_ms[0]
                            num_inference_step_txt2vid_ms.value = readcfg_txt2vid_ms[1]
                            sampler_txt2vid_ms.value = readcfg_txt2vid_ms[2]
                            guidance_scale_txt2vid_ms.value = readcfg_txt2vid_ms[3]
                            num_frames_txt2vid_ms.value = readcfg_txt2vid_ms[4]
                            num_prompt_txt2vid_ms.value = readcfg_txt2vid_ms[5]
                            width_txt2vid_ms.value = readcfg_txt2vid_ms[6]
                            height_txt2vid_ms.value = readcfg_txt2vid_ms[7]
                            seed_txt2vid_ms.value = readcfg_txt2vid_ms[8]
                            use_gfpgan_txt2vid_ms.value = readcfg_txt2vid_ms[9]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2vid_ms = gr.Textbox(lines=4, max_lines=4, label="Prompt", info="Describe what you want in your video", placeholder="Darth vader is surfing on waves, photo realistic, best quality")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2vid_ms = gr.Textbox(lines=4, max_lines=4, label="Negative Prompt", info="Describe what you DO NOT want in your video", placeholder="out of frame, ugly")
                        with gr.Column():
                            out_txt2vid_ms = gr.Video(label="Generated video", height=400, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_txt2vid_ms = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2vid_ms_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2vid_ms_cancel.click(fn=initiate_stop_txt2vid_ms, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2vid_ms_clear_input = gr.ClearButton(components=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2vid_ms_clear_output = gr.ClearButton(components=[out_txt2vid_ms], value="Clear outputs 🧹")                                                   
                            btn_txt2vid_ms.click(
                                fn=video_txt2vid_ms,
                                inputs=[
                                    model_txt2vid_ms,
                                    sampler_txt2vid_ms,
                                    prompt_txt2vid_ms,
                                    negative_prompt_txt2vid_ms,
                                    num_frames_txt2vid_ms,
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... video module ...')
                                        txt2vid_ms_vid2vid_ze = gr.Button(" >> Video Instruct-pix2pix")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2vid_ms_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        txt2vid_ms_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        txt2vid_ms_txt2img_lcm_input = gr.Button(" >> LCM")
                                        txt2vid_ms_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        txt2vid_ms_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        gr.HTML(value='... video module ...')
                                        txt2vid_ms_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")
                                        txt2vid_ms_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
# Txt2vid_zero            
                with gr.TabItem("Text2Video-Zero ", id=42) as tab_txt2vid_ze:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Text2Video-Zero</br>
                                <b>Function : </b>Generate video from a prompt and a negative prompt using <a href='https://github.com/Picsart-AI-Research/Text2Video-Zero' target='_blank'>Text2Video-Zero</a> with <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> Models</br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Video</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/stabilityai/sdxl-turbo' target='_blank'>stabilityai/sdxl-turbo</a>, 
                                <a href='https://huggingface.co/dataautogpt3/OpenDalleV1.1' target='_blank'>dataautogpt3/OpenDalleV1.1</a>, 
                                <a href='https://huggingface.co/dataautogpt3/ProteusV0.4' target='_blank'>dataautogpt3/ProteusV0.4</a>, 
                                <a href='https://huggingface.co/etri-vilab/koala-1b' target='_blank'>etri-vilab/koala-1b</a>, 
                                <a href='https://huggingface.co/etri-vilab/koala-700m' target='_blank'>etri-vilab/koala-700m</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/segmind/Segmind-Vega' target='_blank'>segmind/Segmind-Vega</a>, 
                                <a href='https://huggingface.co/segmind/segmind/SSD-1B' target='_blank'>segmind/SSD-1B</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></br>
                                """
#                                 <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>, 
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with what you want to see in your output video</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output video</br>
                                - (optional) Modify the settings to use another model, modify the number of frames to generate, fps of the output video or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated video is displayed in the <b>Generated video</b> field.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.
                                </div>
                                """
                            )                      
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2vid_ze = gr.Dropdown(choices=model_list_txt2vid_ze, value=model_list_txt2vid_ze[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2vid_ze = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per video. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2vid_ze = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                            with gr.Column():
                                guidance_scale_txt2vid_ze = gr.Slider(0.1, 20.0, step=0.1, value=7.5, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                        with gr.Row():
                            with gr.Column():
                                seed_txt2vid_ze = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                            with gr.Column():
                                num_frames_txt2vid_ze = gr.Slider(1, 1200, step=1, value=8, label="Video Length (frames)", info="Number of frames in the output video")
                            with gr.Column():
                                num_fps_txt2vid_ze = gr.Slider(1, 120, step=1, value=4, label="Frames per second", info="Number of frames per second")
                            with gr.Column():
                                num_chunks_txt2vid_ze = gr.Slider(1, 32, step=1, value=1, label="Chunk size", info="Number of frames processed in a chunk. 1 = no chunks.")
                        with gr.Row():
                            with gr.Column():
                                width_txt2vid_ze = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=576, label="Video Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2vid_ze = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=320, label="Video Height", info="Height of outputs")
                            with gr.Column():
                                num_videos_per_prompt_txt2vid_ze = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of videos to generate in a single run", interactive=False)
                            with gr.Column():
                                num_prompt_txt2vid_ze = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")                            
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    motion_field_strength_x_txt2vid_ze = gr.Slider(0, 50, step=1, value=12, label="Motion field strength x", info="Horizontal motion strength")
                                with gr.Column():
                                    motion_field_strength_y_txt2vid_ze = gr.Slider(0, 50, step=1, value=12, label="Motion field strength y", info="Vertical motion strength")
                                with gr.Column():
                                    timestep_t0_txt2vid_ze = gr.Slider(0, biniou_global_steps_max, step=1, value=7, label="Timestep t0", interactive=False)
                                with gr.Column():
                                    timestep_t1_txt2vid_ze = gr.Slider(1, biniou_global_steps_max, step=1, value=8, label="Timestep t1", interactive=False)
                                    num_inference_step_txt2vid_ze.change(set_timestep_vid_ze, inputs=[num_inference_step_txt2vid_ze, model_txt2vid_ze], outputs=[timestep_t0_txt2vid_ze, timestep_t1_txt2vid_ze])
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2vid_ze = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():    
                                tkme_txt2vid_ze = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token Merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2vid_ze = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2vid_ze = gr.Textbox(value="txt2vid_ze", visible=False, interactive=False)
                                del_ini_btn_txt2vid_ze = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2vid_ze.value) else False)
                                save_ini_btn_txt2vid_ze.click(
                                    fn=write_ini, 
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
                                save_ini_btn_txt2vid_ze.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2vid_ze.click(fn=lambda: del_ini_btn_txt2vid_ze.update(interactive=True), outputs=del_ini_btn_txt2vid_ze)
                                del_ini_btn_txt2vid_ze.click(fn=lambda: del_ini(module_name_txt2vid_ze.value))
                                del_ini_btn_txt2vid_ze.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2vid_ze.click(fn=lambda: del_ini_btn_txt2vid_ze.update(interactive=False), outputs=del_ini_btn_txt2vid_ze)
                        if test_cfg_exist(module_name_txt2vid_ze.value) :
                            readcfg_txt2vid_ze = read_ini_txt2vid_ze(module_name_txt2vid_ze.value)
                            model_txt2vid_ze.value = readcfg_txt2vid_ze[0]
                            num_inference_step_txt2vid_ze.value = readcfg_txt2vid_ze[1]
                            sampler_txt2vid_ze.value = readcfg_txt2vid_ze[2]
                            guidance_scale_txt2vid_ze.value = readcfg_txt2vid_ze[3]
                            seed_txt2vid_ze.value = readcfg_txt2vid_ze[4]
                            num_frames_txt2vid_ze.value = readcfg_txt2vid_ze[5]
                            num_fps_txt2vid_ze.value = readcfg_txt2vid_ze[6]
                            num_chunks_txt2vid_ze.value = readcfg_txt2vid_ze[7]
                            width_txt2vid_ze.value = readcfg_txt2vid_ze[8]
                            height_txt2vid_ze.value = readcfg_txt2vid_ze[9]
                            num_videos_per_prompt_txt2vid_ze.value = readcfg_txt2vid_ze[10]
                            num_prompt_txt2vid_ze.value = readcfg_txt2vid_ze[11]
                            motion_field_strength_x_txt2vid_ze.value = readcfg_txt2vid_ze[12]
                            motion_field_strength_y_txt2vid_ze.value = readcfg_txt2vid_ze[13]
                            timestep_t0_txt2vid_ze.value = readcfg_txt2vid_ze[14]
                            timestep_t1_txt2vid_ze.value = readcfg_txt2vid_ze[15]
                            use_gfpgan_txt2vid_ze.value = readcfg_txt2vid_ze[16]
                            tkme_txt2vid_ze.value = readcfg_txt2vid_ze[17]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2vid_ze = gr.Textbox(lines=4, max_lines=4, label="Prompt", info="Describe what you want in your video", placeholder="a panda is playing guitar on times square")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2vid_ze = gr.Textbox(lines=4, max_lines=4, label="Negative Prompt", info="Describe what you DO NOT want in your video", placeholder="out of frame, ugly")
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
                            out_txt2vid_ze = gr.Video(label="Generated video", height=400, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_txt2vid_ze = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_txt2vid_ze_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_txt2vid_ze_cancel.click(fn=initiate_stop_txt2vid_ze, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_txt2vid_ze_clear_input = gr.ClearButton(components=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2vid_ze_clear_output = gr.ClearButton(components=[out_txt2vid_ze], value="Clear outputs 🧹")                           
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
                                    nsfw_filter,
                                    num_chunks_txt2vid_ze,
                                    use_gfpgan_txt2vid_ze,
                                    tkme_txt2vid_ze,
                                ],
                                outputs=out_txt2vid_ze,
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... video module ...')
                                        txt2vid_ze_vid2vid_ze = gr.Button(" >> Video Instruct-pix2pix")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2vid_ze_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        txt2vid_ze_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        txt2vid_ze_txt2img_lcm_input = gr.Button(" >> LCM")
                                        txt2vid_ze_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        txt2vid_ze_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        gr.HTML(value='... video module ...')
                                        txt2vid_ze_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        txt2vid_ze_animatediff_lcm_input = gr.Button(" >> AnimateLCM")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# animate_lcm
                if ram_size() >= 16 :
                    titletab_tab_animatediff_lcm = "AnimateLCM "
                else :
                    titletab_tab_animatediff_lcm = "AnimateLCM ⛔"
                with gr.TabItem(titletab_tab_animatediff_lcm, id=43) as tab_animatediff_lcm:
                    with gr.Accordion("About", open=False):
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>AnimateLCM</br>
                                <b>Function : </b>Generate video from a prompt and a negative prompt using <a href='https://animatelcm.github.io/' target='_blank'>AnimateLCM</a> with <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> Models</br>
                                <b>Input(s) : </b>Prompt, negative prompt</br>
                                <b>Output(s) : </b>Video</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/emilianJR/epiCRealism' target='_blank'>emilianJR/epiCRealism</a>, 
                                <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></br>
                                """
#                                 <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>, 
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model, modify the number of frames to generate or change dimensions of the outputs</br>
                                - Fill the <b>prompt</b> with what you want to see in your output video</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output video</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated video is displayed in the <b>Generated video</b> field.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Pixify to see them in the models list.
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_animatediff_lcm = gr.Dropdown(choices=model_list_animatediff_lcm, value=model_list_animatediff_lcm[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_animatediff_lcm = gr.Slider(1, biniou_global_steps_max, step=1, value=4, label="Steps", info="Number of iterations per video. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_animatediff_lcm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value="LCM", label="Sampler", info="Sampler to use for inference", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_animatediff_lcm = gr.Slider(0.1, 20.0, step=0.1, value=2.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                seed_animatediff_lcm = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                            with gr.Column():
                                num_frames_animatediff_lcm = gr.Slider(1, 1200, step=1, value=16, label="Video Length (frames)", info="Number of frames in the output video (@8fps)")
                        with gr.Row():
                            with gr.Column():
                                width_animatediff_lcm = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sd15_width, label="Video Width", info="Width of outputs")
                            with gr.Column():
                                height_animatediff_lcm = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=biniou_global_sd15_height, label="Video Height", info="Height of outputs")
                            with gr.Column():
                                num_videos_per_prompt_animatediff_lcm = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of videos to generate in a single run", interactive=False)
                            with gr.Column():
                                num_prompt_animatediff_lcm = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_animatediff_lcm = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs", visible=False)
                            with gr.Column():
                                tkme_animatediff_lcm = gr.Slider(0.0, 1.0, step=0.01, value=0, label="Token Merging ratio", info="0=slow,best quality, 1=fast,worst quality", visible=False)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_animatediff_lcm = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_animatediff_lcm = gr.Textbox(value="animatediff_lcm", visible=False, interactive=False)
                                del_ini_btn_animatediff_lcm = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_animatediff_lcm.value) else False)
                                save_ini_btn_animatediff_lcm.click(
                                    fn=write_ini,
                                    inputs=[
                                        module_name_animatediff_lcm,
                                        model_animatediff_lcm,
                                        num_inference_step_animatediff_lcm,
                                        sampler_animatediff_lcm,
                                        guidance_scale_animatediff_lcm,
                                        seed_animatediff_lcm,
                                        num_frames_animatediff_lcm,
                                        width_animatediff_lcm,
                                        height_animatediff_lcm,
                                        num_videos_per_prompt_animatediff_lcm,
                                        num_prompt_animatediff_lcm,
                                        use_gfpgan_animatediff_lcm,
                                        tkme_animatediff_lcm,
                                        ]
                                    )
                                save_ini_btn_animatediff_lcm.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_animatediff_lcm.click(fn=lambda: del_ini_btn_animatediff_lcm.update(interactive=True), outputs=del_ini_btn_animatediff_lcm)
                                del_ini_btn_animatediff_lcm.click(fn=lambda: del_ini(module_name_animatediff_lcm.value))
                                del_ini_btn_animatediff_lcm.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_animatediff_lcm.click(fn=lambda: del_ini_btn_animatediff_lcm.update(interactive=False), outputs=del_ini_btn_animatediff_lcm)
                        if test_cfg_exist(module_name_animatediff_lcm.value) :
                            readcfg_animatediff_lcm = read_ini_animatediff_lcm(module_name_animatediff_lcm.value)
                            model_animatediff_lcm.value = readcfg_animatediff_lcm[0]
                            num_inference_step_animatediff_lcm.value = readcfg_animatediff_lcm[1]
                            sampler_animatediff_lcm.value = readcfg_animatediff_lcm[2]
                            guidance_scale_animatediff_lcm.value = readcfg_animatediff_lcm[3]
                            seed_animatediff_lcm.value = readcfg_animatediff_lcm[4]
                            num_frames_animatediff_lcm.value = readcfg_animatediff_lcm[5]
                            width_animatediff_lcm.value = readcfg_animatediff_lcm[8]
                            height_animatediff_lcm.value = readcfg_animatediff_lcm[9]
                            num_videos_per_prompt_animatediff_lcm.value = readcfg_animatediff_lcm[10]
                            num_prompt_animatediff_lcm.value = readcfg_animatediff_lcm[11]
                            use_gfpgan_animatediff_lcm.value = readcfg_animatediff_lcm[16]
                            tkme_animatediff_lcm.value = readcfg_animatediff_lcm[17]
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                with gr.Column():
                                    prompt_animatediff_lcm = gr.Textbox(lines=4, max_lines=4, label="Prompt", info="Describe what you want in your video", placeholder="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_animatediff_lcm = gr.Textbox(lines=4, max_lines=4, label="Negative Prompt", info="Describe what you DO NOT want in your video", placeholder="bad quality, worst quality, low resolution")
                        model_animatediff_lcm.change(
                            fn=change_model_type_animatediff_lcm,
                            inputs=[model_animatediff_lcm],
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
                            out_animatediff_lcm = gr.Video(label="Generated video", height=400, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_animatediff_lcm = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_animatediff_lcm_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_animatediff_lcm_cancel.click(fn=initiate_stop_animatediff_lcm, inputs=None, outputs=None)
                        with gr.Column():
                            btn_animatediff_lcm_clear_input = gr.ClearButton(components=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm], value="Clear inputs 🧹")
                        with gr.Column():
                            btn_animatediff_lcm_clear_output = gr.ClearButton(components=[out_animatediff_lcm], value="Clear outputs 🧹")
                            btn_animatediff_lcm.click(
                                fn=video_animatediff_lcm,
                                inputs=[
                                    model_animatediff_lcm,
                                    num_inference_step_animatediff_lcm,
                                    sampler_animatediff_lcm,
                                    guidance_scale_animatediff_lcm,
                                    seed_animatediff_lcm,
                                    num_frames_animatediff_lcm,
                                    height_animatediff_lcm,
                                    width_animatediff_lcm,
                                    num_videos_per_prompt_animatediff_lcm,
                                    num_prompt_animatediff_lcm,
                                    prompt_animatediff_lcm,
                                    negative_prompt_animatediff_lcm,
                                    nsfw_filter,
                                    use_gfpgan_animatediff_lcm,
                                    tkme_animatediff_lcm,
                                ],
                                outputs=out_animatediff_lcm,
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... video module ...')
                                        animatediff_lcm_vid2vid_ze = gr.Button(" >> Video Instruct-pix2pix")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        animatediff_lcm_txt2img_sd_input = gr.Button(" >> Stable Diffusion")
                                        animatediff_lcm_txt2img_kd_input = gr.Button(" >> Kandinsky")
                                        animatediff_lcm_txt2img_lcm_input = gr.Button(" >> LCM")
                                        animatediff_lcm_txt2img_mjm_input = gr.Button(" >> Midjourney-mini") 
                                        animatediff_lcm_txt2img_paa_input = gr.Button(" >> PixArt-Alpha") 
                                        gr.HTML(value='... video module ...')
                                        animatediff_lcm_txt2vid_ms_input = gr.Button(" >> Modelscope")
                                        animatediff_lcm_txt2vid_ze_input = gr.Button(" >> Text2Video-Zero")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# img2vid
                if ram_size() >= 16 :
                    titletab_img2vid = "Stable Video Diffusion "
                else :
                    titletab_img2vid = "Stable Video Diffusion ⛔"
                with gr.TabItem(titletab_img2vid, id=44) as tab_img2vid:
                    with gr.Accordion("About", open=False):
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Stable Video Diffusion</br>
                                <b>Function : </b>Generate video from an input image using <a href='https://stability.ai/news/stable-video-diffusion-open-ai-video-model' target='_blank'>Stable Video Diffusion</a></br>
                                <b>Input(s) : </b>Input image</br>
                                <b>Output(s) : </b>Video</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/stabilityai/stable-video-diffusion-img2vid' target='_blank'>stabilityai/stable-video-diffusion-img2vid</a>, 
                                <a href='https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt' target='_blank'>stabilityai/stable-video-diffusion-img2vid-xt</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an input image</br>
                                - (optional) Modify the settings to use another model, modify the number of frames to generate, fps of the output video or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated video is displayed in the <b>Generated video</b> field.
                                </br>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2vid = gr.Dropdown(choices=model_list_img2vid, value=model_list_img2vid[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_steps_img2vid = gr.Slider(1, biniou_global_steps_max, step=1, value=15, label="Steps", info="Number of iterations per video. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_img2vid = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[5], label="Sampler", info="Sampler to use for inference", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                min_guidance_scale_img2vid = gr.Slider(0.1, 20.0, step=0.1, value=1.0, label="Min guidance scale", info="CFG scale with first frame")
                            with gr.Column():
                                max_guidance_scale_img2vid = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label="Max guidance scale", info="CFG scale with last frame")
                            with gr.Column():
                                seed_img2vid = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():
                                num_frames_img2vid = gr.Slider(1, 1200, step=1, value=14, label="Video Length (frames)", info="Number of frames in the output video")
                            with gr.Column():
                                num_fps_img2vid = gr.Slider(1, 120, step=1, value=7, label="Frames per second", info="Number of frames per second")
                            with gr.Column():
                                decode_chunk_size_img2vid = gr.Slider(1, 32, step=1, value=7, label="Chunk size", info="Number of frames processed in a chunk")
                        with gr.Row():
                            with gr.Column():
                                width_img2vid = gr.Slider(128, biniou_global_width_max_img_create, step=64, value=biniou_global_sdxl_width, label="Video Width", info="Width of outputs")
                            with gr.Column():
                                height_img2vid = gr.Slider(128, biniou_global_height_max_img_create, step=64, value=576, label="Video Height", info="Height of outputs")
                            with gr.Column():
                                num_videos_per_prompt_img2vid = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of videos to generate in a single run", interactive=False)
                            with gr.Column():
                                num_prompt_img2vid = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
#                       with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                motion_bucket_id_img2vid = gr.Slider(0, 256, step=1, value=127, label="Motion bucket ID", info="Higher value = more motion, lower value = less motion")
                            with gr.Column():
                                noise_aug_strength_img2vid = gr.Slider(0.01, 1.0, step=0.01, value=0.02, label="Noise strength", info="Higher value = more motion")
                        with gr.Row():
                            with gr.Column():
                                use_gfpgan_img2vid = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs", visible=False)
                            with gr.Column():
                                tkme_img2vid = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token Merging ratio", info="0=slow,best quality, 1=fast,worst quality", visible=False)
                        model_img2vid.change(fn=change_model_type_img2vid, inputs=model_img2vid, outputs=num_frames_img2vid)
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2vid = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_img2vid = gr.Textbox(value="img2vid", visible=False, interactive=False)
                                del_ini_btn_img2vid = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_img2vid.value) else False)
                                save_ini_btn_img2vid.click(
                                    fn=write_ini,
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
                                save_ini_btn_img2vid.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_img2vid.click(fn=lambda: del_ini_btn_img2vid.update(interactive=True), outputs=del_ini_btn_img2vid)
                                del_ini_btn_img2vid.click(fn=lambda: del_ini(module_name_img2vid.value))
                                del_ini_btn_img2vid.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_img2vid.click(fn=lambda: del_ini_btn_img2vid.update(interactive=False), outputs=del_ini_btn_img2vid)
                        if test_cfg_exist(module_name_img2vid.value) :
                            readcfg_img2vid = read_ini_img2vid(module_name_img2vid.value)
                            model_img2vid.value = readcfg_img2vid[0]
                            num_inference_steps_img2vid.value = readcfg_img2vid[1]
                            sampler_img2vid.value = readcfg_img2vid[2]
                            min_guidance_scale_img2vid.value = readcfg_img2vid[3]
                            max_guidance_scale_img2vid.value = readcfg_img2vid[4]
                            seed_img2vid.value = readcfg_img2vid[5]
                            num_frames_img2vid.value = readcfg_img2vid[6]
                            num_fps_img2vid.value = readcfg_img2vid[7]
                            decode_chunk_size_img2vid.value = readcfg_img2vid[8]
                            height_img2vid.value = readcfg_img2vid[9]
                            width_img2vid.value = readcfg_img2vid[10]
                            num_prompt_img2vid.value = readcfg_img2vid[11]
                            num_videos_per_prompt_img2vid.value = readcfg_img2vid[12]
                            motion_bucket_id_img2vid.value = readcfg_img2vid[13]
                            noise_aug_strength_img2vid.value = readcfg_img2vid[14]
                            use_gfpgan_img2vid.value = readcfg_img2vid[15]
                            tkme_img2vid.value = readcfg_img2vid[16]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    img_img2vid = gr.Image(label="Input image", type="filepath", height=400)
                                    img_img2vid.change(image_upload_event, inputs=img_img2vid, outputs=[width_img2vid, height_img2vid])
                        with gr.Column():
                            out_img2vid = gr.Video(label="Generated video", height=400, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            btn_img2vid = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_img2vid_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_img2vid_cancel.click(fn=initiate_stop_img2vid, inputs=None, outputs=None)
                        with gr.Column():
                            btn_img2vid_clear_input = gr.ClearButton(components=img_img2vid, value="Clear inputs 🧹")
                        with gr.Column():
                            btn_img2vid_clear_output = gr.ClearButton(components=out_img2vid, value="Clear outputs 🧹")
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
                                    use_gfpgan_img2vid,
                                    tkme_img2vid,
                                ],
                                outputs=out_img2vid,
                                show_progress="full",
                            )
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... video module ...')
                                        img2vid_vid2vid_ze = gr.Button(" >> Video Instruct-pix2pix")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

# vid2vid_ze    
                if ram_size() >= 16 :
                    titletab_vid2vid_ze = "Video Instruct-Pix2Pix"
                else :
                    titletab_vid2vid_ze = "Video Instruct-Pix2Pix ⛔"

                with gr.TabItem(titletab_vid2vid_ze, id=45) as tab_vid2vid_ze:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Video Instruct-Pix2Pix</br>
                                <b>Function : </b>Edit an input video with instructions from a prompt and a negative prompt using <a href='https://github.com/timothybrooks/instruct-pix2pix' target='_blank'>Instructpix2pix</a> and <a href='https://github.com/Picsart-AI-Research/Text2Video-Zero' target='_blank'>Text2Video-Zero</a></br>
                                <b>Input(s) : </b>Input video, prompt, negative prompt</br>
                                <b>Output(s) : </b>Video(s)</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/timbrooks/instruct-pix2pix' target='_blank'>timbrooks/instruct-pix2pix</a></br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import a video using the <b>Input video</b> field</br>
                                - Fill the <b>prompt</b> with the instructions for modifying your input video</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output video</br>
                                - (optional) Modify the settings to change the number of frames to process (default=8) or the fps of the output</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated video is displayed in the Generated video field.</br></br>
                                <b>Examples : </b><a href='https://www.timothybrooks.com/instruct-pix2pix/' target='_blank'>Instructpix2pix : Learning to Follow Image Editing Instructions</a>
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_vid2vid_ze = gr.Dropdown(choices=model_list_vid2vid_ze, value=model_list_vid2vid_ze[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_vid2vid_ze = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_vid2vid_ze = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_vid2vid_ze = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                image_guidance_scale_vid2vid_ze = gr.Slider(0.0, 10.0, step=0.1, value=1.5, label="Img CFG Scale", info="Low values : more creativity. High values : more fidelity to the input video")
                            with gr.Column():
                                num_images_per_prompt_vid2vid_ze = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of videos to generate in a single run", interactive=False)
                            with gr.Column():
                                num_prompt_vid2vid_ze = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_vid2vid_ze = gr.Slider(128, biniou_global_width_max_img_modify, step=64, value=biniou_global_sd15_width, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_vid2vid_ze = gr.Slider(128, biniou_global_height_max_img_modify, step=64, value=biniou_global_sd15_height, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_vid2vid_ze = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():
                                num_frames_vid2vid_ze = gr.Slider(0, 1200, step=1, value=8, label="Video Length (frames)", info="Number of frames to process")
                            with gr.Column():
                                num_fps_vid2vid_ze = gr.Slider(1, 120, step=1, value=4, label="Frames per second", info="Number of frames per second")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_vid2vid_ze = gr.Checkbox(value=biniou_global_gfpgan, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_vid2vid_ze = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Token merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_vid2vid_ze = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_vid2vid_ze = gr.Textbox(value="vid2vid_ze", visible=False, interactive=False)
                                del_ini_btn_vid2vid_ze = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_vid2vid_ze.value) else False)
                                save_ini_btn_vid2vid_ze.click(
                                    fn=write_ini, 
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
                                save_ini_btn_vid2vid_ze.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_vid2vid_ze.click(fn=lambda: del_ini_btn_vid2vid_ze.update(interactive=True), outputs=del_ini_btn_vid2vid_ze)
                                del_ini_btn_vid2vid_ze.click(fn=lambda: del_ini(module_name_vid2vid_ze.value))
                                del_ini_btn_vid2vid_ze.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_vid2vid_ze.click(fn=lambda: del_ini_btn_vid2vid_ze.update(interactive=False), outputs=del_ini_btn_vid2vid_ze)
                        if test_cfg_exist(module_name_vid2vid_ze.value) :
                            readcfg_vid2vid_ze = read_ini_vid2vid_ze(module_name_vid2vid_ze.value)
                            model_vid2vid_ze.value = readcfg_vid2vid_ze[0]
                            num_inference_step_vid2vid_ze.value = readcfg_vid2vid_ze[1]
                            sampler_vid2vid_ze.value = readcfg_vid2vid_ze[2]
                            guidance_scale_vid2vid_ze.value = readcfg_vid2vid_ze[3]
                            image_guidance_scale_vid2vid_ze.value = readcfg_vid2vid_ze[4]
                            num_images_per_prompt_vid2vid_ze.value = readcfg_vid2vid_ze[5]
                            num_prompt_vid2vid_ze.value = readcfg_vid2vid_ze[6]
                            width_vid2vid_ze.value = readcfg_vid2vid_ze[7]
                            height_vid2vid_ze.value = readcfg_vid2vid_ze[8]
                            seed_vid2vid_ze.value = readcfg_vid2vid_ze[9]
                            num_frames_vid2vid_ze.value = readcfg_vid2vid_ze[10]
                            num_fps_vid2vid_ze.value = readcfg_vid2vid_ze[11]
                            use_gfpgan_vid2vid_ze.value = readcfg_vid2vid_ze[12]
                            tkme_vid2vid_ze.value = readcfg_vid2vid_ze[13]
                    with gr.Row():
                        with gr.Column():
                             vid_vid2vid_ze = gr.Video(label="Input video", height=400)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_vid2vid_ze = gr.Textbox(lines=5, max_lines=5, label="Prompt", info="Describe what you want to modify in your input video", placeholder="make it Van Gogh Starry Night style")
                                with gr.Column():
                                    negative_prompt_vid2vid_ze = gr.Textbox(lines=5, max_lines=5, label="Negative Prompt", info="Describe what you DO NOT want in your output video", placeholder="out of frame, bad quality, blurry, ugly, text, characters, logo")
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    out_vid2vid_ze = gr.Video(label="Generated video", height=400, interactive=False)
                                    gs_out_vid2vid_ze = gr.State()
                    with gr.Row():
                        with gr.Column():
                            btn_vid2vid_ze = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():                            
                            btn_vid2vid_ze_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_vid2vid_ze_cancel.click(fn=initiate_stop_vid2vid_ze, inputs=None, outputs=None)                              
                        with gr.Column():
                            btn_vid2vid_ze_clear_input = gr.ClearButton(components=[vid_vid2vid_ze, prompt_vid2vid_ze, negative_prompt_vid2vid_ze], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_vid2vid_ze_clear_output = gr.ClearButton(components=[out_vid2vid_ze, gs_out_vid2vid_ze], value="Clear outputs 🧹")
                            btn_vid2vid_ze.click(
                                fn=image_vid2vid_ze,
                                inputs=[
                                    model_vid2vid_ze,
                                    sampler_vid2vid_ze,
                                    vid_vid2vid_ze,
                                    prompt_vid2vid_ze,
                                    negative_prompt_vid2vid_ze,
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')                                        
                                        vid2vid_ze_pix2pix = gr.Button(" >> Instruct pix2pix")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
# 3d
        with gr.TabItem("3D Gen", id=5) as tab_3d:
            with gr.Tabs() as tabs_3d:
# txt2shape
                with gr.TabItem("Shap-E txt2shape ", id=51) as tab_txt2shape:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>txt2shape</br>
                                <b>Function : </b>Generate 3d animated gif or 3d mesh object from a prompt using <a href='https://github.com/openai/shap-e' target='_blank'>Shap-E</a></br>
                                <b>Input(s) : </b>Prompt</br>
                                <b>Output(s) : </b>Animated gif or mesh object</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/openai/shap-e' target='_blank'>openai/shap-e</a>
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Fill the <b>prompt</b> with what you want to see in your output</br>
                                - Select the desired output type : animated Gif or 3D Model (mesh)</br> 
                                - (optional) Modify the settings to generate several images in a single run or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images or 3D models are displayed in the output field. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                """
                            ) 
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2shape = gr.Dropdown(choices=model_list_txt2shape, value=model_list_txt2shape[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2shape = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2shape = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[11], label="Sampler", info="Sampler to use for inference", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2shape = gr.Slider(0.1, 50.0, step=0.1, value=15.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2shape = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run", interactive=False)
                            with gr.Column():
                                num_prompt_txt2shape = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                frame_size_txt2shape = gr.Slider(0, biniou_global_width_max_img_create, step=8, value=64, label="Frame size", info="Size of the outputs")
                            with gr.Column():
                                seed_txt2shape = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility", interactive=False) 
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_txt2shape = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_txt2shape = gr.Textbox(value="txt2shape", visible=False, interactive=False)
                                del_ini_btn_txt2shape = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_txt2shape.value) else False)
                                save_ini_btn_txt2shape.click(
                                    fn=write_ini, 
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
                                save_ini_btn_txt2shape.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_txt2shape.click(fn=lambda: del_ini_btn_txt2shape.update(interactive=True), outputs=del_ini_btn_txt2shape)
                                del_ini_btn_txt2shape.click(fn=lambda: del_ini(module_name_txt2shape.value))
                                del_ini_btn_txt2shape.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_txt2shape.click(fn=lambda: del_ini_btn_txt2shape.update(interactive=False), outputs=del_ini_btn_txt2shape)
                        if test_cfg_exist(module_name_txt2shape.value) :
                            readcfg_txt2shape = read_ini_txt2shape(module_name_txt2shape.value)
                            model_txt2shape.value = readcfg_txt2shape[0]
                            num_inference_step_txt2shape.value = readcfg_txt2shape[1]
                            sampler_txt2shape.value = readcfg_txt2shape[2]
                            guidance_scale_txt2shape.value = readcfg_txt2shape[3]
                            num_images_per_prompt_txt2shape.value = readcfg_txt2shape[4]
                            num_prompt_txt2shape.value = readcfg_txt2shape[5]
                            frame_size_txt2shape.value = readcfg_txt2shape[6]
                            seed_txt2shape.value = readcfg_txt2shape[7]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2shape = gr.Textbox(lines=12, max_lines=12, label="Prompt", info="Describe what you want in your image", placeholder="a firecracker")
                            with gr.Row():
                                with gr.Column():
                                    output_type_txt2shape = gr.Radio(choices=["gif", "mesh"], value="gif", label="Output type", info="Choose output type")
                        with gr.Column(scale=2):
                            out_txt2shape = gr.Gallery(
                                label="Generated images",
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=True
                            )    
                            out_size_txt2shape = gr.Number(value=64, visible=False)
                            mesh_out_txt2shape = gr.Model3D(
                                label="Generated object",
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
                                    download_btn_txt2shape_gif = gr.Button("Zip gallery 💾", visible=True) 
                                    download_btn_txt2shape_mesh = gr.Button("Zip model 💾", visible=False) 
                                with gr.Column():
                                    download_file_txt2shape = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_txt2shape_gif.click(fn=zip_download_file_txt2shape, inputs=[out_txt2shape], outputs=[download_file_txt2shape, download_file_txt2shape]) 
                                    download_btn_txt2shape_mesh.click(fn=zip_mesh_txt2shape, inputs=[gs_mesh_out_txt2shape], outputs=[download_file_txt2shape, download_file_txt2shape]) 
                    with gr.Row():
                        with gr.Column():
                            btn_txt2shape_gif = gr.Button("Generate 🚀", variant="primary", visible=True)
                            btn_txt2shape_mesh = gr.Button("Generate 🚀", variant="primary", visible=False) 
                        with gr.Column():
                            btn_txt2shape_clear_input = gr.ClearButton(components=[prompt_txt2shape], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_txt2shape_clear_output = gr.ClearButton(components=[out_txt2shape, gs_out_txt2shape, mesh_out_txt2shape, gs_mesh_out_txt2shape], value="Clear outputs 🧹")   
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')

                if ram_size() >= 16 :
                    titletab_img2shape = "Shap-E img2shape "
                else :
                    titletab_img2shape = "Shap-E img2shape ⛔"
# img2shape
                with gr.TabItem(titletab_img2shape, id=52) as tab_img2shape:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>img2shape</br>
                                <b>Function : </b>Generate 3d animated gif or 3d mesh object from an imput image using <a href='https://github.com/openai/shap-e' target='_blank'>Shap-E</a></br>
                                <b>Input(s) : </b>Input image</br>
                                <b>Output(s) : </b>Animated gif or mesh object</br>
                                <b>HF model page : </b>
                                <a href='https://huggingface.co/openai/shap-e-img2img' target='_blank'>openai/shap-e-img2img</a>
                                </br>
                                """
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - Upload or import an image using the <b>Input image</b> field. To achieve good results, objects to create should be on a white backgrounds</br>
                                - Select the desired output type : animated Gif or 3D Model (mesh)</br>
                                - (optional) Modify the settings to generate several images in a single run or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images or 3D models are displayed in the output field. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                """
                            ) 
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2shape = gr.Dropdown(choices=model_list_img2shape, value=model_list_img2shape[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_img2shape = gr.Slider(1, biniou_global_steps_max, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_img2shape = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[11], label="Sampler", info="Sampler to use for inference", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2shape = gr.Slider(0.1, 50.0, step=0.1, value=3.0, label="CFG scale", info="Low values : more creativity. High values : more fidelity to the prompts")
                            with gr.Column():
                                num_images_per_prompt_img2shape = gr.Slider(1, biniou_global_batch_size_max, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run", interactive=False)
                            with gr.Column():
                                num_prompt_img2shape = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                frame_size_img2shape = gr.Slider(0, biniou_global_width_max_img_create, step=8, value=64, label="Frame size", info="Size of the outputs")
                            with gr.Column():
                                seed_img2shape = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility", interactive=False) 
                        with gr.Row():
                            with gr.Column():
                                save_ini_btn_img2shape = gr.Button("Save custom defaults settings 💾")
                            with gr.Column():
                                module_name_img2shape = gr.Textbox(value="img2shape", visible=False, interactive=False)
                                del_ini_btn_img2shape = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_img2shape.value) else False)
                                save_ini_btn_img2shape.click(
                                    fn=write_ini, 
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
                                save_ini_btn_img2shape.click(fn=lambda: gr.Info('Settings saved'))
                                save_ini_btn_img2shape.click(fn=lambda: del_ini_btn_img2shape.update(interactive=True), outputs=del_ini_btn_img2shape)
                                del_ini_btn_img2shape.click(fn=lambda: del_ini(module_name_img2shape.value))
                                del_ini_btn_img2shape.click(fn=lambda: gr.Info('Settings deleted'))
                                del_ini_btn_img2shape.click(fn=lambda: del_ini_btn_img2shape.update(interactive=False), outputs=del_ini_btn_img2shape)
                        if test_cfg_exist(module_name_img2shape.value) :
                            readcfg_img2shape = read_ini_img2shape(module_name_img2shape.value)
                            model_img2shape.value = readcfg_img2shape[0]
                            num_inference_step_img2shape.value = readcfg_img2shape[1]
                            sampler_img2shape.value = readcfg_img2shape[2]
                            guidance_scale_img2shape.value = readcfg_img2shape[3]
                            num_images_per_prompt_img2shape.value = readcfg_img2shape[4]
                            num_prompt_img2shape.value = readcfg_img2shape[5]
                            frame_size_img2shape.value = readcfg_img2shape[6]
                            seed_img2shape.value = readcfg_img2shape[7]
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    img_img2shape = gr.Image(label="Input image", height=320, type="pil")
                            with gr.Row():
                                with gr.Column():
                                    output_type_img2shape = gr.Radio(choices=["gif", "mesh"], value="gif", label="Output type", info="Choose output type")
                        with gr.Column(scale=2):
                            out_img2shape = gr.Gallery(
                                label="Generated images",
                                show_label=True,
                                elem_id="gallery",
                                columns=3,
                                height=400,
                                visible=True
                            )    
                            out_size_img2shape = gr.Number(value=64, visible=False)
                            mesh_out_img2shape = gr.Model3D(
                                label="Generated object",
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
                                    download_btn_img2shape_gif = gr.Button("Zip gallery 💾", visible=True) 
                                    download_btn_img2shape_mesh = gr.Button("Zip model 💾", visible=False) 
                                with gr.Column():
                                    download_file_img2shape = gr.File(label="Output", height=30, interactive=False, visible=False)
                                    download_btn_img2shape_gif.click(fn=zip_download_file_img2shape, inputs=[out_img2shape], outputs=[download_file_img2shape, download_file_img2shape]) 
                                    download_btn_img2shape_mesh.click(fn=zip_mesh_img2shape, inputs=[gs_mesh_out_img2shape], outputs=[download_file_img2shape, download_file_img2shape]) 
                    with gr.Row():
                        with gr.Column():
                            btn_img2shape_gif = gr.Button("Generate 🚀", variant="primary", visible=True)
                            btn_img2shape_mesh = gr.Button("Generate 🚀", variant="primary", visible=False) 
                        with gr.Column():
                            btn_img2shape_clear_input = gr.ClearButton(components=[img_img2shape], value="Clear inputs 🧹")
                        with gr.Column():                            
                            btn_img2shape_clear_output = gr.ClearButton(components=[out_img2shape, gs_out_img2shape, mesh_out_img2shape, gs_mesh_out_img2shape], value="Clear outputs 🧹")   
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
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
# Global settings
        with gr.TabItem("Settings", id=6) as tab_settings:
            with gr.Tabs() as tabs_settings:
# UI settings
                with gr.TabItem("WebUI control ", id=61) as tab_models_cleaner:
                    with gr.Row():
                         with gr.Accordion("System", open=True):
                             with gr.Row():
                                 with gr.Column():
                                     btn_restart_ui_settings = gr.Button("Restart Pixify")
                                     btn_restart_ui_settings.click(fn=biniouUIControl.restart_program)
                                 with gr.Column():
                                     btn_reload_ui_settings = gr.Button("Reload WebUI")
                                     btn_reload_ui_settings.click(fn=biniouUIControl.reload_ui, _js="window.location.reload()")
                                 with gr.Column():
                                     btn_close_ui_settings = gr.Button("Shutdown Pixify")
                                     btn_close_ui_settings.click(fn=biniouUIControl.close_program)
                                 with gr.Column():
                                     gr.Number(visible=False)
                    with gr.Row():
                         with gr.Accordion("Updates and optimizations", open=True):
                             with gr.Row():
                                 with gr.Column():
                                     optimizer_update_ui = gr.Radio(choices=["cpu", "cuda", "rocm"], value=biniouUIControl.detect_optimizer(), label="Optimization type", info="Choose CPU (default) or a GPU optimization to use and click Update. You have to restart pixify and reload UI after update.")
                             with gr.Row():
                                 with gr.Column():
                                     btn_update_ui = gr.Button("Update pixify", variant="primary")
                                     btn_update_ui.click(fn=biniouUIControl.biniou_update, inputs=optimizer_update_ui, outputs=optimizer_update_ui)
                                 with gr.Column():
                                     gr.Number(visible=False)
                                 with gr.Column():
                                     gr.Number(visible=False)
                                 with gr.Column():
                                     gr.Number(visible=False)
                    with gr.Row():
                        with gr.Accordion("Common settings", open=True):
                            with gr.Accordion("Backend settings", open=True):
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_server_name = gr.Checkbox(value=biniou_global_server_name, label="LAN accessibility", info="Uncheck to limit access of pixify to localhost only (default = True)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_server_port = gr.Slider(0, 65535, step=1, precision=0, value=biniou_global_server_port, label="Server port", info="Define server port (default = 7860)")
                                    with gr.Column():
                                        biniou_global_settings_inbrowser = gr.Checkbox(value=biniou_global_inbrowser, label="Load in browser at start", info="Open webui in browser when starting pixify (default = False)", interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_auth = gr.Checkbox(value=biniou_global_auth, label="Activate authentication", info="A simple user/pass authentication (default = pixify/pixify). Credentials are stored in ./ini/auth.cfg (default = False)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_auth_message = gr.Textbox(value=biniou_global_auth_message, lines=1, max_lines=3, label="Login message", info="Login screen welcome message. Authentication is required.", interactive=True if biniou_global_auth else False)
                                    with gr.Column():
                                        biniou_global_settings_share = gr.Checkbox(value=biniou_global_share, label="Share online", info="⚠️ Allow online access by a public link to this pixify instance. Authentication is required. (default = False)⚠️", interactive=True if biniou_global_auth else False)
                                        biniou_global_settings_auth.change(biniou_global_settings_auth_switch, biniou_global_settings_auth, [biniou_global_settings_auth_message, biniou_global_settings_share])
                            with gr.Accordion("Images settings", open=True):
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_steps_max = gr.Slider(0, 512, step=1, value=biniou_global_steps_max, label="Maximum steps", info="Maximum number of possible iterations in a generation (default=100)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_batch_size_max = gr.Slider(1, 512, step=1, value=biniou_global_batch_size_max, label="Maximum batch size", info ="Maximum value for a batch size (default=4)", interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_width_max_img_create = gr.Slider(128, 16384, step=64, value=biniou_global_width_max_img_create, label="Maximum image width (create)", info="Maximum width of outputs when using modules that create contents (default = 1280)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_height_max_img_create = gr.Slider(128, 16384, step=64, value=biniou_global_height_max_img_create, label="Maximum image height (create)", info="Maximum height of outputs when using modules that create contents (default = 1280)", interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_width_max_img_modify = gr.Slider(128, 16384, step=64, value=biniou_global_width_max_img_modify, label="Maximum image width (modify)", info="Maximum width of outputs when using modules that modify contents (default = 8192)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_height_max_img_modify = gr.Slider(128, 16384, step=64, value=biniou_global_height_max_img_modify, label="Maximum image height (modify)", info="Maximum height of outputs when using modules that modify contents (default = 8192)", interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_sd15_width = gr.Slider(128, 16384, step=64, value=biniou_global_sd15_width, label="Default image width (SD 1.5 models)", info="Width of outputs when using SD 1.5 models (default = 512)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_sd15_height = gr.Slider(128, 16384, step=64, value=biniou_global_sd15_height, label="Default image height (SD 1.5 models)", info="Height of outputs when using SD 1.5 models (default = 512)", interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_sdxl_width = gr.Slider(128, 16384, step=64, value=biniou_global_sdxl_width, label="Default image width (SDXL models)", info="Width of outputs when using modules that modify contents (default = 8192)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_sdxl_height = gr.Slider(128, 16384, step=64, value=biniou_global_sdxl_height, label="Default image height (SDXL models)", info="Height of outputs when using modules that modify contents (default = 8192)", interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        biniou_global_settings_gfpgan = gr.Checkbox(value=biniou_global_gfpgan, label="Default use of GFPGAN to restore faces", info="Activate/desactivate gfpgan enhancement for all modules using it (default = True)", interactive=True)
                                    with gr.Column():
                                        biniou_global_settings_tkme = gr.Slider(0.0, 1.0, step=0.01, value=biniou_global_tkme, label="Default token merging ratio", info="Set token merging ratio for all modules using it (default = 0.6)", interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_settings = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_settings = gr.Textbox(value="settings", visible=False, interactive=False)
                                    del_ini_btn_settings = gr.Button("Delete custom defaults settings 🗑️", interactive=True if test_cfg_exist(module_name_settings.value) else False)
                                    save_ini_btn_settings.click(fn=write_settings_ini, 
                                        inputs=[
                                            module_name_settings,
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
                                        ],
                                        outputs=None
                                    )
                                    save_ini_btn_settings.click(fn=lambda: gr.Info('Common settings saved'))
                                    save_ini_btn_settings.click(fn=lambda: del_ini_btn_settings.update(interactive=True), outputs=del_ini_btn_settings)
                                    del_ini_btn_settings.click(fn=lambda: del_ini(module_name_settings.value))
                                    del_ini_btn_settings.click(fn=lambda: gr.Info('Common settings deleted'))
                                    del_ini_btn_settings.click(fn=lambda: del_ini_btn_settings.update(interactive=False), outputs=del_ini_btn_settings)
                    with gr.Row():
                        with gr.Accordion("NSFW filter", open=False):
                            with gr.Row():
                                with gr.Column():
                                    safety_checker_ui_settings = gr.Checkbox(bool(int(nsfw_filter.value)), label="Use safety checker", info="⚠️ Warning : Unchecking this box will temporarily disable the safety checker which avoid generation of nsfw and disturbing media contents. This option is ONLY provided for debugging purposes and you should NEVER uncheck it in other use cases. ⚠️", interactive=True)
                                    safety_checker_ui_settings.change(fn=lambda x:int(x), inputs=safety_checker_ui_settings, outputs=nsfw_filter)

# Models cleaner
                with gr.TabItem("Models cleaner", id=62) as tab_models_cleaner:
                    with gr.Row():
                        list_models_cleaner = gr.CheckboxGroup(choices=biniouModelsManager("./models").modelslister(), type="value", label="Installed models list", info="Select the models you want to delete and click \"Delete selected models\" button. Restart pixify to re-synchronize models list.")
                    with gr.Row():
                        with gr.Column():
                            btn_models_cleaner = gr.Button("Delete selected models", variant="primary")
                            btn_models_cleaner.click(fn=biniouModelsManager("./models").modelsdeleter, inputs=[list_models_cleaner])
                            btn_models_cleaner.click(fn=refresh_models_cleaner_list, outputs=list_models_cleaner)
                        with gr.Column():
                            btn_models_cleaner_refresh = gr.Button("Refresh models list")
                            btn_models_cleaner_refresh.click(fn=refresh_models_cleaner_list, outputs=list_models_cleaner)
                        with gr.Column():
                            gr.Number(visible=False)
                        with gr.Column():
                            gr.Number(visible=False)
# LoRA Models manager
                with gr.TabItem("LoRA models manager", id=63) as tab_lora_models_manager:
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>SD models</span>""")
                            with gr.Row():
                                list_lora_models_manager_sd = gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SD").modelslister(), type="value", label="Installed models list", info="Select the LoRA models you want to delete and click \"Delete selected models\" button. Restart pixify to re-synchronize LoRA models list.")
                            with gr.Row():
                                with gr.Column():
                                    btn_lora_models_manager_sd = gr.Button("Delete selected models", variant="primary")
                                    btn_lora_models_manager_sd.click(fn=biniouLoraModelsManager("./models/lora/SD").modelsdeleter, inputs=[list_lora_models_manager_sd])
                                    btn_lora_models_manager_sd.click(fn=refresh_lora_models_manager_list_sd, outputs=list_lora_models_manager_sd)
                                with gr.Column():
                                    btn_lora_models_manager_refresh_sd = gr.Button("Refresh models list")
                                    btn_lora_models_manager_refresh_sd.click(fn=refresh_lora_models_manager_list_sd, outputs=list_lora_models_manager_sd)
                            with gr.Row():
                                with gr.Column():
                                    url_lora_models_manager_sd = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label="LoRA model URL", info="Paste here the url of the LoRA model you want to download. Restart pixify to re-synchronize LoRA models list. Safetensors files only.")
                            with gr.Row():
                                with gr.Column():
                                    btn_url_lora_models_manager_sd = gr.Button("Download LoRA model", variant="primary")
                                    btn_url_lora_models_manager_sd.click(biniouLoraModelsManager("./models/lora/SD").modelsdownloader, inputs=url_lora_models_manager_sd, outputs=url_lora_models_manager_sd)
                                with gr.Column():
                                        gr.Number(visible=False)
                        with gr.Column():
                            gr.HTML("""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>SDXL models</span>""")
                            with gr.Row():
                                list_lora_models_manager_sdxl = gr.CheckboxGroup(choices=biniouLoraModelsManager("./models/lora/SDXL").modelslister(), type="value", label="Installed models list", info="Select the LoRA models you want to delete and click \"Delete selected models\" button. Restart pixify to re-synchronize LoRA models list.")
                            with gr.Row():
                                with gr.Column():
                                    btn_lora_models_manager_sdxl = gr.Button("Delete selected models", variant="primary")
                                    btn_lora_models_manager_sdxl.click(fn=biniouLoraModelsManager("./models/lora/SDXL").modelsdeleter, inputs=[list_lora_models_manager_sdxl])
                                    btn_lora_models_manager_sdxl.click(fn=refresh_lora_models_manager_list_sdxl, outputs=list_lora_models_manager_sdxl)
                                with gr.Column():
                                    btn_lora_models_manager_refresh_sdxl = gr.Button("Refresh models list")
                                    btn_lora_models_manager_refresh_sdxl.click(fn=refresh_lora_models_manager_list_sdxl, outputs=list_lora_models_manager_sdxl)
                            with gr.Row():
                                with gr.Column():
                                    url_lora_models_manager_sdxl = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label="LoRA model URL", info="Paste here the url of the LoRA model you want to download. Restart pixify to re-synchronize LoRA models list. Safetensors files only.")
                            with gr.Row():
                                with gr.Column():
                                    btn_url_lora_models_manager_sdxl = gr.Button("Download LoRA model", variant="primary")
                                    btn_url_lora_models_manager_sdxl.click(biniouLoraModelsManager("./models/lora/SDXL").modelsdownloader, inputs=url_lora_models_manager_sdxl, outputs=url_lora_models_manager_sdxl)
                                with gr.Column():
                                        gr.Number(visible=False)

# Textual inversion Models manager
                with gr.TabItem("Textual inversion manager", id=64) as tab_textinv_manager:
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>SD textual inversion</span>""")
                            with gr.Row():
                                list_textinv_manager_sd = gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SD").modelslister(), type="value", label="Installed textual inversion list", info="Select the textual inversion you want to delete and click \"Delete selected textual inversion\" button. Restart pixify to re-synchronize textual inversion list.")
                            with gr.Row():
                                with gr.Column():
                                    btn_textinv_manager_sd = gr.Button("Delete selected textual inversion", variant="primary")
                                    btn_textinv_manager_sd.click(fn=biniouTextinvModelsManager("./models/TextualInversion/SD").modelsdeleter, inputs=[list_textinv_manager_sd])
                                    btn_textinv_manager_sd.click(fn=refresh_textinv_manager_list_sd, outputs=list_textinv_manager_sd)
                                with gr.Column():
                                    btn_textinv_manager_refresh_sd = gr.Button("Refresh textual inversion list")
                                    btn_textinv_manager_refresh_sd.click(fn=refresh_textinv_manager_list_sd, outputs=list_textinv_manager_sd)
                            with gr.Row():
                                with gr.Column():
                                    url_textinv_manager_sd = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label="Textual inversion URL", info="Paste here the url of the textual inversion you want to download. Restart pixify to re-synchronize textual inversion list. Safetensors files only.")
                            with gr.Row():
                                with gr.Column():
                                    btn_url_textinv_manager_sd = gr.Button("Download textual inversion", variant="primary")
                                    btn_url_textinv_manager_sd.click(biniouTextinvModelsManager("./models/TextualInversion/SD").modelsdownloader, inputs=url_textinv_manager_sd, outputs=url_textinv_manager_sd)
                                with gr.Column():
                                        gr.Number(visible=False)
                        with gr.Column():
                            gr.HTML("""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>SDXL textual inversion</span>""")
                            with gr.Row():
                                list_textinv_manager_sdxl = gr.CheckboxGroup(choices=biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelslister(), type="value", label="Installed textual inversion list", info="Select the textual inversion you want to delete and click \"Delete selected textual inversion\" button. Restart pixify to re-synchronize textual inversion list.")
                            with gr.Row():
                                with gr.Column():
                                    btn_textinv_manager_sdxl = gr.Button("Delete selected textual inversion", variant="primary")
                                    btn_textinv_manager_sdxl.click(fn=biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelsdeleter, inputs=[list_textinv_manager_sdxl])
                                    btn_textinv_manager_sdxl.click(fn=refresh_textinv_manager_list_sdxl, outputs=list_textinv_manager_sdxl)
                                with gr.Column():
                                    btn_textinv_manager_refresh_sdxl = gr.Button("Refresh textual inversion list")
                                    btn_textinv_manager_refresh_sdxl.click(fn=refresh_textinv_manager_list_sdxl, outputs=list_textinv_manager_sdxl)
                            with gr.Row():
                                with gr.Column():
                                    url_textinv_manager_sdxl = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label="Textual inversion URL", info="Paste here the url of the Textual inversion you want to download. Restart pixify to re-synchronize textual inversion list. Safetensors files only.")
                            with gr.Row():
                                with gr.Column():
                                    btn_url_textinv_manager_sdxl = gr.Button("Download textual inversion", variant="primary")
                                    btn_url_textinv_manager_sdxl.click(biniouTextinvModelsManager("./models/TextualInversion/SDXL").modelsdownloader, inputs=url_textinv_manager_sdxl, outputs=url_textinv_manager_sdxl)
                                with gr.Column():
                                        gr.Number(visible=False)

# SD Models downloader
                with gr.TabItem("SD models downloader", id=65) as tab_sd_models_downloader:
                    with gr.Row():
                        with gr.Column():
#                            gr.HTML("""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>SD models</span>""")
                            with gr.Row():
                                with gr.Column():
                                    url_sd_models_downloader = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label="Stable Diffusion model URL", info="Paste here the url of the model you want to download. Restart pixify to re-synchronize models list. SDXL models must contains \"xl\" in their names to be correctly identified. Safetensors files only.")
                            with gr.Row():
                                with gr.Column():
                                    btn_url_sd_models_downloader = gr.Button("Download SD model", variant="primary")
                                    btn_url_sd_models_downloader.click(biniouSDModelsDownloader("./models/Stable_Diffusion").modelsdownloader, inputs=url_sd_models_downloader, outputs=url_sd_models_downloader)
                                with gr.Column():
                                        gr.Number(visible=False)
                                with gr.Column():
                                        gr.Number(visible=False)
                                with gr.Column():
                                        gr.Number(visible=False)

# GGUF Models downloader
                with gr.TabItem("GGUF models downloader", id=66) as tab_gguf_models_downloader:
                    with gr.Row():
                        with gr.Column():
#                            gr.HTML("""<span style='text-align: left; font-size: 24px; font-weight: bold; line-height:24px;'>SD models</span>""")
                            with gr.Row():
                                with gr.Column():
                                    url_gguf_models_downloader = gr.Textbox(value="", lines=1, max_lines=2, interactive=True, label="Chatbot Llama-cpp GGUF model URL", info="Paste here the url of the model you want to download. Restart pixify to re-synchronize models list. gguf files only.")
                            with gr.Row():
                                with gr.Column():
                                    btn_url_gguf_models_downloader = gr.Button("Download GGUF model", variant="primary")
                                    btn_url_gguf_models_downloader.click(biniouSDModelsDownloader("./models/llamacpp").modelsdownloader, inputs=url_gguf_models_downloader, outputs=url_gguf_models_downloader)
                                with gr.Column():
                                        gr.Number(visible=False)
                                with gr.Column():
                                        gr.Number(visible=False)
                                with gr.Column():
                                        gr.Number(visible=False)

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
    resrgan_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_faceswap_num, tab_inpaint_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])       
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
    gfpgan_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_faceswap_num, tab_inpaint_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])    
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

# AnimateLCM outputs
    animatediff_lcm_vid2vid_ze.click(fn=send_to_module_video, inputs=[out_animatediff_lcm, tab_video_num, tab_vid2vid_ze_num], outputs=[vid_vid2vid_ze, tabs, tabs_video])

# AnimateLCM inputs
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
    with gr.Accordion("pixify console", open=False):
        with gr.Row():
            with gr.Column():
                biniou_console_output = gr.Textbox(label="pixifyoutput", value="", lines=5, max_lines=5, show_copy_button=True)
        with gr.Row():
            with gr.Column():
                download_file_console = gr.File(label="Download logfile", value=logfile_biniou, height=30, interactive=False)
                biniou_console_output.change(refresh_logfile, None, download_file_console)
            with gr.Column():
                gr.Number(visible=False)
            with gr.Column():
                gr.Number(visible=False)
            with gr.Column():
                gr.Number(visible=False)

# Exécution de l'UI :
    demo.load(split_url_params, nsfw_filter, [nsfw_filter, url_params_current, safety_checker_ui_settings], _js=get_window_url_params)
    demo.load(read_logs, None, biniou_console_output, every=1)
#    demo.load(fn=lambda: gr.Info('pixifyloading completed. Ready to work !'))
    if biniou_global_server_name:
        print(f">>>[pixify🔮]: Up and running at https://{local_ip()}:{biniou_global_server_port}/?__theme=dark")
    else:
        print(f">>>[pixify🔮]: Up and running at https://127.0.0.1:{biniou_global_server_port}/?__theme=dark")

if __name__ == "__main__":
    demo.queue(concurrency_count=8).launch(
        server_name="webui.pixifyai.art" if biniou_global_server_name else "127.0.0.1",
        server_port=biniou_global_server_port,
        ssl_certfile="./ssl/cert.pem" if not biniou_global_share else None,
        favicon_path="./images/pixify_64.ico",
        ssl_keyfile="./ssl/key.pem" if not biniou_global_share else None,
        ssl_verify=False,
        auth=biniou_auth_values if biniou_global_auth else None,
        auth_message=biniou_global_auth_message if biniou_global_auth else None,
        share=biniou_global_share,
        inbrowser=biniou_global_inbrowser,
#        inbrowser=True if len(sys.argv)>1 and sys.argv[1]=="--inbrowser" else biniou_global_inbrowser,
    )
# Fin du fichier
