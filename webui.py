# https://github.com/Woolverine94/biniou
# Webui.py
import os 
import gradio as gr
import numpy as np
import warnings
import shutil
from PIL import Image
from ressources.common import *
from ressources.llamacpp import *
from ressources.img2txt_git import *
from ressources.whisper import *
from ressources.nllb import *
from ressources.txt2img_sd import *
from ressources.txt2img_kd import *
from ressources.img2img import *
from ressources.pix2pix import *
from ressources.inpaint import *
from ressources.controlnet import *
from ressources.faceswapper import *
from ressources.r_esrgan import *
from ressources.gfpgan import *
from ressources.musicgen import *
from ressources.audiogen import *
from ressources.harmonai import *
from ressources.bark import *
from ressources.txt2vid_ms import *
from ressources.txt2vid_ze import *

tmp_biniou="./.tmp"
if os.path.exists(tmp_biniou) :
    shutil.rmtree(tmp_biniou)
os.makedirs(tmp_biniou, exist_ok=True)

warnings.filterwarnings('ignore') 

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
        }
    """

def split_url_params(url_params) :
    url_params = eval(url_params.replace("'", "\""))
    if "nsfw_filter" in url_params.keys():
        output_nsfw = url_params["nsfw_filter"]
        return output_nsfw
    else :         
        return "1"

## Fonctions communes
def dummy():
    return

def in_and_out(input_value):
    return input_value

# fonctions Exports Outputs 
def send_to_module(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) # /!\ tabs_image = pas bon pour les autres modules

def send_to_module_inpaint(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)
    
def send_to_module_text(content, index, numtab, numtab_item):
    index = int(index)
    return content[index], gr.Tabs.update(selected=numtab), tabs_text.update(selected=numtab_item)    

def send_text_to_module_image (prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)
    
def send_audio_to_module_text(audio, numtab, numtab_item):
    return audio, gr.Tabs.update(selected=numtab), tabs_text.update(selected=numtab_item)    

def send_text_to_module_text(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_text.update(selected=numtab_item)

# fonctions Exports Inputs
def import_to_module(prompt, negative_prompt, numtab, numtab_item):
    return prompt, negative_prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)
    
def import_to_module_audio(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_audio.update(selected=numtab_item)    
    
def import_to_module_video(prompt, negative_prompt, numtab, numtab_item):
    return prompt, negative_prompt, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)   

def import_text_to_module_image(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item)

def import_text_to_module_video(prompt, numtab, numtab_item):
    return prompt, gr.Tabs.update(selected=numtab), tabs_video.update(selected=numtab_item)    

# fonctions Exports Inputs + Outputs
def both_text_to_module_image (content, prompt, numtab, numtab_item):
    return content, prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) 

def both_text_to_module_inpaint_image (content, prompt, numtab, numtab_item):
    return content, content, prompt, gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) 
   
def both_to_module(prompt, negative_prompt, content, index, numtab, numtab_item):
    index = int(index)
    return prompt, negative_prompt, content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) # /!\ tabs_image = pas bon pour les autres modules    

def both_to_module_inpaint(prompt, negative_prompt, content, index, numtab, numtab_item):
    index = int(index)
    return prompt, negative_prompt, content[index], content[index], gr.Tabs.update(selected=numtab), tabs_image.update(selected=numtab_item) # /!\ tabs_image = pas bon pour les autres modules    

def get_select_index(evt: gr.SelectData) :
    return evt.index

## Fonctions spécifiques à Stable Diffusion 
def zip_download_file_txt2img_sd(content):
    savename = zipper(content)
    return savename, download_file_txt2img_sd.update(visible=True) 

def hide_download_file_txt2img_sd():
    return download_file_txt2img_sd.update(visible=False)
    
def update_preview_txt2img_sd(preview):
    return out_txt2img_sd.update(preview)     

## Fonctions spécifiques à Kandinsky 
def zip_download_file_txt2img_kd(content):
    savename = zipper(content)
    return savename, download_file_txt2img_kd.update(visible=True) 

def hide_download_file_txt2img_kd():
    return download_file_txt2img_kd.update(visible=False)
    
## Fonctions spécifiques à img2img 
def zip_download_file_img2img(content):
    savename = zipper(content)
    return savename, download_file_img2img.update(visible=True) 

def hide_download_file_img2img():
    return download_file_img2img.update(visible=False)        
    
def change_source_type_img2img(source_type_img2img):
    if source_type_img2img == "image" :
        return {"source": "upload", "tool": None, "__type__": "update"}
    elif source_type_img2img == "sketch" :
        return {"source": "canvas", "tool": "color-sketch", "__type__": "update"}

## Fonctions spécifiques à pix2pix 
def zip_download_file_pix2pix(content):
    savename = zipper(content)
    return savename, download_file_pix2pix.update(visible=True) 

def hide_download_file_pix2pix():
    return download_file_pix2pix.update(visible=False) 
    
## Fonctions spécifiques à inpaint 
def zip_download_file_inpaint(content):
    savename = zipper(content)
    return savename, download_file_inpaint.update(visible=True) 

def hide_download_file_inpaint():
    return download_file_inpaint.update(visible=False) 

## Fonctions spécifiques à controlnet 
def zip_download_file_controlnet(content):
    savename = zipper(content)
    return savename, download_file_controlnet.update(visible=True) 

def hide_download_file_controlnet():
    return download_file_controlnet.update(visible=False) 

## Fonctions spécifiques à faceswap 
def zip_download_file_faceswap(content):
    savename = zipper(content)
    return savename, download_file_faceswap.update(visible=True) 

def hide_download_file_faceswap():
    return download_file_faceswap.update(visible=False) 

## Fonctions spécifiques à whisper
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
    print(source_audio_whisper)
    print(type(source_audio_whisper))
    return source_audio_whisper.update(source="upload"), source_audio_whisper

color_label = "#7B43EE"
color_label_button = "#4361ee"

theme_gradio = gr.themes.Base().set(
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

with gr.Blocks(theme=theme_gradio) as demo:
    gr.HTML(
        """<a href='https://github.com/Woolverine94/biniou' style='text-decoration: none;'><p style='float:left;'><img src='file/images/biniou_64.png' width='32' height='32'/></p>
        <p style='text-align: left; font-size: 32px; font-weight: bold; line-height:32px;'>biniou</p></a>"""
    )
    nsfw_filter = gr.Textbox(value="1", visible=False)
    with gr.Tabs() as tabs:
# Chat
        with gr.TabItem("Text ✍️", id=1) as tab_text:
            with gr.Tabs() as tabs_text:
# llamacpp
                with gr.TabItem("Chatbot Llama-cpp (gguf) 📝", id=12) as tab_llamacpp:
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
                                <a href='https://huggingface.co/TheBloke/Vigogne-2-7B-Instruct-GGUF' target='_blank'>TheBloke/Vigogne-2-7B-Instruct-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Vigogne-2-13B-Instruct-GGUF' target='_blank'>TheBloke/Vigogne-2-13B-Instruct-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF' target='_blank'>TheBloke/Airoboros-L2-7B-2.1-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/Airoboros-L2-13B-2.1-GGUF' target='_blank'>TheBloke/Airoboros-L2-13B-2.1-GGUF</a>, 
                                <a href='https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF' target='_blank'>TheBloke/CodeLlama-13B-Instruct-GGUF</a></br>
                                """
                            )
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
                                - You could place llama-cpp compatible .gguf models in the directory ./biniou/models/llamacpp. Restart Biniou to see them in the models list.                                  
                                </div>
                                """
                            )
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_llamacpp = gr.Dropdown(choices=list(model_list_llamacpp.keys()), value=list(model_list_llamacpp.keys())[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                max_tokens_llamacpp = gr.Slider(0, 65536, step=16, value=128, label="Max tokens", info="Maximum number of tokens to generate")
                            with gr.Column():
                                seed_llamacpp = gr.Slider(0, 10000000000, step=1, value=1337, label="Seed(0 for random)", info="Seed to use for generation.")    
                        with gr.Row():
                            with gr.Column():
                                stream_llamacpp = gr.Checkbox(value=False, label="Stream", info="Stream results", interactive=False)                            
                            with gr.Column():
                                n_ctx_llamacpp = gr.Slider(0, 65536, step=128, value=512, label="n_ctx", info="Maximum context size")
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
                        history_llamacpp = gr.Textbox(label="Chatbot history", lines=8, max_lines=8, interactive=False)
                        hidden_history_llamacpp = gr.Textbox(label="Chatbot history", visible=False)
                    with gr.Row():                        
                        out_llamacpp = gr.Textbox(label="Chatbot last reply", lines=2, max_lines=2, interactive=False)
                    with gr.Row():
                            prompt_llamacpp = gr.Textbox(label="Input", lines=1, max_lines=1, placeholder="Type your request here ...")                            
                            hidden_prompt_llamacpp = gr.Textbox(value="", visible=False)
                            out_llamacpp.change(fn=in_and_out, inputs=hidden_history_llamacpp, outputs=history_llamacpp)  
                            out_llamacpp.change(fn=in_and_out, inputs=hidden_prompt_llamacpp, outputs=prompt_llamacpp)                              
                    with gr.Row():
                        with gr.Column():
                            btn_llamacpp = gr.Button("Generate 🚀", variant="primary")
                        with gr.Column():
                            btn_llamacpp_continue = gr.Button("Continue ➕")                            
                        with gr.Column():
                            btn_llamacpp_clear_input = gr.ClearButton(components=[prompt_llamacpp], value="Clear inputs 🧹")
                        with gr.Column():                      
                            btn_llamacpp_clear_output = gr.ClearButton(components=[out_llamacpp, history_llamacpp, hidden_history_llamacpp], value="Clear all 🧹") 
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
                            ],
                            outputs=[out_llamacpp, hidden_history_llamacpp],
                            show_progress="full",
                        )
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
                            ],
                            outputs=[out_llamacpp, hidden_history_llamacpp],
                            show_progress="full",
                        )
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
                                out_llamacpp,                                   
                            ],
                            outputs=[out_llamacpp, hidden_history_llamacpp],
                            show_progress="full",
                        )                        
                    with gr.Accordion("Send ...", open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... selected output to ...')
                                        gr.HTML(value='... text module ...')
                                        llamacpp_nllb = gr.Button("✍️ >> Nllb translation")
                                        gr.HTML(value='... image module ...')                                        
                                        llamacpp_txt2img_sd = gr.Button("✍️ >> Stable Diffusion")
                                        llamacpp_txt2img_kd = gr.Button("✍️ >> Kandinsky")                                        
                                        llamacpp_img2img = gr.Button("✍️ >> img2img")
                                        llamacpp_pix2pix = gr.Button("✍️ >> pix2pix")
                                        llamacpp_inpaint = gr.Button("✍️ >> inpaint")
                                        llamacpp_controlnet = gr.Button("✍️ >> ControlNet")
                                        gr.HTML(value='... audio module ...')
                                        llamacpp_musicgen = gr.Button("✍️ >> Musicgen")                                        
                                        llamacpp_audiogen = gr.Button("✍️ >> Audiogen")
                                        llamacpp_bark = gr.Button("✍️ >> Bark")                                        
                                        gr.HTML(value='... video module ...')                                               
                                        llamacpp_txt2vid_ms = gr.Button("✍️ >> Modelscope")
                                        llamacpp_txt2vid_ze = gr.Button("✍️ >> Text2Video-Zero")                                        
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
# Image captioning                                        
                with gr.TabItem("Image captioning 👁️", id=13) as tab_img2txt_git:
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
                            img_img2txt_git = gr.Image(label="Input image", type="pil", height=400)
                        with gr.Column():
                            out_img2txt_git = gr.Textbox(label="Generated captions", lines=15, interactive=False)
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
                                        img2txt_git_nllb = gr.Button("✍️ >> Nllb translation")
                                        gr.HTML(value='... image module ...')                                        
                                        img2txt_git_txt2img_sd = gr.Button("✍️ >> Stable Diffusion")
                                        img2txt_git_txt2img_kd = gr.Button("✍️ >> Kandinsky")                                        
                                        img2txt_git_img2img = gr.Button("✍️ >> img2img")
                                        img2txt_git_pix2pix = gr.Button("✍️ >> pix2pix")
                                        img2txt_git_inpaint = gr.Button("✍️ >> inpaint")
                                        img2txt_git_controlnet = gr.Button("✍️ >> ControlNet")                                        
                                        gr.HTML(value='... audio module ...')
                                        img2txt_git_musicgen = gr.Button("✍️ >> Musicgen")                                        
                                        img2txt_git_audiogen = gr.Button("✍️ >> Audiogen")
                                        gr.HTML(value='... video module ...')                                               
                                        img2txt_git_txt2vid_ms = gr.Button("✍️ >> Modelscope")
                                        img2txt_git_txt2vid_ze = gr.Button("✍️ >> Text2Video-Zero")                                        
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        img2txt_git_img2img_both = gr.Button("🖼️+✍️ >> img2img")
                                        img2txt_git_pix2pix_both = gr.Button("🖼️+✍️ >> pix2pix")
                                        img2txt_git_inpaint_both = gr.Button("🖼️+✍️ >> inpaint")

# Whisper 
                with gr.TabItem("Whisper 👂", id=14) as tab_whisper:
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
                                <a href='https://huggingface.co/openai/whisper-large-v2' target='_blank'>openai/whisper-large-v2</a></br>
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
                            with gr.Row():
                                with gr.Column():
                                    source_type_whisper = gr.Radio(choices=["audio", "micro"], value="audio", label="Input type", info="Choose input type")
                                with gr.Column():
                                    source_language_whisper = gr.Dropdown(choices=language_list_whisper, value=language_list_whisper[13], label="Input language", info="Select input language")    
                            with gr.Row():
                                source_audio_whisper = gr.Audio(label="Source audio", source="upload", type="filepath")
                                source_type_whisper.change(fn=change_source_type_whisper, inputs=source_type_whisper, outputs=source_audio_whisper)
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    output_type_whisper = gr.Radio(choices=["transcribe", "translate"], value="transcribe", label="Task", info="Choose task to execute")
                                with gr.Column():
                                    output_language_whisper = gr.Dropdown(choices=language_list_whisper, value=language_list_whisper[13], label="Output language", info="Select output language", visible=False, interactive=False)
                            with gr.Row():
                                out_whisper = gr.Textbox(label="Output text", lines=9, max_lines=9, interactive=False)
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
                                        whisper_nllb = gr.Button("✍️ >> Nllb translation")
                                        gr.HTML(value='... image module ...')
                                        whisper_txt2img_sd = gr.Button("✍️ >> Stable Diffusion")
                                        whisper_txt2img_kd = gr.Button("✍️ >> Kandinsky")
                                        whisper_img2img = gr.Button("✍️ >> img2img")
                                        whisper_pix2pix = gr.Button("✍️ >> pix2pix")
                                        whisper_inpaint = gr.Button("✍️ >> inpaint")
                                        whisper_controlnet = gr.Button("✍️ >> ControlNet")
                                        gr.HTML(value='... audio module ...')
                                        whisper_musicgen = gr.Button("✍️ >> Musicgen")
                                        whisper_audiogen = gr.Button("✍️ >> Audiogen")
                                        whisper_bark = gr.Button("✍️ >> Bark")
                                        gr.HTML(value='... video module ...')
                                        whisper_txt2vid_ms = gr.Button("✍️ >> Modelscope")
                                        whisper_txt2vid_ze = gr.Button("✍️ >> Text2Video-Zero")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')



# nllb 
                with gr.TabItem("nllb translation 👥", id=15) as tab_nllb:
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
                            with gr.Row():
                                source_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()), value=list(language_list_nllb.keys())[47], label="Input language", info="Select input language")    
                            with gr.Row():
                                prompt_nllb = gr.Textbox(label="Source text", lines=9, max_lines=9, placeholder="Type or paste here the text to translate")
                        with gr.Column():
                            with gr.Row():
                                output_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()), value=list(language_list_nllb.keys())[47], label="Output language", info="Select output language")
                            with gr.Row():
                                out_nllb = gr.Textbox(label="Output text", lines=9, max_lines=9, interactive=False)
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
                                        gr.HTML(value='... image module ...')                                        
                                        nllb_txt2img_sd = gr.Button("✍️ >> Stable Diffusion")
                                        nllb_txt2img_kd = gr.Button("✍️ >> Kandinsky")                                        
                                        nllb_img2img = gr.Button("✍️ >> img2img")
                                        nllb_pix2pix = gr.Button("✍️ >> pix2pix")
                                        nllb_inpaint = gr.Button("✍️ >> inpaint")
                                        nllb_controlnet = gr.Button("✍️ >> ControlNet")                                        
                                        gr.HTML(value='... audio module ...')
                                        nllb_musicgen = gr.Button("✍️ >> Musicgen")                                        
                                        nllb_audiogen = gr.Button("✍️ >> Audiogen")
                                        nllb_bark = gr.Button("✍️ >> Bark")                                        
                                        gr.HTML(value='... video module ...')                                               
                                        nllb_txt2vid_ms = gr.Button("✍️ >> Modelscope")
                                        nllb_txt2vid_ze = gr.Button("✍️ >> Text2Video-Zero")                                        
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')



# Image
        with gr.TabItem("Image 🖼️", id=2) as tab_image:
            with gr.Tabs() as tabs_image:
# Stable Diffusion
                with gr.TabItem("Stable Diffusion 🖼️", id=21) as tab_txt2img_sd:
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
                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0' target='_blank'>stabilityai/stable-diffusion-xl-base-1.0</a>,
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>,
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a>
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
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory ./biniou/models/Stable Diffusion. Restart Biniou to see them in the models list.  
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2img_sd = gr.Dropdown(choices=model_list_txt2img_sd, value=model_list_txt2img_sd[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2img_sd = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_sd = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_sd = gr.Slider(0.1, 20.0, step=0.1, value=7.0, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2img_sd = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_sd = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_sd = gr.Slider(128, 1280, step=64, value=512, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_sd = gr.Slider(128, 1280, step=64, value=512, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_sd = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")    
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_sd = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_txt2img_sd = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Token merging ratio", info="0=slow,better quality, 1=fast,worst quality")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():                        
                                    prompt_txt2img_sd = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():
                                with gr.Column(): 
                                    negative_prompt_txt2img_sd = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
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
                                        txt2img_sd_img2txt_git = gr.Button("🖼️ >> GIT Captioning")      
                                        gr.HTML(value='... image module ...')
                                        txt2img_sd_img2img = gr.Button("🖼️ >> img2img")
                                        txt2img_sd_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        txt2img_sd_inpaint = gr.Button("🖼️ >> inpaint")
                                        txt2img_sd_controlnet = gr.Button("🖼️ >> ControlNet")
                                        txt2img_sd_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        txt2img_sd_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        txt2img_sd_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_sd_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")                                        
                                        txt2img_sd_img2img_input = gr.Button("✍️ >> img2img")
                                        txt2img_sd_pix2pix_input = gr.Button("✍️ >> pix2pix")
                                        txt2img_sd_inpaint_input = gr.Button("✍️ >> inpaint")
                                        txt2img_sd_controlnet_input = gr.Button("✍️ >> ControlNet")
                                        gr.HTML(value='... video module ...')                                        
                                        txt2img_sd_txt2vid_ms_input = gr.Button("✍️ >> Modelscope")
                                        txt2img_sd_txt2vid_ze_input = gr.Button("✍️ >> Text2Video-Zero")                                        
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')                                                                            
                                        txt2img_sd_img2img_both = gr.Button("🖼️ + ✍️ >> img2img")
                                        txt2img_sd_pix2pix_both = gr.Button("🖼️ + ✍️ >> pix2pix")
                                        txt2img_sd_inpaint_both = gr.Button("🖼️ + ✍️ >> inpaint")
                                        txt2img_sd_controlnet_both = gr.Button("🖼️ + ✍️️ >> ControlNet")                                        

# Kandinsky
                if ram_size() >= 16 :
                    titletab_txt2img_kd = "Kandinsky 🖼️"
                else :
                    titletab_txt2img_kd = "Kandinsky 🖼️ ⛔"

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
                                num_inference_step_txt2img_kd = gr.Slider(1, 100, step=1, value=25, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2img_kd = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[5], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2img_kd = gr.Slider(0.1, 20.0, step=0.1, value=4.0, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                num_images_per_prompt_txt2img_kd = gr.Slider(1, 4, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2img_kd = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2img_kd = gr.Slider(128, 1280, step=64, value=512, label="Image Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2img_kd = gr.Slider(128, 1280, step=64, value=512, label="Image Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2img_kd = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2img_kd = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():   
                                    prompt_txt2img_kd = gr.Textbox(lines=6, max_lines=6, label="Prompt", info="Describe what you want in your image", placeholder="An alien cheeseburger creature eating itself, claymation, cinematic, moody lighting")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2img_kd = gr.Textbox(lines=6, max_lines=6, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="low quality, bad quality")
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
                                        txt2img_kd_img2txt_git = gr.Button("🖼️ >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        txt2img_kd_img2img = gr.Button("🖼️ >> img2img")
                                        txt2img_kd_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        txt2img_kd_inpaint = gr.Button("🖼️ >> inpaint")
                                        txt2img_kd_controlnet = gr.Button("🖼️ >> ControlNet")
                                        txt2img_kd_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        txt2img_kd_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        txt2img_kd_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_kd_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        txt2img_kd_img2img_input = gr.Button("✍️ >> img2img")
                                        txt2img_kd_pix2pix_input = gr.Button("✍️ >> pix2pix")
                                        txt2img_kd_inpaint_input = gr.Button("✍️ >> inpaint")
                                        txt2img_kd_controlnet_input = gr.Button("✍️ >> ControlNet")                                        
                                        gr.HTML(value='... video module ...')                                                                                
                                        txt2img_kd_txt2vid_ms_input = gr.Button("✍️ >> Modelscope")
                                        txt2img_kd_txt2vid_ze_input = gr.Button("✍️ >> Text2Video-Zero")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2img_kd_img2img_both = gr.Button("🖼️ + ✍️ >> img2img")
                                        txt2img_kd_pix2pix_both = gr.Button("🖼️ + ✍️ >> pix2pix")
                                        txt2img_kd_inpaint_both = gr.Button("🖼️ + ✍️ >> inpaint")
                                        txt2img_kd_controlnet_both = gr.Button("🖼️ + ✍️ >> ControlNet")                                        
# img2img    
                with gr.TabItem("img2img 🖌️", id=23) as tab_img2img:
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
                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                                <a href='https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0' target='_blank'>stabilityai/stable-diffusion-xl-refiner-1.0</a>,
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
                                - Upload, import an image or draw a sketch as an <b>Input image</b></br>
                                - Set the balance between the input image and the prompt (<b>denoising strength</b>) to a value between 0 and 1 : 0 will completely ignore the prompt, 1 will completely ignore the input image</br>                                
                                - Fill the <b>prompt</b> with what you want to see in your output image</br>
                                - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output image</br>
                                - (optional) Modify the settings to use another model or generate several images in a single run</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated images are displayed in the gallery. Save them individually or create a downloadable zip of the whole gallery.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Biniou to see them in the models list. 
                                </div>
                                """
                            )               
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_img2img = gr.Dropdown(choices=model_list_img2img, value=model_list_img2img[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_img2img = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_img2img = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_img2img = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                num_images_per_prompt_img2img = gr.Slider(1, 4, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_img2img = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_img2img = gr.Slider(128, 8192, step=64, value=512, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_img2img = gr.Slider(128, 8192, step=64, value=512, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_img2img = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_img2img = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_img2img = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Token merging ratio", info="0=slow,better quality, 1=fast,worst quality")                          
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
                                    denoising_strength_img2img = gr.Slider(0.0, 1.0, step=0.01, value=0.75, label="Denoising strength", info="Balance between input image (0) and prompts (1)")  
                            with gr.Row():
                                with gr.Column():
                                    prompt_img2img = gr.Textbox(lines=5, max_lines=5, label="Prompt", info="Describe what you want in your image", placeholder="a cute kitten playing with a ball, dynamic pose, close-up cinematic still, photo realistic, ultra quality, 4k uhd, perfect lighting, HDR, bokeh")
                            with gr.Row():                                    
                                with gr.Column():
                                    negative_prompt_img2img = gr.Textbox(lines=5, max_lines=5, label="Negative Prompt", info="Describe what you DO NOT want in your image", placeholder="out of frame, bad quality, medium quality, blurry, ugly, duplicate, text, characters, logo")
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
                                        img2img_img2txt_git = gr.Button("🖼️ >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        img2img_img2img = gr.Button("🖼️ >> img2img")
                                        img2img_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        img2img_inpaint = gr.Button("🖼️ >> inpaint")
                                        img2img_controlnet = gr.Button("🖼️ >> ControlNet")
                                        img2img_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        img2img_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        img2img_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        img2img_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        img2img_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")
                                        img2img_pix2pix_input = gr.Button("✍️ >> pix2pix")
                                        img2img_inpaint_input = gr.Button("✍️ >> inpaint")
                                        img2img_controlnet_input = gr.Button("✍️ >> ControlNet")                                        
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')
                                        img2img_pix2pix_both = gr.Button("🖼️ + ✍️ >> pix2pix")
                                        img2img_inpaint_both = gr.Button("🖼️ + ✍️ >> inpaint")
                                        img2img_controlnet_both = gr.Button("🖼️ + ✍️ >> ControlNet")

# pix2pix    
                with gr.TabItem("pix2pix 🖌️", id=24) as tab_pix2pix:
                    with gr.Accordion("About", open=False):                
                        with gr.Box():                       
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                <b>Module : </b>Instructpix2pix</br>
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
                                - (optional) Modify the settings to generate several images in a single run or generate several images in a single run</br>
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
                                num_inference_step_pix2pix = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_pix2pix = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_pix2pix = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                image_guidance_scale_pix2pix = gr.Slider(0.0, 10.0, step=0.1, value=1.5, label="Img CFG Scale", info="Low values : more creativity. High values : more corresponding to the input image")
                            with gr.Column():
                                num_images_per_prompt_pix2pix = gr.Slider(1, 4, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_pix2pix = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_pix2pix = gr.Slider(128, 8192, step=64, value=512, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_pix2pix = gr.Slider(128, 8192, step=64, value=512, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_pix2pix = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_pix2pix = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_pix2pix = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Token merging ratio", info="0=slow,better quality, 1=fast,worst quality")
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
                                        pix2pix_img2txt_git = gr.Button("🖼️ >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        pix2pix_img2img = gr.Button("🖼️ >> img2img")
                                        pix2pix_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        pix2pix_inpaint = gr.Button("🖼️ >> inpaint")
                                        pix2pix_controlnet = gr.Button("🖼️ >> ControlNet")
                                        pix2pix_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        pix2pix_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        pix2pix_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')                                        
                                        pix2pix_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        pix2pix_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")                                        
                                        pix2pix_img2img_input = gr.Button("✍️ >> img2img")
                                        pix2pix_inpaint_input = gr.Button("✍️ >> inpaint")
                                        pix2pix_controlnet_input = gr.Button("✍️ >> ControlNet")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')                                        
                                        pix2pix_img2img_both = gr.Button("🖼️ + ✍️ >> img2img")
                                        pix2pix_inpaint_both = gr.Button("🖼️ + ✍️ >> inpaint")
                                        pix2pix_controlnet_both = gr.Button("🖼️ + ✍️ >> ControlNet")
# inpaint    
                with gr.TabItem("inpaint 🖌️", id=25) as tab_inpaint:
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
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Biniou to see them in the models list.
                                </div>
                                """
                            )                   
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_inpaint = gr.Dropdown(choices=model_list_inpaint, value=model_list_inpaint[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_inpaint = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_inpaint = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_inpaint = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="CFG Scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                num_images_per_prompt_inpaint= gr.Slider(1, 4, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_inpaint = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_inpaint = gr.Slider(128, 8192, step=64, value=512, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_inpaint = gr.Slider(128, 8192, step=64, value=512, label="Image Height", info="Height of outputs", interactive=False)
                            with gr.Column():
                                seed_inpaint = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_inpaint = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_inpaint = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Token merging ratio", info="0=slow,better quality, 1=fast,worst quality")
                    with gr.Row():
                        with gr.Column(scale=2):
                             rotation_img_inpaint = gr.Number(value=0, visible=False)
                             img_inpaint = gr.Image(label="Input image", type="pil", height=400, tool="sketch")
                             img_inpaint.upload(image_upload_event_inpaint, inputs=img_inpaint, outputs=[width_inpaint, height_inpaint, img_inpaint, rotation_img_inpaint], preprocess=False)
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
                                        inpaint_img2txt_git = gr.Button("🖼️ >> GIT Captioning")      
                                        gr.HTML(value='... image module ...')
                                        inpaint_img2img = gr.Button("🖼️ >> img2img")
                                        inpaint_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        inpaint_inpaint = gr.Button("🖼️ >> inpaint")
                                        inpaint_controlnet = gr.Button("🖼️ >> ControlNet")
                                        inpaint_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        inpaint_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        inpaint_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        inpaint_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        inpaint_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")                                        
                                        inpaint_img2img_input = gr.Button("✍️ >> img2img")
                                        inpaint_pix2pix_input = gr.Button("✍️ >> pix2pix")
                                        inpaint_controlnet_input = gr.Button("✍️ >> ControlNet")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    
                                        gr.HTML(value='... image module ...')                                        
                                        inpaint_img2img_both = gr.Button("🖼️ + ✍️ >> img2img")
                                        inpaint_pix2pix_both = gr.Button("🖼️ + ✍️ >> pix2pix")
                                        inpaint_controlnet_both = gr.Button("🖼️ + ✍️ >> ControlNet")
# ControlNet
                with gr.TabItem("ControlNet 🖼️", id=26) as tab_controlnet:
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
                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>, 
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
                            )
                        with gr.Box():
                            gr.HTML(
                                """
                                <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                <div style='text-align: justified'>
                                <b>Usage :</b></br>
                                - (optional) Modify the settings to use another model, change the settings for ControlNet or adjust threshold on canny</br>
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
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory ./biniou/models/Stable Diffusion. Restart Biniou to see them in the models list.
                                </div>
                                """
                            )                
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_controlnet = gr.Dropdown(choices=model_list_controlnet, value=model_list_controlnet[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_controlnet = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per image. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_controlnet = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_controlnet = gr.Slider(0.1, 20.0, step=0.1, value=7.0, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                num_images_per_prompt_controlnet = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Batch size", info ="Number of images to generate in a single run")
                            with gr.Column():
                                num_prompt_controlnet = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_controlnet = gr.Slider(128, 1280, step=64, value=512, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_controlnet = gr.Slider(128, 1280, step=64, value=512, label="Image Height", info="Height of outputs", interactive=False)
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
                                use_gfpgan_controlnet = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():
                                tkme_controlnet = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Token merging ratio", info="0=slow,better quality, 1=fast,worst quality")
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
                                    btn_controlnet_preview = gr.Button("Preview 👁️")
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
                                        controlnet_img2txt_git = gr.Button("🖼️ >> GIT Captioning")         
                                        gr.HTML(value='... image module ...')
                                        controlnet_img2img = gr.Button("🖼️ >> img2img")
                                        controlnet_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        controlnet_inpaint = gr.Button("🖼️ >> inpaint")
                                        controlnet_controlnet = gr.Button("🖼️ >> ControlNet")
                                        controlnet_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        controlnet_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        controlnet_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        controlnet_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        controlnet_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")                                        
                                        controlnet_img2img_input = gr.Button("✍️ >> img2img")
                                        controlnet_pix2pix_input = gr.Button("✍️ >> pix2pix")
                                        controlnet_inpaint_input = gr.Button("✍️ >> inpaint")         
                                        gr.HTML(value='... video module ...')                                        
                                        controlnet_txt2vid_ms_input = gr.Button("✍️ >> Modelscope")
                                        controlnet_txt2vid_ze_input = gr.Button("✍️ >> Text2Video-Zero")                                        
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
                                        gr.HTML(value='... image module ...')                                                                            
                                        controlnet_img2img_both = gr.Button("🖼️ + ✍️ >> img2img")
                                        controlnet_pix2pix_both = gr.Button("🖼️ + ✍️ >> pix2pix")
                                        controlnet_inpaint_both = gr.Button("🖼️ + ✍️ >> inpaint")                                        
# faceswap    
                with gr.TabItem("Faceswap 🎭", id=27) as tab_faceswap:
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
                                width_faceswap = gr.Slider(128, 8192, step=64, value=512, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_faceswap = gr.Slider(128, 8192, step=64, value=512, label="Image Height", info="Height of outputs", interactive=False)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_faceswap = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")    
                    with gr.Row():
                        with gr.Column():
                             img_source_faceswap = gr.Image(label="Source image", height=400, type="filepath")
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
                                        faceswap_img2txt_git = gr.Button("🖼️ >> GIT Captioning")
                                        gr.HTML(value='... image module ...')
                                        faceswap_img2img = gr.Button("🖼️ >> img2img")
                                        faceswap_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        faceswap_inpaint = gr.Button("🖼️ >> inpaint")
                                        faceswap_controlnet = gr.Button("🖼️ >> ControlNet")
                                        faceswap_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        faceswap_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                                        faceswap_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    

# Real ESRGAN    
                with gr.TabItem("Real ESRGAN 🔎", id=28) as tab_resrgan:
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
                                width_resrgan = gr.Slider(128, 8192, step=64, value=512, label="Image Width", info="Width of input", interactive=False)
                            with gr.Column():
                                height_resrgan = gr.Slider(128, 8192, step=64, value=512, label="Image Height", info="Height of input", interactive=False)
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_resrgan = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")                                 
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
                                        resrgan_img2txt_git = gr.Button("🖼️ >> GIT Captioning") 
                                        gr.HTML(value='... image module ...')
                                        resrgan_img2img = gr.Button("🖼️ >> img2img")
                                        resrgan_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        resrgan_inpaint = gr.Button("🖼️ >> inpaint")
                                        resrgan_controlnet = gr.Button("🖼️ >> ControlNet")
                                        resrgan_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        resrgan_gfpgan = gr.Button("🖼️ >> GFPGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                         
# GFPGAN    
                with gr.TabItem("GFPGAN 🔎", id=29) as tab_gfpgan:
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
                                width_gfpgan = gr.Slider(128, 8192, step=64, value=512, label="Image Width", info="Width of outputs", interactive=False)
                            with gr.Column():
                                height_gfpgan = gr.Slider(128, 8192, step=64, value=512, label="Image Height", info="Height of outputs", interactive=False)
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
                                        gfpgan_img2txt_git = gr.Button("🖼️ >> GIT Captioning")   
                                        gr.HTML(value='... image module ...')
                                        gfpgan_img2img = gr.Button("🖼️ >> img2img")
                                        gfpgan_pix2pix = gr.Button("🖼️ >> pix2pix")
                                        gfpgan_inpaint = gr.Button("🖼️ >> inpaint")
                                        gfpgan_controlnet = gr.Button("🖼️ >> ControlNet")
                                        gfpgan_faceswap = gr.Button("🖼️ >> Faceswap target")
                                        gfpgan_resrgan = gr.Button("🖼️ >> Real ESRGAN")
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                        
# Audio
        with gr.TabItem("Audio 🎵", id=3) as tab_audio:
            with gr.Tabs() as tabs_audio:        
# Musicgen
                with gr.TabItem("MusicGen 🎶", id=31) as tab_musicgen:
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
                                <a href='https://huggingface.co/facebook/musicgen-melody' target='_blank'>facebook/musicgen-melody</a></br>                                
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
                                cfg_coef_musicgen = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
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
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... audio module ...')                                        
                                        musicgen_audiogen_input = gr.Button("✍️ >> Audiogen")                                        
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    

# Audiogen
                if ram_size() >= 16 :
                    titletab_audiogen = "AudioGen 🔊"
                else :
                    titletab_audiogen = "AudioGen 🔊 ⛔"
                
                with gr.TabItem(titletab_audiogen, id=32) as tab_audiogen:

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
                                cfg_coef_audiogen = gr.Slider(0.1, 20.0, step=0.1, value=3.0, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
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
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... audio module ...')
                                        audiogen_musicgen_input = gr.Button("✍️ >> Musicgen")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                                    

# Harmonai
                with gr.TabItem("Harmonai 🔊", id=33) as tab_harmonai:
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
                                steps_harmonai = gr.Slider(1, 100, step=1, value=50, label="Steps", info="Number of iterations per audio. Results and speed depends of sampler")
                            with gr.Column():
                                seed_harmonai = gr.Slider(0, 10000000000, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():
                                length_harmonai = gr.Slider(1, 1200, value=5, step=1, label="Audio length (sec)")
                            with gr.Column():
                                batch_size_harmonai = gr.Slider(1, 4, step=1, value=1, label="Batch size", info ="Number of audios to generate in a single run")
                            with gr.Column():
                                batch_repeat_harmonai = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
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
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')                       
# Bark
                with gr.TabItem("Bark 🗣️", id=34) as tab_bark:
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
        with gr.TabItem("Video 🎬", id=4) as tab_video:
            with gr.Tabs() as tabs_video:           
# Modelscope            
                if ram_size() >= 16 :
                    titletab_txt2vid_ms = "Modelscope 📼"
                else :
                    titletab_txt2vid_ms = "Modelscope 📼 ⛔"
                    
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
                                num_inference_step_txt2vid_ms = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per video. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2vid_ms = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                        with gr.Row():
                            with gr.Column():
                                guidance_scale_txt2vid_ms = gr.Slider(0.1, 20.0, step=0.1, value=4.0, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
                            with gr.Column():
                                num_frames_txt2vid_ms = gr.Slider(1, 1200, step=1, value=8, label="Video Length (frames)", info="Number of frames in the output video")
                            with gr.Column():
                                num_prompt_txt2vid_ms = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")
                        with gr.Row():
                            with gr.Column():
                                width_txt2vid_ms = gr.Slider(128, 1280, step=64, value=576, label="Video Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2vid_ms = gr.Slider(128, 1280, step=64, value=320, label="Video Height", info="Height of outputs")
                            with gr.Column():
                                seed_txt2vid_ms = gr.Slider(0, 10000000000, step=1, value=0, label="Seed(0 for random)", info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2vid_ms = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2vid_ms = gr.Textbox(lines=4, max_lines=4, label="Prompt", info="Describe what you want in your video", placeholder="Darth vader is surfing on waves, photo realistic, best quality")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2vid_ms = gr.Textbox(lines=4, max_lines=4, label="Negative Prompt", info="Describe what you DO NOT want in your video", placeholder="out of frame, ugly")
                        with gr.Column():
                            out_txt2vid_ms = gr.Video(label="Generated video", height=400)
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
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2vid_ms_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        txt2vid_ms_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")
                                        gr.HTML(value='... video module ...')
                                        txt2vid_ms_txt2vid_ze_input = gr.Button("✍️ >> Text2Video-Zero")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')
# Txt2vid_zero            
                with gr.TabItem("Text2Video-Zero 📼", id=42) as tab_txt2vid_ze:
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
                                <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>, 
                                <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></br>
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
                                - (optional) Modify the settings to use another model, modify the number of frames to generate, fps of the output video or change dimensions of the outputs</br>
                                - Click the <b>Generate</b> button</br>
                                - After generation, generated video is displayed in the <b>Generated video</b> field.
                                </br>
                                <b>Models :</b></br>
                                - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /biniou/models/Stable Diffusion. Restart Biniou to see them in the models list.
                                </div>
                                """
                            )                      
                    with gr.Accordion("Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                model_txt2vid_ze = gr.Dropdown(choices=model_list_txt2vid_ze, value=model_list_txt2vid_ze[0], label="Model", info="Choose model to use for inference")
                            with gr.Column():
                                num_inference_step_txt2vid_ze = gr.Slider(1, 100, step=1, value=10, label="Steps", info="Number of iterations per video. Results and speed depends of sampler")
                            with gr.Column():
                                sampler_txt2vid_ze = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()), value=list(SCHEDULER_MAPPING.keys())[0], label="Sampler", info="Sampler to use for inference")
                            with gr.Column():
                                guidance_scale_txt2vid_ze = gr.Slider(0.1, 20.0, step=0.1, value=7.5, label="CFG scale", info="Low values : more creativity. High values : more corresponding to the prompts")
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
                                width_txt2vid_ze = gr.Slider(128, 1280, step=64, value=576, label="Video Width", info="Width of outputs")
                            with gr.Column():
                                height_txt2vid_ze = gr.Slider(128, 1280, step=64, value=320, label="Video Height", info="Height of outputs")
                            with gr.Column():
                                num_videos_per_prompt_txt2vid_ze = gr.Slider(1, 4, step=1, value=1, label="Batch size", info ="Number of videos to generate in a single run")
                            with gr.Column():
                                num_prompt_txt2vid_ze = gr.Slider(1, 32, step=1, value=1, label="Batch count", info="Number of batch to run successively")                            
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    motion_field_strength_x_txt2vid_ze = gr.Slider(0, 50, step=1, value=12, label="Motion field strength x", info="Horizontal motion strength")
                                with gr.Column():
                                    motion_field_strength_y_txt2vid_ze = gr.Slider(0, 50, step=1, value=12, label="Motion field strength y", info="Vertical motion strength")
                                with gr.Column():
                                    timestep_t0_txt2vid_ze = gr.Slider(0, 100, step=1, value=7, label="Timestep t0", interactive=False)
                                with gr.Column():
                                    timestep_t1_txt2vid_ze = gr.Slider(1, 100, step=1, value=8, label="Timestep t1", interactive=False)
                                    num_inference_step_txt2vid_ze.change(set_timestep_txt2vid_ze, inputs=num_inference_step_txt2vid_ze, outputs=[timestep_t0_txt2vid_ze, timestep_t1_txt2vid_ze])
                        with gr.Row():
                            with gr.Column():    
                                use_gfpgan_txt2vid_ze = gr.Checkbox(value=True, label="Use GFPGAN to restore faces", info="Use GFPGAN to enhance faces in the outputs")
                            with gr.Column():    
                                tkme_txt2vid_ze = gr.Slider(0.0, 1.0, step=0.01, value=0.6, label="Token Merging ratio", info="0=slow,best quality, 1=fast,worst quality")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    prompt_txt2vid_ze = gr.Textbox(lines=4, max_lines=4, label="Prompt", info="Describe what you want in your video", placeholder="a panda is playing guitar on times square")
                            with gr.Row():
                                with gr.Column():
                                    negative_prompt_txt2vid_ze = gr.Textbox(lines=4, max_lines=4, label="Negative Prompt", info="Describe what you DO NOT want in your video", placeholder="out of frame, ugly")
                        with gr.Column():
                            out_txt2vid_ze = gr.Video(label="Generated video", height=400)
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
                            with gr.Column():
                                with gr.Box():
                                    with gr.Group():
                                        gr.HTML(value='... input prompt(s) to ...')
                                        gr.HTML(value='... image module ...')
                                        txt2vid_ze_txt2img_sd_input = gr.Button("✍️ >> Stable Diffusion")
                                        txt2vid_ze_txt2img_kd_input = gr.Button("✍️ >> Kandinsky")
                                        gr.HTML(value='... video module ...')                                        
                                        txt2vid_ze_txt2vid_ms_input = gr.Button("✍️ >> Modelscope")
                            with gr.Column():
                                with gr.Box():                                
                                    with gr.Group():
                                        gr.HTML(value='... both to ...')        
   
    tab_text_num = gr.Number(value=tab_text.id, precision=0, visible=False)
    tab_image_num = gr.Number(value=tab_image.id, precision=0, visible=False)
    tab_audio_num = gr.Number(value=tab_audio.id, precision=0, visible=False)    
    tab_video_num = gr.Number(value=tab_video.id, precision=0, visible=False)

    tab_llamacpp_num = gr.Number(value=tab_llamacpp.id, precision=0, visible=False)    
    tab_img2txt_git_num = gr.Number(value=tab_img2txt_git.id, precision=0, visible=False)    
    tab_whisper_num = gr.Number(value=tab_whisper.id, precision=0, visible=False)        
    tab_nllb_num = gr.Number(value=tab_nllb.id, precision=0, visible=False)    
    tab_txt2img_sd_num = gr.Number(value=tab_txt2img_sd.id, precision=0, visible=False)
    tab_txt2img_kd_num = gr.Number(value=tab_txt2img_kd.id, precision=0, visible=False)
    tab_img2img_num = gr.Number(value=tab_img2img.id, precision=0, visible=False)
    tab_pix2pix_num = gr.Number(value=tab_pix2pix.id, precision=0, visible=False)
    tab_inpaint_num = gr.Number(value=tab_inpaint.id, precision=0, visible=False)
    tab_controlnet_num = gr.Number(value=tab_controlnet.id, precision=0, visible=False)    
    tab_faceswap_num = gr.Number(value=tab_faceswap.id, precision=0, visible=False)
    tab_resrgan_num = gr.Number(value=tab_resrgan.id, precision=0, visible=False)
    tab_gfpgan_num = gr.Number(value=tab_gfpgan.id, precision=0, visible=False)
    tab_musicgen_num = gr.Number(value=tab_musicgen.id, precision=0, visible=False)
    tab_audiogen_num = gr.Number(value=tab_audiogen.id, precision=0, visible=False)
    tab_harmonai_num = gr.Number(value=tab_harmonai.id, precision=0, visible=False)
    tab_bark_num = gr.Number(value=tab_bark.id, precision=0, visible=False)
    tab_txt2vid_ms_num = gr.Number(value=tab_txt2vid_ms.id, precision=0, visible=False)
    tab_txt2vid_ze_num = gr.Number(value=tab_txt2vid_ze.id, precision=0, visible=False)    

# Llamacpp outputs   
    llamacpp_nllb.click(fn=send_text_to_module_text, inputs=[out_llamacpp, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    llamacpp_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_llamacpp, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    llamacpp_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_llamacpp, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    llamacpp_img2img.click(fn=send_text_to_module_image, inputs=[out_llamacpp, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    llamacpp_pix2pix.click(fn=send_text_to_module_image, inputs=[out_llamacpp, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    llamacpp_inpaint.click(fn=send_text_to_module_image, inputs=[out_llamacpp, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    llamacpp_controlnet.click(fn=send_text_to_module_image, inputs=[out_llamacpp, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])    
    llamacpp_musicgen.click(fn=import_to_module_audio, inputs=[out_llamacpp, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])    
    llamacpp_audiogen.click(fn=import_to_module_audio, inputs=[out_llamacpp, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    llamacpp_bark.click(fn=import_to_module_audio, inputs=[out_llamacpp, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])    
    llamacpp_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_llamacpp, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    llamacpp_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_llamacpp, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])    

# GIT Captions outputs
    img2txt_git_nllb.click(fn=send_text_to_module_text, inputs=[out_img2txt_git, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])    
    img2txt_git_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    img2txt_git_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    img2txt_git_img2img.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    img2txt_git_pix2pix.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    img2txt_git_inpaint.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    img2txt_git_controlnet.click(fn=send_text_to_module_image, inputs=[out_img2txt_git, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])    
    img2txt_git_musicgen.click(fn=import_to_module_audio, inputs=[out_img2txt_git, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])    
    img2txt_git_audiogen.click(fn=import_to_module_audio, inputs=[out_img2txt_git, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    img2txt_git_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_img2txt_git, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    img2txt_git_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_img2txt_git, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])    

# GIT Captions both
    img2txt_git_img2img_both.click(fn=both_text_to_module_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_img2img_num], outputs=[img_img2img, prompt_img2img, tabs, tabs_image])
    img2txt_git_pix2pix_both.click(fn=both_text_to_module_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, prompt_pix2pix, tabs, tabs_image])
    img2txt_git_inpaint_both.click(fn=both_text_to_module_inpaint_image, inputs=[img_img2txt_git, out_img2txt_git, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, prompt_pix2pix, tabs, tabs_image])
    
# Whisper outputs
    whisper_nllb.click(fn=send_text_to_module_text, inputs=[out_whisper, tab_text_num, tab_nllb_num], outputs=[prompt_nllb, tabs, tabs_text])
    whisper_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    whisper_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    whisper_img2img.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    whisper_pix2pix.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    whisper_inpaint.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    whisper_controlnet.click(fn=send_text_to_module_image, inputs=[out_whisper, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])    
    whisper_musicgen.click(fn=import_to_module_audio, inputs=[out_whisper, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])    
    whisper_audiogen.click(fn=import_to_module_audio, inputs=[out_whisper, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    whisper_bark.click(fn=import_to_module_audio, inputs=[out_whisper, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])    
    whisper_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_whisper, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    whisper_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_whisper, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])       

# Nllb outputs
    nllb_txt2img_sd.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, tabs, tabs_image])
    nllb_txt2img_kd.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, tabs, tabs_image])
    nllb_img2img.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, tabs, tabs_image])
    nllb_pix2pix.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, tabs, tabs_image])
    nllb_inpaint.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, tabs, tabs_image])
    nllb_controlnet.click(fn=send_text_to_module_image, inputs=[out_nllb, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, tabs, tabs_image])    
    nllb_musicgen.click(fn=import_to_module_audio, inputs=[out_nllb, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])    
    nllb_audiogen.click(fn=import_to_module_audio, inputs=[out_nllb, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    nllb_bark.click(fn=import_to_module_audio, inputs=[out_nllb, tab_audio_num, tab_bark_num], outputs=[prompt_bark, tabs, tabs_audio])    
    nllb_txt2vid_ms.click(fn=import_text_to_module_video, inputs=[out_nllb, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, tabs, tabs_video])
    nllb_txt2vid_ze.click(fn=import_text_to_module_video, inputs=[out_nllb, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, tabs, tabs_video])
      
# txt2img_sd outputs
    txt2img_sd_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_sd_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_sd_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])    
    txt2img_sd_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_sd_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])        
    txt2img_sd_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_sd_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_sd_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])    

# txt2img_sd inputs
    txt2img_sd_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    txt2img_sd_img2img_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    txt2img_sd_pix2pix_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    txt2img_sd_inpaint_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    txt2img_sd_controlnet_input.click(fn=import_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])    
    txt2img_sd_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_sd_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])    
    
# txt2img_sd both
    txt2img_sd_img2img_both.click(fn=both_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_sd_pix2pix_both.click(fn=both_to_module, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_sd_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_sd_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])

# txt2img_kd outputs
    txt2img_kd_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    txt2img_kd_pix2pix.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    txt2img_kd_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])    
    txt2img_kd_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])
    txt2img_kd_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])        
    txt2img_kd_resrgan.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    txt2img_kd_gfpgan.click(fn=send_to_module, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    txt2img_kd_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_txt2img_kd, sel_out_txt2img_kd, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])
    
# txt2img_kd inputs
    txt2img_kd_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2img_kd_img2img_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    txt2img_kd_pix2pix_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    txt2img_kd_inpaint_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    txt2img_kd_controlnet_input.click(fn=import_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    txt2img_kd_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2img_kd_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])        
    
# txt2img_kd both
    txt2img_kd_img2img_both.click(fn=both_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    txt2img_kd_pix2pix_both.click(fn=both_to_module, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    txt2img_kd_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    txt2img_kd_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, gs_out_txt2img_kd, sel_out_txt2img_kd, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])    

# img2img outputs
    img2img_img2img.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    img2img_pix2pix.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    img2img_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2img_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])    
    img2img_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])    
    img2img_resrgan.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    img2img_gfpgan.click(fn=send_to_module, inputs=[gs_out_img2img, sel_out_img2img, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    img2img_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_img2img, sel_out_img2img, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])    

# img2img inputs
    img2img_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    img2img_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])    
    img2img_pix2pix_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    img2img_inpaint_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    img2img_controlnet_input.click(fn=import_to_module, inputs=[prompt_img2img, negative_prompt_img2img, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])    
    
# img2img both
    img2img_pix2pix_both.click(fn=both_to_module, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    img2img_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    img2img_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_img2img, negative_prompt_img2img, gs_out_img2img, sel_out_img2img, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])    

# pix2pix outputs
    pix2pix_img2img.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    pix2pix_pix2pix.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])    
    pix2pix_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])
    pix2pix_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])        
    pix2pix_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])    
    pix2pix_resrgan.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    pix2pix_gfpgan.click(fn=send_to_module, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    pix2pix_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_pix2pix, sel_out_pix2pix, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])

# pix2pix inputs
    pix2pix_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    pix2pix_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])    
    pix2pix_img2img_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    pix2pix_inpaint_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    pix2pix_controlnet_input.click(fn=import_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])
    
# pix2pix both
    pix2pix_img2img_both.click(fn=both_to_module, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    pix2pix_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])
    pix2pix_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_pix2pix, negative_prompt_pix2pix, gs_out_pix2pix, sel_out_pix2pix, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet,img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])

# inpaint outputs
    inpaint_img2img.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    inpaint_pix2pix.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])    
    inpaint_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])    
    inpaint_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])        
    inpaint_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])        
    inpaint_resrgan.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    inpaint_gfpgan.click(fn=send_to_module, inputs=[gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    inpaint_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_inpaint, sel_out_inpaint, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])    

# inpaint inputs
    inpaint_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    inpaint_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])    
    inpaint_img2img_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    inpaint_pix2pix_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    inpaint_controlnet_input.click(fn=import_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, tabs, tabs_image])    
    
# inpaint both
    inpaint_img2img_both.click(fn=both_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    inpaint_pix2pix_both.click(fn=both_to_module, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    inpaint_controlnet_both.click(fn=both_to_module_inpaint, inputs=[prompt_inpaint, negative_prompt_inpaint, gs_out_inpaint, sel_out_inpaint, tab_image_num, tab_controlnet_num], outputs=[prompt_controlnet, negative_prompt_controlnet, img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])

# ControlNet outputs
    controlnet_img2img.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    controlnet_pix2pix.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    controlnet_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image]) 
    controlnet_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])    
    controlnet_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    controlnet_resrgan.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    controlnet_gfpgan.click(fn=send_to_module, inputs=[gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    controlnet_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_controlnet, sel_out_controlnet, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])

# controlnet inputs
    controlnet_txt2img_sd_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    controlnet_txt2img_kd_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])
    controlnet_img2img_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, tabs, tabs_image])
    controlnet_pix2pix_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, tabs, tabs_image])
    controlnet_inpaint_input.click(fn=import_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint, tabs, tabs_image])
    controlnet_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    controlnet_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_controlnet, negative_prompt_controlnet, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])    

# ControlNet both 
    controlnet_img2img_both.click(fn=both_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_img2img_num], outputs=[prompt_img2img, negative_prompt_img2img, img_img2img, tabs, tabs_image])
    controlnet_pix2pix_both.click(fn=both_to_module, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_pix2pix_num], outputs=[prompt_pix2pix, negative_prompt_pix2pix, img_pix2pix, tabs, tabs_image])
    controlnet_inpaint_both.click(fn=both_to_module_inpaint, inputs=[prompt_controlnet, negative_prompt_controlnet, gs_out_controlnet, sel_out_controlnet, tab_image_num, tab_inpaint_num], outputs=[prompt_inpaint, negative_prompt_inpaint,img_inpaint, gs_img_inpaint, tabs, tabs_image])

# Faceswap outputs
    faceswap_img2img.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    faceswap_pix2pix.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    faceswap_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image]) 
    faceswap_controlnet.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet,  gs_img_source_controlnet, tabs, tabs_image])     
    faceswap_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_faceswap_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])
    faceswap_resrgan.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    faceswap_gfpgan.click(fn=send_to_module, inputs=[gs_out_faceswap, sel_out_faceswap, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    faceswap_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_faceswap, sel_out_faceswap, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])

# resrgan outputs
    resrgan_img2img.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    resrgan_pix2pix.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    resrgan_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, tabs, tabs_image])    
    resrgan_controlnet.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, gs_img_source_controlnet, tabs, tabs_image])            
    resrgan_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_resrgan, sel_out_resrgan, tab_faceswap_num, tab_inpaint_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])       
    resrgan_gfpgan.click(fn=send_to_module, inputs=[gs_out_resrgan, sel_out_resrgan, tab_image_num, tab_gfpgan_num], outputs=[img_gfpgan, tabs, tabs_image])
    resrgan_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_resrgan, sel_out_resrgan, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])        

# gfpgan outputs
    gfpgan_img2img.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])
    gfpgan_pix2pix.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_pix2pix_num], outputs=[img_pix2pix, tabs, tabs_image])
    gfpgan_inpaint.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_inpaint_num], outputs=[img_inpaint, gs_img_inpaint, gs_img_source_controlnet, tabs, tabs_image])  
    gfpgan_controlnet.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_controlnet_num], outputs=[img_source_controlnet, tabs, tabs_image])                  
    gfpgan_faceswap.click(fn=send_to_module_inpaint, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_faceswap_num, tab_inpaint_num], outputs=[img_target_faceswap, gs_img_target_faceswap, tabs, tabs_image])    
    gfpgan_resrgan.click(fn=send_to_module, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_image_num, tab_resrgan_num], outputs=[img_resrgan, tabs, tabs_image])
    gfpgan_img2txt_git.click(fn=send_to_module_text, inputs=[gs_out_gfpgan, sel_out_gfpgan, tab_text_num, tab_img2txt_git_num], outputs=[img_img2txt_git, tabs, tabs_text])

# Musicgen inputs
    musicgen_audiogen_input.click(fn=import_to_module_audio, inputs=[prompt_musicgen, tab_audio_num, tab_audiogen_num], outputs=[prompt_audiogen, tabs, tabs_audio])
    
# Audiogen inputs    
    audiogen_musicgen_input.click(fn=import_to_module_audio, inputs=[prompt_audiogen, tab_audio_num, tab_musicgen_num], outputs=[prompt_musicgen, tabs, tabs_audio])

# Bark inputs
    bark_whisper.click(fn=send_audio_to_module_text, inputs=[out_bark, tab_text_num, tab_whisper_num], outputs=[source_audio_whisper, tabs, tabs_text])

# Modelscope inputs    
    txt2vid_ms_txt2vid_ze_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_video_num, tab_txt2vid_ze_num], outputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tabs, tabs_video])
    txt2vid_ms_txt2img_sd_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])
    txt2vid_ms_txt2img_kd_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])

# Modelscope inputs    
    txt2vid_ze_txt2vid_ms_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_video_num, tab_txt2vid_ms_num], outputs=[prompt_txt2vid_ms, negative_prompt_txt2vid_ms, tabs, tabs_video])
    txt2vid_ze_txt2img_sd_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_image_num, tab_txt2img_sd_num], outputs=[prompt_txt2img_sd, negative_prompt_txt2img_sd, tabs, tabs_image])    
    txt2vid_ze_txt2img_kd_input.click(fn=import_to_module_video, inputs=[prompt_txt2vid_ze, negative_prompt_txt2vid_ze, tab_image_num, tab_txt2img_kd_num], outputs=[prompt_txt2img_kd, negative_prompt_txt2img_kd, tabs, tabs_image])

# Exécution de l'UI :
    demo.load(split_url_params, nsfw_filter, nsfw_filter, _js=get_window_url_params)
if __name__ == "__main__":
    demo.queue(concurrency_count=8).launch(server_name="0.0.0.0", server_port=7860, ssl_certfile="./ssl/cert.pem", ssl_keyfile="./ssl/key.pem", ssl_verify=False)

# Fin du fichier
