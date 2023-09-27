# https://github.com/Woolverine94/biniou
# whisper.py
import gradio as gr
import os
import time
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
from pydub import AudioSegment
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from ressources.common import *

device_whisper = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_whisper = "./models/whisper/"
os.makedirs(model_path_whisper, exist_ok=True)

model_list_whisper = {}

for filename in os.listdir(model_path_whisper):
    f = os.path.join(model_path_whisper, filename)
    if os.path.isfile(f) and (filename.endswith('.bin') or filename.endswith('.safetensors')) :
        model_list_whisper.update(f)

model_list_whisper_builtin = {
    "openai/whisper-tiny": "model.safetensors",
    "openai/whisper-base": "model.safetensors",        
    "openai/whisper-medium": "model.safetensors",
    "openai/whisper-large": "model.safetensors",
    "openai/whisper-large-v2": "model.safetensors", 
}

model_list_whisper.update(model_list_whisper_builtin)

language_list_whisper = [
    "afrikaans",
    "arabic",
    "armenian",
    "azerbaijani",
    "belarusian",
    "bosnian",
    "bulgarian",
    "catalan",
    "chinese",
    "croatian",
    "czech",
    "danish",
    "dutch",
    "english",
    "estonian",
    "finnish",
    "french",
    "galician",
    "german",
    "greek",
    "hebrew",
    "hindi",
    "hungarian",
    "icelandic",
    "indonesian",
    "italian",
    "japanese",
    "kannada",
    "kazakh",
    "korean",
    "latvian",
    "lithuanian",
    "macedonian",
    "malay",
    "marathi",
    "maori",
    "nepali",
    "norwegian",
    "persian",
    "polish",
    "portuguese",
    "romanian",
    "russian",
    "serbian",
    "slovak",
    "slovenian",
    "spanish",
    "swahili",
    "swedish",
    "tagalog",
    "tamil",
    "thai",
    "turkish",
    "ukrainian",
    "urdu",
    "vietnamese",
    "welsh",
]

# Bouton Cancel
stop_whisper = False

def initiate_stop_whisper() :
    global stop_whisper
    stop_whisper = True

def check_whisper(step, timestep, latents) :
    global stop_whisper
    if stop_whisper == False :
        return
    elif stop_whisper == True :
        stop_whisper = False
        try:
            del ressources.whisper.pipe_whisper
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def convert_seconds_to_timestamp(seconds):
    RELIQUAT0 = int(seconds)
    MSECONDES = round(seconds-(RELIQUAT0), 3)
    MSECONDES_FINAL = str(int(MSECONDES*1000)).ljust(3, '0')
    RELIQUAT1 = int(seconds/60)
    SECONDES = RELIQUAT0-(RELIQUAT1*60)
    RELIQUAT2 = int(RELIQUAT1/60)
    MINUTES = int((RELIQUAT0-((RELIQUAT2*3600)+SECONDES))/60)
    RELIQUAT3 = RELIQUAT2/24
    HEURES = int((RELIQUAT0-((MINUTES*60)+SECONDES))/3600)
    total = f"{str(HEURES).zfill(2)}:{str(MINUTES).zfill(2)}:{str(SECONDES).zfill(2)},{MSECONDES_FINAL}"
    return total

# def download_model(modelid_whisper_final):
#     if modelid_whisper_final[0:9] != "./models/":
#         hf_hub_path_config_whisper = hf_hub_download(
#             repo_id=modelid_whisper_final, 
#             filename="config.json", 
#             repo_type="model", 
#             cache_dir=model_path_whisper, 
#             resume_download=True,
#             local_files_only=True if offline_test() else None
#         )
#         hf_hub_path_whisper = hf_hub_download(
#             repo_id=modelid_whisper_final, 
#             filename=model_list_whisper[modelid_whisper_final], 
#             repo_type="model", 
#             cache_dir=model_path_whisper, 
#             resume_download=True,
#             local_files_only=True if offline_test() else None
#         )
#     return hf_hub_path_config_whisper, hf_hub_path_whisper 

def text_whisper(
    modelid_whisper, 
    srt_output_whisper, 
    source_language_whisper, 
    source_audio_whisper, 
    output_type_whisper, 
    output_language_whisper, 
    progress_whisper=gr.Progress(track_tqdm=True)
    ):
        
    sample_rate_whisper = 16000    
    audio_whisper = AudioSegment.from_file(source_audio_whisper)
    audio_whisper = audio_whisper.set_frame_rate(sample_rate_whisper)
    audio_whisper = audio_whisper.set_channels(1)     
    audio_whisper = audio_whisper.get_array_of_samples()
    audio_whisper = np.array(audio_whisper)
    
    model_whisper = WhisperForConditionalGeneration.from_pretrained(
        modelid_whisper, 
        cache_dir=model_path_whisper, 
        resume_download=True, 
        local_files_only=True if offline_test() else None
    )
    
    tokenizer_whisper = AutoTokenizer.from_pretrained(
        modelid_whisper,
        cache_dir=model_path_whisper, 
        resume_download=True, 
        local_files_only=True if offline_test() else None
    )
    
    feat_ex_whisper = AutoFeatureExtractor.from_pretrained(
        modelid_whisper,
        cache_dir=model_path_whisper, 
        resume_download=True, 
        local_files_only=True if offline_test() else None        
    )
    
    pipe_whisper = pipeline(
        "automatic-speech-recognition", 
        model=model_whisper, 
        tokenizer=tokenizer_whisper, 
        feature_extractor=feat_ex_whisper, 
        chunk_length_s=30, 
        device=device_whisper, 
        torch_dtype=torch.float32
    )
    
    if srt_output_whisper == False :
        transcription_whisper_final = pipe_whisper(audio_whisper.copy(), generate_kwargs={"task": output_type_whisper}, batch_size=8)["text"]
    elif srt_output_whisper == True :
        transcription_whisper = pipe_whisper(
            audio_whisper.copy(), 
            batch_size=8, 
            generate_kwargs={"task": output_type_whisper}, 
            return_timestamps=True
        )["chunks"]
        
        transcription_whisper_final = ""
        
        for i in range(len(transcription_whisper)) : 
            timestamp_start, timestamp_end = transcription_whisper[i]["timestamp"]
            transcribe = transcription_whisper[i]["text"]
            timestamp_start_final = convert_seconds_to_timestamp(timestamp_start)
            timestamp_end_final = convert_seconds_to_timestamp(timestamp_end)
            transcription_whisper_final = transcription_whisper_final+ f"{i+1}"+ "\n"+ f"{timestamp_start_final}"+ " --> "+ f"{timestamp_end_final}"+ "\n"+ transcribe+ "\n"+ "\n"

    write_file(transcription_whisper_final)

    del audio_whisper, model_whisper, tokenizer_whisper, feat_ex_whisper, pipe_whisper
    clean_ram()

    return transcription_whisper_final

