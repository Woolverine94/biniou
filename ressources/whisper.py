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
    "distil-whisper/distil-large-v2": "model.safetensors", 
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
    reliquat0 = int(seconds)
    reliquat1 = int(seconds/60)
    reliquat2 = int(reliquat1/60)
    msecondes = round(seconds-(reliquat0), 3)
    msecondes_final = str(int(msecondes*1000)).ljust(3, '0')
    secondes = reliquat0-(reliquat1*60)
    minutes = int((reliquat0-((reliquat2*3600)+secondes))/60)
    heures = int((reliquat0-((minutes*60)+secondes))/3600)
    total = f"{str(heures).zfill(2)}:{str(minutes).zfill(2)}:{str(secondes).zfill(2)},{msecondes_final}"
    return total

@metrics_decoration
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
        low_cpu_mem_usage=True,
        resume_download=True, 
        local_files_only=True if offline_test() else None
    )
    model_whisper = model_whisper.to_bettertransformer()

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
        transcription_whisper_final = pipe_whisper(
            audio_whisper.copy(), 
            generate_kwargs={"task": output_type_whisper}, 
            batch_size=8
        )["text"]
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

    filename_whisper = write_file(transcription_whisper_final)

    del audio_whisper, model_whisper, tokenizer_whisper, feat_ex_whisper, pipe_whisper
    clean_ram()

    return transcription_whisper_final
