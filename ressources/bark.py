# https://github.com/Woolverine94/biniou
# bark.py
import gradio as gr
import os
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav
import random
from ressources.common import *

device_label_bark, model_arch = detect_device()
device_bark = torch.device(device_label_bark)

model_path_bark = "./models/Bark/"
os.makedirs(model_path_bark, exist_ok=True)

model_list_bark = [
    "suno/bark-small",
    "suno/bark",
]

voice_preset_list_bark = {
    "DE Male": "v2/de_speaker_9",
    "DE Female": "v2/de_speaker_8",
    "EN Male": "v2/en_speaker_6", 
    "EN Female": "v2/en_speaker_9",
    "ES Male": "v2/es_speaker_7",
    "ES Female": "v2/es_speaker_9",
    "FR Male" : "v2/fr_speaker_8",
    "FR Female": "v2/fr_speaker_5",
    "HI Male": "v2/hi_speaker_8",
    "HI Female": "v2/hi_speaker_9",
    "JA Male": "v2/ja_speaker_6",
    "JA Female": "v2/ja_speaker_7",
    "KO Male": "v2/ko_speaker_9",
    "KO Female": "v2/ko_speaker_0",
    "PL Male": "v2/pl_speaker_8",
    "PL Female": "v2/pl_speaker_9",
    "PT Male": "v2/pt_speaker_9",
    "RU Male": "v2/ru_speaker_7",
    "RU Female": "v2/ru_speaker_5",
    "TR Male": "v2/tr_speaker_9",
    "TR Female": "v2/tr_speaker_5",
    "ZH Male": "v2/zh_speaker_8",
    "ZH Female": "v2/zh_speaker_9",
}

@metrics_decoration
def music_bark(
    prompt_bark, 
    model_bark, 
    voice_preset_bark, 
    progress_bark=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Bark 🗣️ ]: starting module")

    processor = AutoProcessor.from_pretrained(
        model_bark, 
        cache_dir=model_path_bark, 
        torch_dtype=model_arch,
        resume_download=True,
        local_files_only=True if offline_test() else None
    )

    pipe_bark = BarkModel.from_pretrained(
        model_bark, 
        cache_dir=model_path_bark,
        torch_dtype=model_arch,
        resume_download=True,
        local_files_only=True if offline_test() else None        
        )
    pipe_bark = pipe_bark.to(device_bark)

#    pipe_bark = BetterTransformer.transform(pipe_bark, keep_original_model=False)
    if device_label_bark == "cuda" :
        pipe_bark.enable_cpu_offload()
    pipe_bark = pipe_bark.to_bettertransformer()

    voice_preset = voice_preset_list_bark[voice_preset_bark]
    inputs = processor(prompt_bark, voice_preset=voice_preset)
    audio_array = pipe_bark.generate(**inputs, do_sample=True)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = pipe_bark.generation_config.sample_rate
    savename = f"outputs/{timestamper()}.wav"
    write_wav(savename, sample_rate, audio_array)

    print(f">>>[Bark 🗣️ ]: generated 1 audio file")
    reporting_bark = f">>>[Bark 🗣️ ]: "+\
        f"Settings : Model={model_bark} | "+\
        f"Voice preset={voice_preset_bark} | "+\
        f"Prompt={prompt_bark}"
    print(reporting_bark) 
   
    del processor, pipe_bark, audio_array    
    clean_ram()

    print(f">>>[Bark 🗣️ ]: leaving module")    
    return savename
