# https://github.com/Woolverine94/biniou
# faceswapper.py
import os
import cv2
import copy
import insightface
import onnxruntime
import numpy as np
from PIL import Image
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
from huggingface_hub import snapshot_download, hf_hub_download

# Gestion des modèles
model_path_faceswap = "./models/faceswap/"
os.makedirs(model_path_faceswap, exist_ok=True)

model_list_faceswap = {}

# for filename in os.listdir(model_path_faceswap):
#     f = os.path.join(model_path_faceswap, filename)
#     if os.path.isfile(f) and filename.endswith('.onnx') :
#         print(filename, f)
#         model_list_faceswap.update({f: ""})

model_list_faceswap_builtin = {
    "thebiglaskowski/inswapper_128.onnx": "inswapper_128.onnx",
}

model_list_faceswap.update(model_list_faceswap_builtin)

def download_model(modelid_faceswap):
    if modelid_faceswap[0:9] != "./models/":
        hf_hub_path_faceswap = hf_hub_download(
            repo_id=modelid_faceswap, 
            filename=model_list_faceswap[modelid_faceswap], 
            repo_type="model", 
            cache_dir=model_path_faceswap, 
            local_dir=model_path_faceswap, 
            local_dir_use_symlinks=True, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        modelid_faceswap = hf_hub_path_faceswap
    return modelid_faceswap    

@metrics_decoration
def image_faceswap(
    modelid_faceswap, 
    img_source_faceswap, 
    img_target_faceswap, 
    source_index_faceswap, 
    target_index_faceswap, 
    use_gfpgan_faceswap, 
    progress_txt2img_sd=gr.Progress(track_tqdm=True)
    ):
   
    if source_index_faceswap == "":
        source_index_faceswap = "0"
    if target_index_faceswap == "":
        target_index_faceswap = "0"
    
    model_path = os.path.join(model_path_faceswap, model_list_faceswap[modelid_faceswap])
    modelid_faceswap = download_model(modelid_faceswap)
    source_img = cv2.imread(img_source_faceswap)
    target_img = cv2.imread(img_target_faceswap)
    providers = onnxruntime.get_available_providers()
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root=model_path_faceswap, providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(320, 320))
    face_swapper = insightface.model_zoo.get_model(model_path)
    target_analyze = face_analyser.get(cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR))
    target_faces = sorted(target_analyze, key=lambda x: x.bbox[0])
    num_target_faces = len(target_faces)
    source_analyze = face_analyser.get(cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR))
    source_faces = sorted(source_analyze, key=lambda x: x.bbox[0])
    num_source_faces = len(source_faces)
    temp_frame = copy.deepcopy(target_img)
    source_index_faceswap = source_index_faceswap.split(',')    
    target_index_faceswap = target_index_faceswap.split(',')
    final_image = []
    
    for i in range(len(source_index_faceswap)):
        source_faces_final = int(source_index_faceswap[i])
        target_faces_final = int(target_index_faceswap[i])
        source_face = source_faces[source_faces_final]
        target_face = target_faces[target_faces_final]
        temp_frame = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
    
    temp_frame = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    timestamp = time.time()
    savename = f"outputs/{timestamp}.png"
    
    if use_gfpgan_faceswap == True :
        temp_frame = image_gfpgan_mini(temp_frame)
    
    temp_frame.save(savename)
    final_image.append(temp_frame)
    
    del source_img, target_img, providers, face_analyser, face_swapper, target_analyze, target_faces, source_analyze, source_faces, temp_frame    
    clean_ram()
    
    return final_image, final_image 
