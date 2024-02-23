"""
Usage : 
*******
from tools import biniouModelsManager as BMM
a = biniouModelsManager("../models").modelslister()
"""

import os
import sys
import shutil
import requests as req
import gradio as gr
from tqdm import tqdm
import torch

class biniouModelsManager:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def human_readable_size(self, size):
        self.size = size
        if 3 < len(str(size)) < 7:
            self.size_final = f"{round((size/1024), 2)} KiB"
        elif 7 <= len(str(size)) < 10:
            self.size_final = f"{round((size/(1024 ** 2)), 2)} MiB"
        elif len(str(size)) >= 10:
            self.size_final = f"{round((size/(1024 ** 3)), 2)} GiB"
        else:
            self.size_final = f"{size} octets"
        return self.size_final

    def dirlister_models(self, directory):
        self.directory = directory
        filtered_list = []
        if sys.platform == "win32":
            self.directory = self.directory.replace("/", "\\")
        for root, dirs, files in os.walk(self.directory, followlinks=False, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                if (name.count(os.path.sep) < 4) and not (".lock") in name and not (".json") in name:
                    size = os.path.getsize(name)
                    size_final = self.human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
            for name in dirs:
                name = os.path.join(root, name)
                if (2 < name.count(os.path.sep) < 4) and not (".lock") in name and not (".json") in name:
                    size = 0
                    for r, d, f in os.walk(name, followlinks=False):
                        for fi in f:
                            if os.path.islink(os.path.join(r, fi)) == False:
                                size += os.path.getsize(os.path.join(r, fi))
                    size_final = self.human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
        return filtered_list

    def dirlister_cache(self, directory):
        self.directory = directory
        filtered_list = []
        if sys.platform == "win32":
            self.directory = self.directory.replace("/", "\\")
        for root, dirs, files in os.walk(self.directory, followlinks=False, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                if (name.count(os.path.sep) < 5) and not (".lock") in name and not (".json") in name:
                    size = os.path.getsize(name)
                    size_final = self.human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
            for name in dirs:
                name = os.path.join(root, name)
                if (3 < name.count(os.path.sep) < 5) and not (".lock") in name and not (".json") in name:
                    size = 0
                    for r, d, f in os.walk(name, followlinks=False):
                        for fi in f:
                            if os.path.islink(os.path.join(r, fi)) == False:
                                size += os.path.getsize(os.path.join(r, fi))
                    size_final = self.human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
        return filtered_list

    def modelslister(self, ):
        cachedirname = "../.cache/huggingface/hub"
        filtered_list_models = self.dirlister_models(self.models_dir)
        filtered_list_cache = self.dirlister_cache(cachedirname)
        filtered_list = filtered_list_models + filtered_list_cache
        models_list = sorted(filtered_list)
        return models_list

    def modelsdeleter(self, delete_models_list):
        self.delete_models_list = delete_models_list
        for i in range(len(delete_models_list)):
            self.delete_model = self.delete_models_list[i].split(":")[0]
            if os.path.isfile(self.delete_model):
                os.remove(self.delete_model)
            elif os.path.isdir(self.delete_model):
                shutil.rmtree(self.delete_model)
            print(f">>>[Models cleaner 🧹 ]: Model {self.delete_model} deleted.")
        return

    def modelsdownloader(self,):
        pass

class biniouLoraModelsManager:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def dirlister_models(self, directory):
        self.directory = directory
        filtered_list = []
        if sys.platform == "win32":
            self.directory = self.directory.replace("/", "\\")
        for root, dirs, files in os.walk(self.directory, followlinks=False, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                if (name.count(os.path.sep) < 5) and not (".lock") in name and not (".json") in name:
                    size = os.path.getsize(name)
                    size_final = biniouModelsManager(self.directory).human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
            for name in dirs:
                name = os.path.join(root, name)
                if (3 < name.count(os.path.sep) < 5) and not (".lock") in name and not (".json") in name:
                    size = 0
                    for r, d, f in os.walk(name, followlinks=False):
                        for fi in f:
                            if os.path.islink(os.path.join(r, fi)) == False:
                                size += os.path.getsize(os.path.join(r, fi))
                    size_final = biniouModelsManager(self.directory).human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
        return filtered_list

    def modelslister(self, ):
        self.filtered_list_models = self.dirlister_models(self.models_dir)
        self.models_list = sorted(self.filtered_list_models)
        return self.models_list

    def modelsdeleter(self, delete_models_list):
        self.delete_models_list = delete_models_list
        for i in range(len(delete_models_list)):
            self.delete_model = self.delete_models_list[i].split(":")[0]
            if os.path.isfile(self.delete_model):
                os.remove(self.delete_model)
            elif os.path.isdir(self.delete_model):
                shutil.rmtree(self.delete_model)
            print(f">>>[LoRA models manager 🛠️ ]: LoRA model {self.delete_model} deleted.")
        return

    def modelsdownloader(self, url, progress=gr.Progress(track_tqdm=True)):
        self.url = url
        self.version = self.models_dir.split('/')[-1]
        self.filename = self.url.split('/')[-1]
        self.path = "./models/lora/"+ self.version+ "/"+ self.filename
        with req.get(self.url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(self.path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        print(f">>>[LoRA models manager 🛠️ ]: LoRA model {self.filename} downloaded.")
        return

class biniouTextinvModelsManager:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def dirlister_models(self, directory):
        self.directory = directory
        filtered_list = []
        if sys.platform == "win32":
            self.directory = self.directory.replace("/", "\\")
        for root, dirs, files in os.walk(self.directory, followlinks=False, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                if (name.count(os.path.sep) < 5) and not (".lock") in name and not (".json") in name:
                    size = os.path.getsize(name)
                    size_final = biniouModelsManager(self.directory).human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
            for name in dirs:
                name = os.path.join(root, name)
                if (3 < name.count(os.path.sep) < 5) and not (".lock") in name and not (".json") in name:
                    size = 0
                    for r, d, f in os.walk(name, followlinks=False):
                        for fi in f:
                            if os.path.islink(os.path.join(r, fi)) == False:
                                size += os.path.getsize(os.path.join(r, fi))
                    size_final = biniouModelsManager(self.directory).human_readable_size(size)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
        return filtered_list

    def modelslister(self, ):
        self.filtered_list_models = self.dirlister_models(self.models_dir)
        self.models_list = sorted(self.filtered_list_models)
        return self.models_list

    def modelsdeleter(self, delete_models_list):
        self.delete_models_list = delete_models_list
        for i in range(len(delete_models_list)):
            self.delete_model = self.delete_models_list[i].split(":")[0]
            if os.path.isfile(self.delete_model):
                os.remove(self.delete_model)
            elif os.path.isdir(self.delete_model):
                shutil.rmtree(self.delete_model)
            print(f">>>[Textual inversion manager 🛠️ ]: Textual inversion {self.delete_model} deleted.")
        return

    def modelsdownloader(self, url, progress=gr.Progress(track_tqdm=True)):
        self.url = url
        self.version = self.models_dir.split('/')[-1]
        self.filename = self.url.split('/')[-1]
        self.path = "./models/TextualInversion/"+ self.version+ "/"+ self.filename
        with req.get(self.url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(self.path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        print(f">>>[Textual inversion manager 🛠️ ]: Textual inversion {self.filename} downloaded.")
        return

class biniouSDModelsDownloader:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def modelsdownloader(self, url, progress=gr.Progress(track_tqdm=True)):
        self.url = url
        self.filename = self.url.split('/')[-1]
        self.path = self.models_dir+ "/"+ self.filename
        with req.get(self.url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(self.path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        print(f">>>[SD models downloader 💾 ]: SD model {self.filename} downloaded.")
        return

class biniouGGUFModelsDownloader:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def modelsdownloader(self, url, progress=gr.Progress(track_tqdm=True)):
        self.url = url
        self.filename = self.url.split('/')[-1]
        self.path = self.models_dir+ "/"+ self.filename
        with req.get(self.url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(self.path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        print(f">>>[GGUF models downloader 💾 ]: GGUF model {self.filename} downloaded.")
        return

class biniouUIControl:
    def __init__(self):
        return

    def restart_program():
        print(f">>>[WebUI control 🧠 ]: Restarting biniou ...")
        os.execv(sys.executable, ['python'] + sys.argv)
        return

    def reload_ui():
        print(f">>>[WebUI control 🧠 ]: Reloading WebUI ...")
        return

    def close_program():
        print(f">>>[WebUI control 🧠 ]: Closing biniou. Bye !")
        os._exit(0)
        return

    def detect_optimizer():
        opt = torch.__version__
        pf = sys.platform
        if pf == "linux":
            if "cpu" in opt:
                optimizer = "cpu"
            elif "rocm" in opt:
                optimizer = "rocm"
            else:
                optimizer = "cuda"
        elif pf == "win32":
            if "cpu" in opt:
                optimizer = "cpu"
            else:
                optimizer = "cuda"
        return optimizer

    def biniou_update(optimizer):
        current = biniouUIControl.detect_optimizer()
        untorch = "pip uninstall -y torch torchvision torchaudio"
        if sys.platform == "win32":
            if optimizer == "cpu":
                if current != "cpu":
                    os.system(untorch)
                os.system(".\\update_win.cmd")
            elif optimizer == "cuda":
                if current != "cuda":
                    os.system(untorch)
                os.system(".\\update_win_cuda.cmd")
        elif sys.platform == "linux":
            if optimizer == "cpu":
                if current != "cpu":
                    os.system(untorch)
                os.system("./update.sh")
            elif optimizer == "cuda":
                if current != "cuda":
                    os.system(untorch)
                os.system("./update_cuda.sh")
            elif optimizer == "rocm":
                if current != "rocm":
                    os.system(untorch)
                os.system("./update_rocm.sh")
        print(f">>>[WebUI control 🧠 ]: update for {optimizer} finished.")
        return optimizer
