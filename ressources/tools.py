"""
Usage : 
*******
from tools import biniouModelsManager as BMM
a = biniouModelsManager("../models").modelslister()
"""

import os
import sys
import shutil

class biniouModelsManager:
    def __init__(self, models_dir):
            self.models_dir = models_dir

    def human_readable_size(self, size):
        self.size = size
        if 3 < len(str(size)) < 7:
            self.size_final = f"{round((size/1024), 2)} Ko"
        elif 7 <= len(str(size)) < 10:
            self.size_final = f"{round((size/(1024 ** 2)), 2)} Mo"
        elif len(str(size)) >= 10:
            self.size_final = f"{round((size/(1024 ** 3)), 2)} Go"
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
            print(f">>>[Global settings 🛠️ ]: Model {self.delete_model} deleted.")
        return

    def modelsdownloader(self,):
        pass

