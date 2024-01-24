"""
Usage : 
*******
from tools import biniouModelsManager as BMM
a = biniouModelsManager("../models").modelslister()
"""

import os
import shutil

class biniouModelsManager:
    def __init__(self, models_dir):
            self.models_dir = models_dir

    def human_readable_size(self, size):
        self.size = size
        if 3 < len(str(size)) < 7:
            self.size_final = f"{round((size/1024), 2)} Ko"
        elif 7 <= len(str(size)) < 10:
            self.size_final = f"{round((size/(1024*1024)), 2)} Mo"
        elif len(str(size)) >= 10:
            self.size_final = f"{round((size/(1024*1024*1024)), 2)} Go"
        else:
            self.size_final = f"{size} octets"
        return self.size_final

    def modelslister(self, ):
#        self.models_dir = models_dir
#        filtered_list = {}
        filtered_list = []
        for root, dirs, files in os.walk(self.models_dir, followlinks=False, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                if (name.count(os.path.sep) < 4) and not (".lock") in name and not (".json") in name:
                    size = os.path.getsize(name)
                    size_final = self.human_readable_size(size)
#                    content = {name:(size_final)}
#                    filtered_list.update(content)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
            for name in dirs:
                size = 0
                name = os.path.join(root, name)
                if (2 < name.count(os.path.sep) < 4) and not (".lock") in name and not (".json") in name:
                    for r, d, f in os.walk(name):
                        for fi in f:
                            size += os.path.getsize(os.path.join(r, fi))
                    size_final = self.human_readable_size(size)
#                    content = {name:(size_final)}
#                    filtered_list.update(content)
                    content = f"{name}:{size_final}"
                    filtered_list.append(content)
#        models_list = dict(sorted(filtered_list.items()))
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

