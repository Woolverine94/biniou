#!/bin/bash
echo "Biniou update ..."
git pull

echo "Biniou env update"
source ./env/bin/activate
pip install -U pip
pip install -U wheel
pip install -U torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
FORCE_CMAKE=1 pip install -U llama-cpp-python
pip uninstall -y diffusers
pip install -U -r requirements.txt

