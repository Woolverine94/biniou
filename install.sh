#!/bin/bash
PYVER=$(python3 --version|sed -ne 's/^.* \([0-9]\)\.\([0-9]*\)\.[0-9]*/\1\2/p')

mkdir -p ./outputs
mkdir -p ./ssl
mkdir -p ./models/Audiocraft

## Creating self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout ./ssl/key.pem -out ./ssl/cert.pem -sha256 -days 3650 -nodes -subj "/C=FR/ST=Paris/L=Paris/O=Biniou/OU=/CN="

## Creating virtual environment
python3 -m venv ./env
source ./env/bin/activate

## Install packages :
pip install -U pip
pip install wheel
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.24/llama_cpp_python-0.2.24-cp${PYVER}-cp${PYVER}-manylinux_2_17_x86_64.whl
# FORCE_CMAKE=1 pip install llama-cpp-python==0.2.13
pip install -r requirements.txt

exit 0
