#!/bin/bash

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
FORCE_CMAKE=1 pip install llama-cpp-python
pip install -r requirements.txt

exit 0
