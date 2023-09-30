#!/bin/bash

mkdir -p ./outputs
mkdir -p ./ssl
mkdir -p ./models/Audiocraft

## Création des clés
openssl req -x509 -newkey rsa:4096 -keyout ./ssl/key.pem -out ./ssl/cert.pem -sha256 -days 3650 -nodes -subj "/C=FR/ST=Paris/L=Paris/O=Biniou/OU=/CN="

## Création de l'environnement virtuel : 
python3 -m venv ./env
source ./env/bin/activate

## Installer pytorch (CPU) :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
FORCE_CMAKE=1 pip install llama-cpp-python==0.2.7
pip install -r requirements.txt

exit 0
