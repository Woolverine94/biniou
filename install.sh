#!/usr/bin/env bash
ENV_PYTHON_TEST="$ENV_BINIOU_PYTHON_VER"
if [ "$ENV_PYTHON_TEST" != "" ]
  then
    PYTHON_VER='$ENV_PYTHON_TEST'
  else
    PYTHON_VER='python3'
fi

mkdir -p ./outputs
mkdir -p ./ssl
mkdir -p ./models/Audiocraft

## Creating self-signed certificate
[ ! -e ./ssl/key.pem ] && openssl req -x509 -newkey rsa:4096 -keyout ./ssl/key.pem -out ./ssl/cert.pem -sha256 -days 3650 -nodes -subj "/C=FR/ST=Paris/L=Paris/O=Biniou/OU=/CN="

## Creating virtual environment
eval $PYTHON_VER -m venv ./env
source ./env/bin/activate

## Install packages :
set -e # stop if any command below fails
pip install -U pip
pip install wheel
pip install -r requirements.txt
FORCE_CMAKE=1 pip install llama-cpp-python
# pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

exit 0
