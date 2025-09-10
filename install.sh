#!/bin/bash
ENV_PYTHON_TEST="$ENV_BINIOU_PYTHON_VER"
if [ "$ENV_PYTHON_TEST" != "" ]
  then
    PYTHON_VER='$ENV_PYTHON_TEST'
  else
    PYTHON_VER='python3'
fi

# Exit if python > 3.11 and python3.11 not found
if [ "$(python3 --version|sed -ne 's/^.*[0-9]*\.\([0-9]*\)\.[0-9]*.*$/\1/p')" -gt 11 ] && [ "$(which python3.11)" == "" ]
  then
    echo "Error : biniou requires python<=3.11. Try the One-Click Installer for your distribution to correct this."
    exit 1
fi

mkdir -p ./outputs
mkdir -p ./ssl
mkdir -p ./models/Audiocraft

## Creating self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout ./ssl/key.pem -out ./ssl/cert.pem -sha256 -days 3650 -nodes -subj "/C=FR/ST=Paris/L=Paris/O=Biniou/OU=/CN="

## Creating virtual environment
eval $PYTHON_VER -m venv ./env
source ./env/bin/activate

## Install packages :
pip install -U pip
pip install wheel
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
FORCE_CMAKE=1 pip install llama-cpp-python
pip install -r requirements.txt

exit $?
