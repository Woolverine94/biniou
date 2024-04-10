#!/bin/bash
if [ -f ".ini/llamacpp_backend.cfg" ]
  then 
    LLAMACPP_ARGS="$(cat .ini/llamacpp_backend.cfg)"
  else
    LLAMACPP_ARGS=""
fi

echo "Biniou update ..."
git pull

echo "Biniou env update"
source ./env/bin/activate
pip install -U pip
pip install -U wheel
pip install -U torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
FORCE_CMAKE=1 CMAKE_ARGS="$LLAMACPP_ARGS" pip install -U llama-cpp-python
pip uninstall -y diffusers
pip install -U -r requirements.txt

