#!/bin/bash
PYVER=$(python3 --version|sed -ne 's/^.* \([0-9]\)\.\([0-9]*\)\.[0-9]*/\1\2/p')

echo "Biniou update ..."
git pull

echo "Biniou env update"
source ./env/bin/activate
pip install -U pip
pip install -U wheel
pip install -U torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.24/llama_cpp_python-0.2.24-cp${PYVER}-cp${PYVER}-manylinux_2_17_x86_64.whl
# FORCE_CMAKE=1 pip install -U llama-cpp-python
pip install -U -r requirements.txt

