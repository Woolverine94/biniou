#!/usrbin/env bash
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
pip install -U torch==2.1.0 torchvision torchaudio
pip install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip uninstall -y photomaker
pip install -U -r requirements.txt

