FROM debian:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y bash sudo apt-utils git pip python3 python3-venv gcc perl make ffmpeg openssl libtcmalloc-minimal4

# Setup user biniou and use it to install
RUN adduser --disabled-password --gecos '' biniou
USER biniou

# Pull repo
RUN cd /home/biniou && git clone --branch 0.0.1 https://github.com/Woolverine94/biniou.git
WORKDIR /home/biniou/biniou

# Install biniou
RUN ./install.sh
RUN mkdir -p /home/biniou/.cache/huggingface -p /home/biniou/biniou/gfpgan
RUN chmod +x /home/biniou/biniou/webui.sh

# Replace pyTorch cpu-only version by CUDA-enabled one
# RUN . ./env/bin/activate && pip uninstall -y torch torchvision torchaudio && ./update_cuda.sh && deactivate

ENV DEBIAN_FRONTEND=dialog

# Exposing port 7860
EXPOSE 7860/tcp

# Launch at startup
CMD [ "/home/biniou/biniou/webui.sh" ]

