FROM debian:latest
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y bash sudo apt-utils git pip python3 python3-venv gcc perl make ffmpeg openssl

# Setup user biniou and use it to install
RUN adduser --disabled-password --gecos '' biniou
USER biniou

# Pull repo
RUN cd /home/biniou && git clone https://github.com/Woolverine94/biniou.git
WORKDIR /home/biniou/biniou

# Install biniou
RUN ./install.sh
RUN mkdir -p /home/biniou/.cache/huggingface -p /home/biniou/biniou/gfpgan
RUN chmod +x /home/biniou/biniou/webui.sh

# Exposing port 7860
EXPOSE 7860/tcp

# Launch at startup
CMD [ "/home/biniou/biniou/webui.sh" ]

