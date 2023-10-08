<h1 align="center">
  <br/>
  <img src="./images/biniou_64.png"/>biniou
  <br/>
</h1>

<p align="center">
  <img src="./pix/biniou.png" alt="biniou screenshot"/>
</p>

<p align="justify">biniou is a self-hosted webui for several kinds of GenAI (generative artificial intelligence). You can generate multimedia contents with AI and use chatbot on your own computer, even without dedicated GPU and starting from 8GB RAM. Can work offline (once deployed and required models downloaded).</p>

<p style="text-align: center;">
<a href="#debian-12---ubuntu-22043">Linux</a> • <a href="#windows-10--windows-11">Windows</a> • <a href="#macos-homebrew-install">macOS (experimental)</a> • <a href="#dockerfile">Docker</a> | <a href="https://github.com/Woolverine94/biniou/wiki">Documentation</a>
</p>

---

## Updates

🆕 **2023-10-09** :  Temporary removed "anything-v4.5-vae-swapped" from the model list for the Stable Diffusion based modules. You could still use the **.safetensors** file by placing it in the folder *./models/Stable_Diffusion/*.<br/>
🆕 **2023-10-08** :  Modelscope bugfix, Chatbot enhancements and new default model for llama-cpp<br/>
🆕 **2023-10-07** :  Compel compatibility and better integration for SDXL in Stable Diffusion, img2img and ControlNet<br/>
🆕 **2023-10-06** : New features, documentation update<br/>
🆕 **2023-10-04** : Bugfixes and settings tuning on image captioning, ControlNet preview not fully displaying and deprecation warnings during install<br/>
🆕 **2023-10-03** : Bugfixes of installer, Text2Video-Zero module and broken zip gallery on Windows systems<br/>
**2023-10-02** : Adding <a href="https://github.com/Woolverine94/biniou/wiki">Wiki documentation</a><br/>
**2023-10-01** : Adding docker support through <a href="#dockerfile">Dockerfile</a><br/>
**2023-09-30** : Adding <a href="#macos-homebrew-install">macOS Homebrew experimental installer</a><br/>
**2023-09-29** : Adding <a href="#windows-10--windows-11">Windows 10/11 installer</a><br/>

---

## Menu
<p align="left">
  • <a href="#features">Features</a><br/>
  • <a href="#prerequisites">Prerequisites</a><br/>
  • <a href="#installation">Installation</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#debian-12---ubuntu-22043">Debian 12 / Ubuntu 22.04.3</a><br/>  
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#windows-10--windows-11">Windows 10 / Windows 11</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#macos-homebrew-install">macOS Homebrew install</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#dockerfile">Dockerfile</a><br/>
  • <a href="#how-to-use">How To Use</a><br/>
  • <a href="#good-to-know">Good to know</a><br/>
  • <a href="#credits">Credits</a><br/>
  • <a href="#license">License</a><br/>
</p>

---

## Features
* **Text generation using  :**
  - llama-cpp based chatbot (uses .gguf models)
  - Microsoft GIT image captioning
  - Whisper speech-to-text
  - nllb translation (200 languages)

* **Image generation and modification using :**
  - Stable Diffusion
  - Kandinsky (require 16GB+ RAM) 
  - Stable Diffusion Img2img
  - Instruct Pix2Pix
  - Stable Diffusion Inpaint
  - Stable Diffusion ControlNet
  - Insight Face faceswapping 
  - Real ESRGAN upscaler
  - GFPGAN face restoration

* **Audio generation using :**
  - Musicgen
  - Audiogen (require 16GB+ RAM)
  - Harmonai
  - Bark text-to-speech 

* **Video generation using :**
  - Modelscope (require 16GB+ RAM)
  - Text2Video-Zero

* **Other features**
  - Communication between modules : send an output as an input to another module
  - Change your model by a simple dropdown menu or download and add it manually 
  - Based on 🤗 Huggingface and gradio
  - Cross platform : Linux, Windows 10/11 and macOS(experimental, via homebrew)
  - Convenient Dockerfile for cloud instances

---

## Prerequisites
* **Minimal hardware :**
  - 64bit CPU
  - 8GB RAM
  - Storage requirements :
    - for Linux : at least 20GB for installation without models.
    - for Windows : at least 30GB for installation without models.
    - for macOS : at least ??GB for installation without models.
  - Storage type : SSD (HDD has not been tested)
  - Internet access (required only for installation) : unlimited bandwith optical fiber internet access

* **Recommended hardware :**
  - Massively multicore 64bit CPU
  - 16GB+ RAM
  - Storage requirements :
    - for Linux : around 100GB for installation including all defaults models.
    - for Windows : around 100GB for installation including all defaults models.
    - for macOS : around ??GB for installation including all defaults models.
  - Storage type : SSD Nvme
  - Internet access (required only for installation) : unlimited bandwith optical fiber internet access

* **Operating system :**
  - a 64 bit OS :
    - Debian 12 
    - Ubuntu 22.04.3 
    - Windows 10 22H2
    - Windows 11 22H2
    - macOS ???

><u>Note :</u> biniou does not still support Cuda or ROCm and thus does not need a dedicated GPU to run. You can install it in a virtual machine.

---

## Installation 

⚠️ As biniou is still in an early stage of development, it is **recommended** to install it in an "expendable" virtual machine ⚠️

### Debian 12 /  Ubuntu 22.04.3

  1. **Install** the pre-requisites as root :
```bash
apt install git pip python3 python3-venv gcc perl make ffmpeg openssl
```

  2. **Clone** this repository as user : 
```bash
git clone https://github.com/Woolverine94/biniou.git
```

  3. **Launch** the installer :
```bash
cd ./biniou
./install.sh
```

### Windows 10 / Windows 11

Windows installation has more prerequisites than linux one, and requires following softwares (that will be installed automatically during the install process) : 
  - Git 
  - Python 
  - OpenSSL
  - Visual Studio Build tools
  - Windows 10/11 SDK
  - Vcredist x86/64
  - ffmpeg
  - ... and all their dependencies.

<p align="justify">It's a lot of changes on your operating system, and this <b>could potentially</b> bring unwanted behaviors on your system, depending on which softwares are already installed on it.</br></br>
⚠️ You should really make a backup of your system and datas before starting the installation process. ⚠️ 
</p>

  1. **Download** [wget](https://eternallybored.org/misc/wget/) for windows :<br/> 
[https://eternallybored.org/misc/wget/1.21.4/64/wget.exe](https://eternallybored.org/misc/wget/1.21.4/64/wget.exe)<br/> 

><u>Note :</u> **DO NOT** move wget from your default downloads folder, it will be used by the install script and is expected to be in the same directory.

  2. **Download and execute** from your default downloads folder :<br/> 
    * **for Windows 10 :** [install_win10.cmd](https://raw.githubusercontent.com/Woolverine94/biniou/main/install_win10.cmd)<br/> 
    * **for Windows 11 :** [install_win11.cmd](https://raw.githubusercontent.com/Woolverine94/biniou/main/install_win11.cmd)<br/> 

*<p align=center>(right-click on the link and select "Save Target/Link as ..." to download)</br><br/></p>*

All the installation is automated, but Windows UAC will ask you confirmation for each software installed during the "prerequisites" phase. 

### macOS Homebrew install

⚠️ Homebrew install is ***theoretically*** compatible with macOS, but has not been tested. Use at your own risk. ⚠️

  1. **Install** [Homebrew](https://brew.sh/) for your operating system
 
  2. **Install** required homebrew "bottles" : 
```bash
brew install git python3 gcc gcc@11 perl make ffmpeg openssl
```

  3. **Install** python virtualenv : 
```bash
python3 -m pip install virtualenv
```

  4. **Clone** this repository as user : 
```bash
git clone https://github.com/Woolverine94/biniou.git
```

  5. **Launch** the installer :
```bash
cd ./biniou
./install.sh
```

### Dockerfile

*These instructions assumes that you already have a configured and working docker environment.*

  1. **Create** the docker image :
```bash
docker build -t biniou https://github.com/Woolverine94/biniou.git
```
 
  2. **Launch** the container : 
```bash
docker run -it -p 7860:7860 \
-v biniou_outputs:/home/biniou/biniou/outputs \
-v biniou_models:/home/biniou/biniou/models \
-v biniou_cache:/home/biniou/.cache/huggingface \
-v biniou_gfpgan:/home/biniou/biniou/gfpgan \
biniou:latest
```

 3. **Access** the webui by the url :<br/>
[https://127.0.0.1:7860](https://127.0.0.1:7860) or [https://127.0.0.1:7860/?__theme=dark](https://127.0.0.1:7860/?__theme=dark) for dark theme (recommended) <br/>
... or replace 127.0.0.1 by ip of your container

><u>Note :</u> to save storage space, the previous container launch command defines common shared volumes for all biniou containers.<br/>

---

## How To Use

  1. **Launch** by executing from the biniou directory :
  - **for Linux :**
```bash
cd /home/$USER/biniou
./webui.sh
```
  - **for Windows :**

<p align="justify">Double-click <b>webui.cmd</b> in the biniou directory (C:\Users\%username%\biniou\). When asked by the UAC, configure the firewall according to your network type to authorize access to the webui

><u>Note :</u> First start could be very slow on Windows 11 (comparing to others OS).

  2. **Access** the webui by the url :<br/>
[https://127.0.0.1:7860](https://127.0.0.1:7860) or [https://127.0.0.1:7860/?__theme=dark](https://127.0.0.1:7860/?__theme=dark) for dark theme (recommended) <br/>
You can also access biniou from any device (including smartphones) on the same LAN/Wifi network by replacing 127.0.0.1 in the url with biniou host ip address.<br/>

  3. **Quit** by using the keyboard shortcut CTRL+C in the Terminal

  4. **Update** this application (biniou + python virtual environment) by running from the biniou directory : 

  - **for Linux :**

```bash
./update.sh
```

  - **for Windows, double-click `update_win.cmd`**

---

## Good to know

* Most frequent cause of crash is not enough memory on the host. Symptom is biniou program closing and returning to/closing the terminal without specific error message. You can use biniou with 8GB RAM, but 16GB at least is recommended to avoid OOM (out of memory) error. 

* biniou use a lot of differents AI models, which requires a lot of space : if you want to use all the modules in biniou, you will need around 100GB of disk space only for the default model of each module. Models are downloaded on the first run of each module or when you select a new model in a module and generate content. Models are stored in the directory /models of the biniou installation. Unused models could be deleted to save some space. 

* Consequently, you will need a fast internet access to download models.

* A backup of every content generated is available inside the /outputs directory of the biniou folder.

* biniou doesn't still use CUDA and only rely on CPU for all operations. It use a specific CPU-only version of pyTorch. The result is a better compatibility with a wide range of hardware, but degraded performances. Depending on your hardware, expect slowness. 

* Defaults settings are selected to permit generation of contents on low-end computers, with the best ratio performance/quality. If you have a configuration above the minimal settings, you could try using other models, increase media dimensions or duration, modify inference parameters or others settings (like token merging for images) to obtain better quality contents.

* biniou is licensed under GNU GPL3, but each model used in biniou has its own license. Please consult each model license to know what you can and cannot do with the models. For each model, you can find a link to the huggingface page of the model in the "About" section of the associated module.

* Don't have too much expectations : biniou is in an early stage of development, and most open source software used in it are in development (some are still experimentals).

* Every biniou modules offers 2 accordions elements **About** and **Settings** :
  - **About** is a quick help features that describes the module and give instructions and tips on how to use it.
  - **Settings** is a panel setting specific to the module that let you configure the generation parameters.

---

## Credits

This application uses the following softwares and technologies :

- [Huggingface 🤗](https://huggingface.co/) : Diffusers and Transformers libraries and almost all the generatives models.
- [Gradio](https://www.gradio.app/) : webUI
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) : python bindings for llama-cpp
- [Microsoft GIT](https://github.com/microsoft/GenerativeImage2Text) : Image2text
- [Whisper](https://openai.com/research/whisper) : speech2text
- [nllb translation](https://ai.meta.com/research/no-language-left-behind/) : language translation
- [Stable Diffusion](https://stability.ai/stable-diffusion) : txt2img, img2img, inpaint, ControlNet, Text2Video-Zero
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) : txt2img
- [Instruct pix2pix](https://www.timothybrooks.com/instruct-pix2pix) : pix2pix
- [Controlnet Auxiliary models](https://github.com/patrickvonplaten/controlnet_aux) : preview models for ControlNet module
- [Insight Face](https://insightface.ai/) : faceswapping
- [Real ESRGAN](https://github.com/xinntao/Real-ESRGAN) : upscaler
- [GFPGAN](https://github.com/TencentARC/GFPGAN) : face restoration
- [Audiocraft](https://audiocraft.metademolab.com/) : musicgen, audiogen
- [Harmonai](https://www.harmonai.org/) : harmonai
- [Bark](https://github.com/suno-ai/bark) : text2speech
- [Modelscope text-to-video-synthesis](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) : txt2vid
- [compel](https://github.com/damian0815/compel) : Prompt enhancement for various `StableDiffusionPipeline`-based modules
- [tomesd](https://github.com/dbolya/tomesd) : Token merging for various `StableDiffusionPipeline`-based modules
- [Python](https://www.python.org/) 
- [PyTorch](https://pytorch.org/)
- [Git](https://git-scm.com/) 
- [ffmpeg](https://ffmpeg.org/)

... and all their dependencies

---

## License

GNU General Public License v3.0

---

> GitHub [@Woolverine94](https://github.com/Woolverine94) &nbsp;&middot;&nbsp;
