<h1 align="center">
  <br>
  biniou
  <br>
</h1>

<p align="center">
  <img src="./pix/biniou.gif" alt="biniou screenshot"/>
</p>

<h4 align="justify">biniou is a self-hosted webui for several kinds of GenAI (generative artificial intelligence). You can generate multimedia contents with AI and use chatbot on your own computer, even without dedicated GPU and starting from 8GB RAM. Can work offline (once deployed and required models downloaded).</h4>

## Menu
<p align="left">
  • <a href="#features">Features</a><br/>
  • <a href="#installation">Installation</a><br/>
  • <a href="#how-to-use">How To Use</a><br/>
  • <a href="#good-to-know">Good to know</a><br/>
  • <a href="#credits">Credits</a><br/>
  • <a href="#license">License</a><br/>
</p>


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
  - Based on Huggingface 🤗 and gradio
  - Cross platform (to be done)

## Installation 

⚠️ As biniou is still in a early stage of development, it is highly recommended to install it in an "expendable" virtual machine ⚠️

### Debian 12 /  Ubuntu 22.04

  1. Install the pre-requisites as root :
```bash
apt install git pip python3 python3-venv gcc perl make ffmpeg openssl
```

  2. Clone this repository as user : 
```bash
git clone https://github.com/Woolverine94/biniou.git
```

  3. Access the directory and launch the installer :
```bash
cd ./biniou
./install.sh
```

## How To Use

  1. Run this application by executing from the biniou directory : 
```bash
./webui.sh
```

  2. Once loaded, access the webui by the url :<br/>
[https://127.0.0.1:7860](https://127.0.0.1:7860)<br/>
  url for dark theme (recommended) :<br/>
[https://127.0.0.1:7860/?__theme=dark](https://127.0.0.1:7860/?__theme=dark)<br/>

- You can also access biniou from any device (including smartphone) on the same LAN/Wifi network of the biniou host using : <br/>
https://<biniou_host_ip>/<br/> 
or<br/> 
https://<biniou_host_ip>/?__theme=dark

- Update this application (biniou + python virtual environment) by running from the biniou directory : 
```bash
./update.sh
```

## Good to know

- Most frequent cause of crash is not enough memory on the host. Symptom is biniou program closing and returning to the terminal without specific error message. You can use biniou with 8GB RAM, but 16GB at least is recommended to avoid OOM (out of memory) error. 

- biniou use a lot of differents AI models, which requires a lot of space : if you want to use all the modules in biniou, you will need around 100GB of disk space for the default associated models. Models are downloaded on the first run of each module or when you select a new model in a module. Models are stored in the directory /models of the biniou installation. Unused models could be deleted to save some space. 

- Consequently, you will need a fast internet access to download models.

- A backup of every content generated is available inside the /outputs directory of the biniou folder.

- biniou doesn't still use CUDA and only rely on CPU for all operations. It use a specific CPU-only version of pyTorch. The result is a better compatibility with a wide range of hardware, but degraded performances. Depending on your hardware, expect slowness. 

- Defaults settings are selected to permit generation of contents on low-end computers, with the best ratio performance/quality. If you have a configuration above the minimal settings, you could try using other models, increase media dimensions or duration, modify inference parameters or others settings (like token merging for images) to obtain better quality contents.

- biniou is licensed under GNU GPL3, but each model used in biniou has its own license. Please consult each model license to know what you can and cannot do with the models. For each model, you can find a link to the huggingface page of the model in the "About" section of the associated module.

- Don't have too much expectations : biniou is in an early stage of development, and most open source software used in it are in development (some are still experimentals).

## Credits

This application uses the following technologies :

(to be done)

## License

GNU General Public License v3.0

---

> GitHub [@Woolverine94](https://github.com/Woolverine94) &nbsp;&middot;&nbsp;
