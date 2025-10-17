#!/bin/sh
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for Debian-based distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"
if [ "$(lsb_release -si|grep Debian)" != "" ]
  then
    su root -c "apt -y install git pip python3 python3-venv python3-dev python3-pkgconfig libavformat-dev libavdevice-dev gcc perl make ffmpeg openssl libtcmalloc-minimal4"
  elif [ "$(cat /etc/os-release|grep VERSION_CODENAME|grep noble)" != "" ] || [ "$(cat /etc/os-release|grep UBUNTU_CODENAME|grep noble)" != "" ]
    then
      sudo add-apt-repository -y ppa:deadsnakes/ppa
      sudo apt update
      sudo apt -y install git pip python3.11 python3.11-venv python3.11-dev python3-pkgconfig libavformat-dev libavdevice-dev gcc perl make ffmpeg openssl libtcmalloc-minimal4
  else
    sudo apt -y install git pip python3 python3-venv  python3-pkgconfig libavformat-dev libavdevice-dev gcc perl make ffmpeg openssl libtcmalloc-minimal4
fi
echo ">>>[biniou oci 🧠 ]: Cloning repository"
git clone --branch main https://github.com/Woolverine94/biniou.git
echo ">>>[biniou oci 🧠 ]: Installing Virtual environment"
cd ./biniou
if [ "$(cat /etc/os-release|grep VERSION_CODENAME|grep noble)" != "" ] || [ "$(cat /etc/os-release|grep UBUNTU_CODENAME|grep noble)" != "" ]
  then
    ENV_BINIOU_PYTHON_VER="python3.11" ./install.sh
  else
    ./install.sh
fi

returncode=$?

if [ "$returncode" = 0 ]
  then
    echo ">>>[biniou oci 🧠 ]: Installation succeeded. Use cd biniou && ./webui.sh to launch biniou. Press enter to exit"
  else
    echo ">>>[biniou oci 🧠 ]: Installation failed. Something went wrong. Check the logs. Press enter to exit"
fi

read dummy
exit 0
