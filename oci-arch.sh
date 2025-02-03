#!/bin/sh
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for Arch-based distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"

if [ "$(cat /etc/os-release|grep "cachyos")" != "" ]
  then
    if [ "$(groups|grep 'wheel')" == "" ]
      then
        su root -c "pacman -Sy --noconfirm git python311 python-pip python-pipenv perl make cmake ffmpeg openssl lib32-gperftools;"
      else
        sudo sh -c "pacman -Sy --noconfirm git python311 python-pip python-pipenv perl make cmake ffmpeg openssl lib32-gperftools;"
    fi
fi

echo ">>>[biniou oci 🧠 ]: Cloning repository"
git clone --branch main https://github.com/Woolverine94/biniou.git
echo ">>>[biniou oci 🧠 ]: Installing Virtual environment"
cd ./biniou
ENV_BINIOU_PYTHON_VER="python3.11" ./install.sh

echo ">>>[biniou oci 🧠 ]: Opening port 7860/tcp and restarting firewall"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    su root -c "ufw allow 7860/tcp"
  else
    sudo sh -c "ufw allow 7860/tcp"
fi

echo ">>>[biniou oci 🧠 ]: Installation finished. Use cd biniou && ./webui.sh to launch biniou. Press enter to exit"
read dummy
exit 0

