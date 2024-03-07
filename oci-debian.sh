#!/bin/sh
echo ">>>[biniou oci 🧠]: biniou one-click installer for Debian based-distributions"
echo ">>>[biniou 🧠]: Installing prerequisites"
if [ $(lsb_release -si|grep Debian) != "" ]
  then
    su root -c "apt -y install git pip python3 python3-venv gcc perl make ffmpeg openssl google-perftools"
  else
    sudo apt -y install git pip python3 python3-venv gcc perl make ffmpeg openssl google-perftools
fi
echo ">>>[biniou 🧠]: Cloning repository"
git clone https://github.com/Woolverine94/biniou.git --branch main
echo ">>>[biniou 🧠]: Installing Virtual environment"
cd ./biniou
./install.sh
echo "Installation finished. Use ./webui.sh to launch biniou. Press enter to exit"
read dummy
exit 0

