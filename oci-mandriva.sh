#!/bin/sh
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for OpenMandriva distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    su root -c "dnf -y install dnf-plugins-core; dnf -y install git python python-pip python-virtualenv gcc clang glibc-static-devel perl make ffmpeg openssl lib64tcmalloc_minimal4"
  else
    sudo sh -c "dnf -y install dnf-plugins-core; dnf -y install git python python-pip python-virtualenv gcc clang glibc-static-devel perl make ffmpeg openssl lib64tcmalloc_minimal4"
fi

echo ">>>[biniou oci 🧠 ]: Cloning repository"
git clone --branch main https://github.com/Woolverine94/biniou.git
echo ">>>[biniou oci 🧠 ]: Installing Virtual environment"
cd ./biniou
ENV_BINIOU_PYTHON_VER="python3.11" ./install.sh

# echo ">>>[biniou oci 🧠 ]: Opening port 7860/tcp and restarting firewall"
# 
# if [ "$(groups|grep 'wheel')" == "" ]
#   then 
#     su root -c "firewall-cmd --permanent --add-port 7860/tcp; firewall-cmd --reload"
#   else
#     sudo sh -c "firewall-cmd --permanent --add-port 7860/tcp; firewall-cmd --reload"
# fi

echo ">>>[biniou oci 🧠 ]: Installation finished. Use cd biniou && ./webui.sh to launch biniou. Press enter to exit"
read dummy
exit 0
