#!/bin/sh
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for Debian based-distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    if [ "$(cat /etc/os-release|grep "Tumbleweed")" != "" ]
      then
        su root -c "zypper --non-interactive install git-core python311 python3-virtualenv python3-pip python311-devel perl make cmake ffmpeg openssl libtcmalloc_minimal4 libgthread-2_0-0; zypper --non-interactive install -t pattern devel_basis"
    else
        su root -c "zypper --non-interactive install git-core python311 python3-virtualenv python3-pip python311-devel gcc11 perl make cmake ffmpeg openssl libtcmalloc_minimal4 libgthread-2_0-0; zypper --non-interactive install -t pattern devel_basis"
    fi
  else
    if [ "$(cat /etc/os-release|grep "Tumbleweed")" != "" ]
      then
        sudo sh -c "zypper --non-interactive install git-core python311 python3-virtualenv python3-pip python311-devel perl make cmake ffmpeg openssl libtcmalloc_minimal4 libgthread-2_0-0; zypper --non-interactive install -t pattern devel_basis"
    else
        sudo sh -c "zypper --non-interactive install git-core python311 python3-virtualenv python3-pip python311-devel gcc11 perl make cmake ffmpeg openssl libtcmalloc_minimal4 libgthread-2_0-0; zypper --non-interactive install -t pattern devel_basis"
    fi
fi

echo ">>>[biniou oci 🧠 ]: Cloning repository"
git clone --branch 0.0.1 https://github.com/Woolverine94/biniou.git
echo ">>>[biniou oci 🧠 ]: Installing Virtual environment"
cd ./biniou
ENV_BINIOU_PYTHON_VER="python3.11" ./install.sh

echo ">>>[biniou oci 🧠 ]: Opening port 7860/tcp and restarting firewall"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    su root -c "firewall-cmd --zone=public --permanent --add-port 7860/tcp; firewall-cmd --reload"
  else
    sudo sh -c "firewall-cmd --zone=public --permanent --add-port 7860/tcp; firewall-cmd --reload"
fi

echo ">>>[biniou oci 🧠 ]: Installation finished. Use cd biniou && ./webui.sh to launch biniou. Press enter to exit"
read dummy
exit 0

