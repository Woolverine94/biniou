#!/bin/sh
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for Red-Hat-based distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    if [ "$(cat /etc/os-release|grep "rhel")" != "" ]
      then
        su root -c "dnf -y install dnf-plugins-core epel-release; dnf config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; dnf config-manager --set-enabled crb; dnf -y install git python3.11 python3.11-pip python3-virtualenv python3.11-devel gcc perl make ffmpeg openssl gperftools-libs"
    elif [ "$(cat /etc/os-release|grep "fedora")" != "" ]
      then
        su root -c "dnf -y install dnf-plugins-core; dnf -y install git python3.11 pip python3-virtualenv python3.11-devel gcc perl make ffmpeg-free openssl gperftools-libs"
    fi
else
    if [ "$(cat /etc/os-release|grep "rhel")" != "" ]
      then
        sudo sh -c "dnf -y install dnf-plugins-core epel-release; dnf config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; dnf config-manager --set-enabled crb; dnf -y install git python3.11 python3.11-pip python3-virtualenv python3.11-devel gcc perl make ffmpeg openssl gperftools-libs"
    elif [ "$(cat /etc/os-release|grep "fedora")" != "" ]
      then
        sudo sh -c "dnf -y install dnf-plugins-core; dnf -y install git python3.11 pip python3-virtualenv python3.11-devel gcc perl make ffmpeg-free openssl gperftools-libs"
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
    if [ "$(cat /etc/os-release|grep "rhel")" != "" ]
      then
        su root -c "firewall-cmd --zone=public --permanent --add-port 7860/tcp; firewall-cmd --reload"
    elif [ "$(cat /etc/os-release|grep "fedora")" != "" ]
      then
        su root -c "firewall-cmd --permanent --add-port 7860/tcp; firewall-cmd --reload"
    fi
else
    if [ "$(cat /etc/os-release|grep "rhel")" != "" ]
      then
        sudo sh -c "firewall-cmd --zone=public --permanent --add-port 7860/tcp; firewall-cmd --reload"
    elif [ "$(cat /etc/os-release|grep "fedora")" != "" ]
      then
        sudo sh -c "firewall-cmd --permanent --add-port 7860/tcp; firewall-cmd --reload"
    fi
fi

echo ">>>[biniou oci 🧠 ]: Installation finished. Use cd biniou && ./webui.sh to launch biniou. Press enter to exit"
read dummy
exit 0
