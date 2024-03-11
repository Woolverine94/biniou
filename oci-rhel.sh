#!/bin/sh
ORIG_PYTHON="$(ls -l $(which python3)|sed -ne 's/^.* -> \(.*\)$/\1/p')"
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for Red-Hat based-distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    su root -c "dnf -y install dnf-plugins-core epel-release; dnf config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; dnf config-manager --set-enabled crb; dnf -y install git python3.11 python3.11-pip python3-virtualenv python3.11-devel gcc perl make ffmpeg openssl google-perftools; echo \">>>[biniou oci 🧠 ]: Modifying default python version\"; rm /usr/bin/python3; ln -s /usr/bin/python3.11 /usr/bin/python3"
  else
    sudo sh -c "dnf -y install dnf-plugins-core epel-release; dnf config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; dnf config-manager --set-enabled crb; dnf -y install git python3.11 python3.11-pip python3-virtualenv python3.11-devel gcc perl make ffmpeg openssl google-perftools; echo \">>>[biniou oci 🧠 ]: Modifying default python version\"; rm /usr/bin/python3; ln -s /usr/bin/python3.11 /usr/bin/python3"
fi

echo ">>>[biniou oci 🧠 ]: Cloning repository"
git clone https://github.com/Woolverine94/biniou.git --branch main
echo ">>>[biniou oci 🧠 ]: Installing Virtual environment"
cd ./biniou
./install.sh
echo ">>>[biniou oci 🧠 ]: Restoring default python version"

if [ "$(groups|grep 'wheel')" == "" ]
  then
    su root -c "rm /usr/bin/python3; ln -s $ORIG_PYTHON /usr/bin/python3"
else 
    sudo sh -c "rm /usr/bin/python3; ln -s $ORIG_PYTHON /usr/bin/python3"
fi

rm ./env/bin/python3
ln -s /usr/bin/python3.11 ./env/bin/python3

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

