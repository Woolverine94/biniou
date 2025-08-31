#!/bin/bash
## Detection of TCMalloc
RELEASE="$(cat /etc/os-release|grep ^ID)"

if [ "$(echo $RELEASE|grep 'debian')" !=  "" ]
  then
    TCMALLOC_PATH="/lib/x86_64-linux-gnu"
  elif [ "$(echo $RELEASE|grep 'rhel')" !=  "" ] || [ "$(echo $RELEASE|grep 'fedora')" !=  "" ] || [ "$(echo $RELEASE|grep 'openmandriva')" !=  "" ] 
    then
      TCMALLOC_PATH="/lib64"
  elif [ "$(echo $RELEASE|grep 'opensuse')" !=  "" ]
    then
      TCMALLOC_PATH="/usr/lib64"
  elif [ "$(echo $RELEASE|grep 'cachyos')" !=  "" ]
    then
      TCMALLOC_PATH="/usr/lib64"
fi

if [ "$(ls -l $TCMALLOC_PATH/libtcmalloc.so* 2>/dev/null)" != "" ]
  then
    TCMALLOC_NAME="$(ls -l $TCMALLOC_PATH/libtcmalloc.so* 2>/dev/null|sed -ne 's/^.*\/\(.*\) ->.*/\1/p'|tail -n 1)"
elif [ "$(ls -l $TCMALLOC_PATH/libtcmalloc_minimal.so* 2>/dev/null)" != "" ]
  then
    TCMALLOC_NAME="$(ls -l $TCMALLOC_PATH/libtcmalloc_minimal.so* 2>/dev/null|sed -ne 's/^.*\/\(.*\) ->.*/\1/p'|tail -n 1)"
fi

## Activate python venv
source ./env/bin/activate

## Launch Biniou
if [ "$TCMALLOC_NAME" != "" ]
  then

    if [ "$(echo $TCMALLOC_NAME|grep 'minimal')" != "" ]
      then
        echo ">>>[biniou 🧠]: Detected TCMalloc_minimal installation : using it."
    else
        echo ">>>[biniou 🧠]: Detected TCMalloc installation : using it."
    fi

    export LD_PRELOAD=$TCMALLOC_PATH/$TCMALLOC_NAME:$LD_PRELOAD
fi

HF_HUB_DISABLE_XET=1 AUDIOCRAFT_CACHE_DIR='./models/Audiocraft/' python3 webui.py

exit 0
