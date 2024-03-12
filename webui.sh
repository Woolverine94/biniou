#!/bin/bash
## Detection of TCMalloc
RELEASE="$(cat /etc/os-release|grep ^ID)"

if [ "$(echo $RELEASE|grep 'debian')" !=  "" ]
  then
    TCMALLOC_PATH="/lib/x86_64-linux-gnu"
  elif [ "$(echo $RELEASE|grep 'rhel')" !=  "" ]
    then
      TCMALLOC_PATH="/lib64"
fi

TCMALLOC_NAME="$(ls -l $TCMALLOC_PATH/libtcmalloc.so* 2>/dev/null|sed -ne 's/^.*\/\(.*\) ->.*/\1/p')"

## Activate python venv
source ./env/bin/activate

## Launch Biniou
if [ "$TCMALLOC_NAME" != "" ]
  then
    echo ">>>[biniou 🧠]: Detected TCMalloc installation : using it."
    export LD_PRELOAD=$TCMALLOC_PATH/$TCMALLOC_NAME:$LD_PRELOAD
fi

AUDIOCRAFT_CACHE_DIR='./models/Audiocraft/' python3 webui.py

exit 0
