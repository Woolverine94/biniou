#!/bin/bash
## Declaring variables
TCMALLOC_NAME=$(ls -Al /lib/x86_64-linux-gnu/libtcmalloc.so* 2>/dev/null|sed -ne 's/^.*\/\(.*\)->.*/\1/p')

## Activate python venv
source ./env/bin/activate

## Launch Biniou
if [ "$TCMALLOC_NAME" != "" ]
  then
    echo "Detected TCMalloc installation : using it."
    export LD_PRELOAD=/lib/x86_64-linux-gnu/$TCMALLOC_NAME:$LD_PRELOAD 
fi

AUDIOCRAFT_CACHE_DIR='./models/Audiocraft/' python3 webui.py
