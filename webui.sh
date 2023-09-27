#!/bin/bash
## Activate python venv
source ./env/bin/activate
## Launch Biniou
AUDIOCRAFT_CACHE_DIR='./models/Audiocraft/' python3 webui.py
