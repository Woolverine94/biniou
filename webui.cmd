REM ***********************
REM *** BINIOU LAUNCHER ***
REM ***********************

call venv.cmd
set path=%path%%userprofile%\AppData\Local\Programs\ffmpeg\ffmpeg-master-latest-win64-gpl\bin;
set AUDIOCRAFT_CACHE_DIR=%userprofile%\biniou\models\Audiocraft\
set HF_HUB_DISABLE_XET=1
python .\webui.py
