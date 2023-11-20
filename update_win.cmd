REM ****************************
REM *** BINIOU UPDATE SCRIPT ***
REM ****************************

echo "Biniou update ..."
git pull

echo "Biniou env update"
call venv.cmd
python -m pip install --upgrade pip
python -m pip install --upgrade wheel
python -m pip install --upgrade torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install --upgrade llama-cpp-python
python -m pip install --upgrade -r requirements.txt
echo "Update finished ! You could now launch biniou by double-clicking %userprofile%\biniou\webui.cmd"
pause

