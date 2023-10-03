REM ****************************
REM *** BINIOU UPDATE SCRIPT ***
REM ****************************

echo "Biniou update ..."
set path=%path%%ProgramFiles%\Git\cmd;
git pull

echo "Biniou env update"
call venv.cmd
python -m pip install --upgrade wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install --upgrade https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.7/llama_cpp_python-0.2.7-cp311-cp311-win_amd64.whl
python -m pip install --upgrade -r requirements.txt
echo "Update finished ! You could now launch biniou by double-clicking %userprofile%\biniou\webui.cmd"
pause

