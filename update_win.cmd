REM ****************************
REM *** BINIOU UPDATE SCRIPT ***
REM ****************************

echo "Biniou update ..."
git pull

set filename=".ini/llamacpp_backend.cfg"
if exist %filename% (
  set FORCE_CMAKE=1
  set /p CMAKE_ARGS=<%filename%
)

echo "Biniou env update"
call venv.cmd
python -m pip install --upgrade pip==25.2
python -m pip install --upgrade wheel
python -m pip install --upgrade torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m pip uninstall --yes photomaker
python -m pip install --upgrade -r requirements.txt
echo "Update finished ! You could now launch biniou by double-clicking %userprofile%\biniou\webui.cmd"
REM pause

