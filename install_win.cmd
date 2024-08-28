REM ******************************************************
REM *** DEFINING INSTALL PATH FOR BINIOU DIRECTORY : *****
REM *** ONLY USE ABSOLUTE PATH, WITHOUT TRAILING SLASH ***
REM ******************************************************
set DEFAULT_BINIOU_DIR="%userprofile%"

if not exist "%DEFAULT_BINIOU_DIR%" (
  md "%DEFAULT_BINIOU_DIR%"
)

REM **************************************************
REM *** DOWNLOADING AND INSTALLING PREREQUISITES : ***
REM **************************************************
set URL_GIT="https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe"
set URL_OPENSSL="https://download.firedaemon.com/FireDaemon-OpenSSL/FireDaemon-OpenSSL-x64-3.3.1.exe"
set URL_PYTHON="https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set URL_VSBT="https://aka.ms/vs/17/release/vs_BuildTools.exe"
set URL_FFMPEG="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set URL_VCREDIST="https://aka.ms/vs/17/release/vc_redist.x64.exe"
set WIN10_SDK="Microsoft.VisualStudio.Component.Windows10SDK.20348"
set WIN11_SDK="Microsoft.VisualStudio.Component.Windows11SDK.22621"

curl -o "%tmp%\wget.exe" "https://eternallybored.org/misc/wget/1.21.4/64/wget.exe"
%tmp%\wget -O "%tmp%\vs_BuildTools.exe" %URL_VSBT%
%tmp%\wget -O "%tmp%\git.exe" %URL_GIT%
%tmp%\wget -O "%tmp%\openssl.exe" %URL_OPENSSL% --no-check-certificate
%tmp%\wget -O "%tmp%\python.exe" %URL_PYTHON% --no-check-certificate
%tmp%\wget -O "%tmp%\ffmpeg-master-latest-win64-gpl.zip" %URL_FFMPEG%
%tmp%\wget -O "%tmp%\vcredist.exe" %URL_VCREDIST%

start /wait %tmp%\vs_BuildTools.exe --add %WIN10_SDK% --add %WIN11_SDK% --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --passive --wait
start /wait %tmp%\git.exe /silent
start /wait %tmp%\openssl.exe /passive
start /wait %tmp%\python.exe /passive
start /wait %tmp%\vcredist.exe  /q /norestart
start /wait powershell -command "Expand-Archive %tmp%\ffmpeg-master-latest-win64-gpl.zip %userprofile%\AppData\Local\Programs\ffmpeg -Force"

REM ****************************
REM *** CLONING REPOSITORY : ***
REM ****************************
cd "%DEFAULT_BINIOU_DIR%"
set path=%path%;%ProgramW6432%\Git\cmd;
git clone --branch main https://github.com/Woolverine94/biniou.git
git config --global --add safe.directory "%DEFAULT_BINIOU_DIR%/biniou"
cd "%DEFAULT_BINIOU_DIR%\biniou"

REM ******************************
REM *** CREATING DIRECTORIES : ***
REM ******************************
mkdir "%DEFAULT_BINIOU_DIR%\biniou\outputs"
mkdir "%DEFAULT_BINIOU_DIR%\biniou\ssl"
mkdir "%DEFAULT_BINIOU_DIR%\biniou\models\Audiocraft"

REM ***********************************************
REM *** INSTALLING PYTHON VIRTUAL ENVIRONMENT : ***
REM ***********************************************
"%ProgramW6432%\FireDaemon OpenSSL 3\bin\openssl.exe" req -x509 -newkey rsa:4096 -keyout "%DEFAULT_BINIOU_DIR%\biniou\ssl\key.pem" -out "%DEFAULT_BINIOU_DIR%\biniou\ssl\cert.pem" -sha256 -days 3650 -nodes -subj "/C=FR/ST=Paris/L=Paris/O=Biniou/OU=/CN="
"%userprofile%\AppData\Local\Programs\Python\Python311\python.exe" -m pip install --upgrade pip
"%userprofile%\AppData\Local\Programs\Python\Python311\Scripts\pip" install virtualenv
"%userprofile%\AppData\Local\Programs\Python\Python311\python.exe" -m venv env
call venv.cmd
python.exe -m pip install --upgrade pip
pip install wheel
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install llama-cpp-python
pip install -r requirements.txt
echo "Installation finished ! You could now launch biniou by double-clicking %DEFAULT_BINIOU_DIR%\biniou\webui.cmd"
pause
