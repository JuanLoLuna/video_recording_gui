@echo off
REM Start Thalamus on the RECEIVING laptop (Windows).
REM Ports 50070-50072 avoid the Sleeve clock service that already listens on 50050.
REM -C loads thalamus-contrib (needed for the satellite's H264 stream); install it with
REM   "%THALAMUS_PYTHON%" -m pip install thalamus-contrib
REM Certificates: copy ca.pem, thalamus.internal.pem and thalamus.internal-key.pem from the
REM satellite's ~/certs folder into %HERE%certs (never into git). Without them the connection
REM to the satellite is refused, because the satellite requires TLS client certificates.
REM Python: set THALAMUS_PYTHON to the python.exe of your Thalamus environment, e.g.
REM   conda env:  set THALAMUS_PYTHON=%USERPROFILE%\miniforge3\envs\thalamus\python.exe
REM   venv:       set THALAMUS_PYTHON=C:\thalamus_venv\Scripts\python.exe
REM Default: the python of the active conda env, else C:\thalamus_venv.
if not defined THALAMUS_PYTHON if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set THALAMUS_PYTHON=%CONDA_PREFIX%\python.exe
if not defined THALAMUS_PYTHON set THALAMUS_PYTHON=C:\thalamus_venv\Scripts\python.exe
set HERE=%~dp0
set CERTS=%HERE%certs
echo Using %THALAMUS_PYTHON%
"%THALAMUS_PYTHON%" -m thalamus.pipeline -C -c "%HERE%configs\receiver.json" -p 50070 -u 50071 -d 50072 --ca "%CERTS%\ca.pem" --cert "%CERTS%\thalamus.internal.pem" --key "%CERTS%\thalamus.internal-key.pem"
