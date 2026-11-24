@echo off
REM Start Thalamus on the RECEIVING laptop (Windows).
REM Ports 50070-50072 avoid the Sleeve clock service that already listens on 50050.
REM -C loads thalamus-contrib (needed for the satellite's H264 stream); install it with
REM   "%THALAMUS_VENV%\Scripts\python.exe" -m pip install thalamus-contrib
REM Certificates: copy ca.pem, thalamus.internal.pem and thalamus.internal-key.pem from the
REM satellite's ~/certs folder into %HERE%certs (never into git). Without them the connection
REM to the satellite is refused, because the satellite requires TLS client certificates.
REM Edit THALAMUS_VENV below if the virtual environment lives elsewhere.
set THALAMUS_VENV=C:\thalamus_venv
set HERE=%~dp0
set CERTS=%HERE%certs
"%THALAMUS_VENV%\Scripts\python.exe" -m thalamus.pipeline -C -c "%HERE%configs\receiver.json" -p 50070 -u 50071 -d 50072 --ca "%CERTS%\ca.pem" --cert "%CERTS%\thalamus.internal.pem" --key "%CERTS%\thalamus.internal-key.pem"
