@echo off
REM Start Thalamus on the RECEIVING laptop. Ports 50070-50072 avoid the
REM Sleeve clock service that already listens on 50050.
REM Edit THALAMUS_VENV below if the virtual environment lives elsewhere.
set THALAMUS_VENV=C:\thalamus_venv
set HERE=%~dp0
"%THALAMUS_VENV%\Scripts\python.exe" -m thalamus.pipeline -c "%HERE%configs\receiver.json" -p 50070 -u 50071 -d 50072
