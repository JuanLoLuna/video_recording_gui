@echo off
REM Start Thalamus on the SATELLITE computer (the one with the camera).
REM Edit THALAMUS_VENV below if the virtual environment lives elsewhere.
set THALAMUS_VENV=C:\thalamus_venv
set HERE=%~dp0
"%THALAMUS_VENV%\Scripts\python.exe" -m thalamus.pipeline -c "%HERE%configs\satellite.json" --open -p 50060 -u 50061 -d 50062
