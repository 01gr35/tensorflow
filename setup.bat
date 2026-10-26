@echo off
echo ===================================================
echo Iniciando la instalacion del entorno de TensorFlow...
echo ===================================================

:: 1. Activar o crear el entorno virtual
if exist .venv\Scripts\activate.bat (
    echo Activando el entorno virtual .venv...
    call .venv\Scripts\activate.bat
) else if exist tf-env-312\Scripts\activate.bat (
    echo Activando el entorno virtual tf-env-312...
    call tf-env-312\Scripts\activate.bat
) else (
    echo Creando entorno virtual .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

:: 2. Actualizar pip
echo Actualizando pip...
python -m pip install --upgrade pip

:: 3. Instalar los requerimientos
echo Instalando librerias desde requirements.txt...
pip install -r requirements.txt

echo.
echo ===================================================
echo ¡Instalacion completada con exito!
echo ===================================================
pause
