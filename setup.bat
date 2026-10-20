@echo off
echo ===================================================
echo Iniciando la instalacion del entorno de TensorFlow...
echo ===================================================

:: 1. Activar el entorno virtual (ajusta el nombre si es diferente)
echo Activando el entorno virtual tf-env-312...
call tf-env-312\Scripts\activate

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