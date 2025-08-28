@echo off
echo Starting Diabetic Patient Readmission Analysis UI...
echo.

REM Check if Streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit is not installed. Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Launching Streamlit application...
echo Open your browser and go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py

pause
