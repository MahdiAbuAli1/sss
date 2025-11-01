@echo off
cls
echo =================================================================
echo.
echo           SYSTEM CHECK & DIAGNOSTICS SCRIPT FOR RAG SYSTEM
echo.
echo =================================================================
echo.
echo This script will test Conda, Ollama, and Python environments.
echo Press any key to start...
pause > nul
cls

:: Section 1: Checking Core Installations
echo [SECTION 1/5] Checking Core Installations...
echo --------------------------------------------
where conda > nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Conda is found in the system path.
) else (
    echo [ERROR] Conda is NOT found. Please ensure Miniconda is installed and its path is correct.
    goto :end_script
)

where ollama > nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is found in the system path.
) else (
    echo [ERROR] Ollama is NOT found. Please ensure Ollama is installed.
    goto :end_script
)
echo.
echo Press any key to continue...
pause > nul
cls

:: Section 2: Checking Conda Environments
echo [SECTION 2/5] Checking Conda Environments...
echo ------------------------------------------
echo Listing all available Conda environments:
call conda env list
echo.
echo Please verify that 'support_env' and 'agent_env' appear in the list above.
echo.
echo Press any key to continue...
pause > nul
cls

:: Section 3: Checking Ollama Models
echo [SECTION 3/5] Checking Ollama Models...
echo --------------------------------------
echo Listing all available Ollama models:
call ollama list
echo.
echo Please verify that your models (e.g., qwen3:4b, qwen3-embedding) appear in the list.
echo If the list is empty, the models were not copied correctly to C:\Users\mahdi\.ollama
echo.
echo Press any key to continue...
pause > nul
cls

:: Section 4: Testing Primary Environment 'support_env'
echo [SECTION 4/5] Testing 'support_env' and its libraries...
echo --------------------------------------------------------
echo Activating support_env...
call conda activate support_env

echo.
echo [TEST] Checking Python version...
call python --version

echo.
echo [TEST] Checking 'torch' (CPU version)...
call python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Torch is using CPU: {not torch.cuda.is_available()}')"

echo.
echo [TEST] Checking 'langchain'...
call python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"

echo.
echo [TEST] Checking 'unstructured' for document processing...
call python -c "from unstructured.partition.pdf import partition_pdf; print('Unstructured (for PDF) seems OK.')"

echo.
echo [TEST] Checking 'async_lru' (the library you added)...
call python -c "import async_lru; print('async_lru library is installed correctly.')"

echo.
echo Deactivating environment...
call conda deactivate
echo.
echo Press any key to continue...
pause > nul
cls

:: Section 5: Final Report and Next Steps
echo [SECTION 5/5] Final Report and Next Steps...
echo -------------------------------------------
echo.
echo The diagnostic script has finished.
echo.
echo - If you saw [SUCCESS] everywhere, your system is likely ready.
echo - If you saw any [ERROR] messages, please review the specific section to fix the issue.
echo.
echo Next Step:
echo 1. Open a NEW Anaconda Prompt.
echo 2. Activate your environment: conda activate support_env
echo 3. Navigate to your project folder: cd %USERPROFILE%\support_service_platform
echo 4. Run your application: uvicorn main:app
echo.

:end_script
echo =================================================================
echo.
echo                           END OF SCRIPT
echo.
echo =================================================================
pause
