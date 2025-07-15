@echo off
echo ========================================
echo  Sistema de Otimização de Acasalamento
echo ========================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Python não está instalado ou não está no PATH
    echo Por favor, instale Python 3.11 ou superior
    pause
    exit /b 1
)

echo Verificando Python... OK
echo.

REM Verificar se pip está disponível
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: pip não está disponível
    echo Por favor, reinstale Python com pip
    pause
    exit /b 1
)

echo Verificando pip... OK
echo.

REM Instalar dependências necessárias
echo Instalando dependências necessárias...
echo.

pip install streamlit pandas numpy plotly >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Falha na instalação das dependências
    echo Tentando instalar manualmente...
    pip install streamlit
    pip install pandas
    pip install numpy
    pip install plotly
)

echo Dependências instaladas... OK
echo.

REM Executar o sistema
echo Iniciando sistema de otimização...
echo.
echo O sistema será aberto no seu navegador padrão
echo Para parar o sistema, pressione Ctrl+C
echo.

REM Aguardar um pouco e abrir o navegador
timeout /t 3 >nul
start http://localhost:5000

REM Executar Streamlit
streamlit run app.py --server.port 5000 --server.address 0.0.0.0

echo.
echo Sistema encerrado.
pause