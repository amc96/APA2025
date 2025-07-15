#!/bin/bash

echo "========================================"
echo "  Sistema de Otimização de Acasalamento"
echo "========================================"
echo

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERRO: Python não está instalado"
    echo "Por favor, instale Python 3.11 ou superior"
    exit 1
fi

# Usar python3 se disponível, senão python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

echo "Verificando Python... OK"
echo

# Verificar versão do Python
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Versão do Python: $PYTHON_VERSION"

# Verificar se pip está disponível
if ! command -v $PIP_CMD &> /dev/null; then
    echo "ERRO: pip não está disponível"
    echo "Por favor, instale pip ou reinstale Python"
    exit 1
fi

echo "Verificando pip... OK"
echo

# Instalar dependências necessárias
echo "Instalando dependências necessárias..."
echo

$PIP_CMD install streamlit pandas numpy plotly > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERRO: Falha na instalação das dependências"
    echo "Tentando instalar manualmente..."
    $PIP_CMD install streamlit
    $PIP_CMD install pandas
    $PIP_CMD install numpy
    $PIP_CMD install plotly
fi

echo "Dependências instaladas... OK"
echo

# Executar o sistema
echo "Iniciando sistema de otimização..."
echo
echo "O sistema será aberto no seu navegador padrão"
echo "Para parar o sistema, pressione Ctrl+C"
echo

# Aguardar um pouco e abrir o navegador (se disponível)
sleep 3
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000 2>/dev/null &
elif command -v open &> /dev/null; then
    open http://localhost:5000 2>/dev/null &
fi

# Executar Streamlit
streamlit run app.py --server.port 5000 --server.address 0.0.0.0

echo
echo "Sistema encerrado."