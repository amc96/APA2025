# Como Executar o Sistema de Otimização

## Execução Rápida

### Windows
1. Execute o arquivo `executar_sistema.bat`
2. O sistema irá automaticamente:
   - Verificar se Python está instalado
   - Instalar as dependências necessárias
   - Iniciar o sistema na porta 5000
   - Abrir o navegador automaticamente

### Linux/Mac
1. Execute o arquivo `executar_sistema.sh`
2. O sistema irá automaticamente:
   - Verificar se Python está instalado
   - Instalar as dependências necessárias
   - Iniciar o sistema na porta 5000
   - Abrir o navegador automaticamente

## Execução Manual

Se preferir executar manualmente:

### 1. Instalar dependências
```bash
pip install streamlit pandas numpy plotly
```

### 2. Executar o sistema
```bash
streamlit run app.py --server.port 5000
```

### 3. Acessar no navegador
Abra seu navegador e acesse: `http://localhost:5000`

## Requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes do Python)
- Conexão com a internet (para instalar dependências)

## Dependências

O sistema utiliza as seguintes bibliotecas Python:
- streamlit: Interface web
- pandas: Manipulação de dados
- numpy: Computação numérica
- plotly: Visualização de gráficos

## Solução de Problemas

### Python não encontrado
- Certifique-se de que Python está instalado
- Verifique se Python está no PATH do sistema

### Erro de permissão (Linux/Mac)
```bash
chmod +x executar_sistema.sh
```

### Dependências não instaladas
Execute manualmente:
```bash
pip install -r dependencies.txt
```

## Parar o Sistema

Para parar o sistema, pressione `Ctrl+C` no terminal onde está rodando.