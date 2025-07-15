# Documentação Técnica - Sistema de Otimização de Acasalamento Animal

## Visão Geral

Este sistema implementa um algoritmo GRASP (Greedy Randomized Adaptive Search Procedure) para otimizar acasalamentos de animais, minimizando coeficientes de coancestralidade entre a prole.

## Arquitetura do Sistema

### Estrutura de Arquivos

```
├── app.py                    # Aplicação principal Streamlit
├── data_processor.py         # Processamento e validação de dados
├── grasp_algorithm.py        # Implementação do algoritmo GRASP
├── executar_sistema.bat      # Executável único para Windows
├── executar_sistema.sh       # Executável único para Linux/macOS
├── dependencies.txt          # Lista de dependências
├── DOCUMENTACAO_TECNICA.md   # Esta documentação
├── MANUAL_USUARIO.md         # Manual do usuário
├── README.md                 # Documentação principal
└── replit.md                 # Configurações do projeto
```

## Módulos e Funções

### 1. app.py - Aplicação Principal

#### Funções Principais:

**`generate_random_params()`**
- **Propósito**: Gera parâmetros aleatórios para o algoritmo GRASP com distribuição inteligente
- **Retorna**: Tupla com (max_iterations, alpha, local_search_iterations)
- **Valores Otimizados**: 
  - max_iterations: 50-1000 (distribuição não uniforme favorecendo valores médios)
  - alpha: 0.1-0.9 (distribuição normal centrada em 0.5)
  - local_search_iterations: 3-20 (otimizado para performance)

**Configuração da Interface**
- Layout wide com sidebar expandida
- Dois painéis: "Análise de Dados" e "Resultados da Otimização"
- Redirecionamento automático para resultados após execução
- Interface completamente em português brasileiro

**Funcionalidades de Upload**
- Validação robusta de arquivos CSV
- Verificação de colunas obrigatórias: Animal_1, Animal_2, Coef
- Processamento otimizado em tempo real
- Tratamento detalhado de erros com informações de debug

**Execução do Algoritmo**
- Suporte para múltiplas execuções (1-100)
- Parâmetros configuráveis: iterações (50-1000), alpha (0.1-1.0), busca local (3-100)
- Sistema de progresso sincronizado em tempo real
- Suporte a até 1000 cruzamentos por execução
- Armazenamento de resultados em session state
- Callback de progresso integrado ao algoritmo GRASP

### 2. data_processor.py - Processamento de Dados

#### Classe `DataProcessor`

**`__init__(self, uploaded_file)`**
- **Propósito**: Inicializa o processador com arquivo CSV
- **Parâmetros**: uploaded_file (objeto Streamlit)
- **Funcionalidade**: Carrega dados e inicializa variáveis

**`validate_data(self)`**
- **Propósito**: Valida estrutura e conteúdo do arquivo CSV
- **Validações**:
  - Presença das colunas: Animal_1, Animal_2, Coef
  - Tipos de dados corretos
  - Ausência de valores nulos
- **Retorna**: True se válido, False caso contrário

**`extract_animals_from_pair(self, pair_string)`**
- **Propósito**: Extrai animais individuais de string de par
- **Parâmetros**: pair_string (formato 'animal1_animal2')
- **Retorna**: Tupla (animal1, animal2)
- **Uso**: Processa dados onde pares são representados como strings

## Melhorias Implementadas (2025-07-14)

### 1. Correção Crítica de Bug
- **Problema**: Erro "too many values to unpack (expected 2)" no algoritmo GRASP
- **Causa**: Inconsistência no formato de tuplas em greedy_randomized_construction
- **Solução**: Corrigido extending da lista additional_candidates para usar formato correto (i, j)
- **Impacto**: Sistema agora processa arquivos sem erros críticos

### 2. Expansão de Limites de Cruzamentos
- **Anterior**: Limitado a 15-20 cruzamentos
- **Atual**: Suporte a 1-1000 cruzamentos
- **Benefício**: Permite processamento de datasets maiores e mais complexos

### 3. Otimização de Interface
- **Tradução Completa**: Todos os comentários e mensagens em português brasileiro
- **Tratamento de Erros**: Informações detalhadas de debug para troubleshooting
- **Sistema de Progresso**: Barras sincronizadas com execução real do algoritmo

### 4. Melhorias de Performance
- **Range de Iterações**: Expandido para 50-1000 com otimizações adaptativas
- **Convergência**: Detecção antecipada para evitar processamento desnecessário
- **Callbacks**: Sistema de progresso integrado ao núcleo do algoritmo

### 5. Robustez do Sistema
- **Validação**: Tratamento robusto de erros com informações específicas
- **Flexibilidade**: Parâmetros configuráveis com distribuição inteligente
- **Escalabilidade**: Algoritmo otimizado para datasets de tamanhos variados

**`process_data(self)`**
- **Propósito**: Processa dados brutos e cria mapeamentos
- **Funcionalidades**:
  - Identifica animais únicos
  - Separa fêmeas e machos
  - Cria mapeamentos para IDs (f1, f2, m1, m2, etc.)
  - Prepara dados para otimização

**`create_coancestry_matrix(self)`**
- **Propósito**: Cria matriz de coancestralidade completa
- **Estrutura**: Matriz simétrica com todos os cruzamentos possíveis
- **Valores**: Coeficientes do CSV + valor 1 para animais idênticos
- **Dimensões**: n_animais x n_animais

**`create_breeding_matrix(self)`**
- **Propósito**: Cria matriz de breeding otimizada
- **Estrutura**: Matriz fêmeas x machos
- **Uso**: Entrada para algoritmo GRASP
- **Dimensões**: n_fêmeas x n_machos

**`get_animal_mapping(self)`**
- **Propósito**: Obtém mapeamento de IDs originais para novos IDs
- **Retorna**: Dicionário {ID_original: ID_novo}
- **Formato**: {'animal_123': 'f1', 'animal_456': 'm1'}

**`get_breeding_pairs_dataframe(self)`**
- **Propósito**: Gera DataFrame com todos os pares possíveis
- **Colunas**: Female, Male, Female_Original, Male_Original, Coancestry
- **Uso**: Análise e visualização de dados

### 3. grasp_algorithm.py - Algoritmo GRASP

#### Classe `GRASPOptimizer`

**`__init__(self, coancestry_matrix, max_iterations, alpha, local_search_iterations)`**
- **Propósito**: Inicializa otimizador GRASP
- **Parâmetros**:
  - coancestry_matrix: Matriz de coancestralidade (numpy.ndarray)
  - max_iterations: Número máximo de iterações
  - alpha: Fator de aleatoriedade (0=puro guloso, 1=puro aleatório)
  - local_search_iterations: Iterações de busca local

**`calculate_total_cost(self, solution)`**
- **Propósito**: Calcula custo total de uma solução
- **Parâmetros**: solution (lista de índices de machos)
- **Retorna**: Custo total (soma dos coeficientes)
- **Fórmula**: Σ(coancestry_matrix[i, solution[i]]) para i=0..n_fêmeas

**`greedy_randomized_construction(self)`**
- **Propósito**: Constrói solução usando construção gulosa aleatória
- **Algoritmo**:
  1. Para cada fêmea, cria lista de candidatos
  2. Ordena por custo crescente
  3. Seleciona aleatoriamente dos melhores (parâmetro alpha)
  4. Atualiza disponibilidade de machos
- **Retorna**: Solução inicial

**`local_search(self, solution)`**
- **Propósito**: Melhora solução usando busca local
- **Estratégia**: Troca de pares (2-opt)
- **Algoritmo**:
  1. Tenta trocar assignações entre duas fêmeas
  2. Aceita troca se melhora o custo total
  3. Repete por número definido de iterações
- **Retorna**: Solução melhorada

**`optimize(self, progress_callback=None)`**
- **Propósito**: Executa algoritmo GRASP completo
- **Fases**:
  1. Construção gulosa aleatória
  2. Busca local
  3. Atualização da melhor solução
- **Parâmetros**: progress_callback (função opcional para progresso)
- **Retorna**: Tupla (melhor_solução, melhor_custo, custos_por_iteração)

**`get_solution_statistics(self, solution)`**
- **Propósito**: Calcula estatísticas de uma solução
- **Métricas**:
  - Custo total
  - Custo médio por par
  - Custo mínimo e máximo
  - Desvio padrão
  - Distribuição de custos
- **Retorna**: Dicionário com estatísticas

## Algoritmo GRASP - Detalhes Técnicos

### Fase de Construção Gulosa Aleatória

1. **Criação de Lista de Candidatos**:
   ```python
   candidates = [(cost, male_idx) for male_idx in available_males]
   candidates.sort()  # Ordenar por custo
   ```

2. **Lista Restrita de Candidatos (RCL)**:
   ```python
   rcl_size = max(1, int(alpha * len(candidates)))
   rcl = candidates[:rcl_size]
   ```

3. **Seleção Aleatória**:
   ```python
   selected = random.choice(rcl)
   ```

### Fase de Busca Local

1. **Geração de Vizinhança**:
   - Troca de assignações entre duas fêmeas
   - Verificação de melhoria no custo total

2. **Critério de Aceitação**:
   ```python
   if new_cost < current_cost:
       current_solution = new_solution
       current_cost = new_cost
   ```

### Critérios de Convergência

- Número máximo de iterações atingido
- Número máximo de iterações sem melhoria
- Custo alvo atingido (opcional)

## Estruturas de Dados

### Matriz de Coancestralidade Completa
```python
# Dimensões: n_animais x n_animais
# Valores: coeficientes do CSV + 1.0 para diagonal principal
coancestry_matrix[i][j] = coef_ij if i != j else 1.0
```

### Matriz de Breeding
```python
# Dimensões: n_fêmeas x n_machos
# Valores: apenas coeficientes entre fêmeas e machos
breeding_matrix[i][j] = coef_femea_i_macho_j
```

### Solução
```python
# Lista de índices de machos para cada fêmea
# solution[i] = j significa que fêmea i cruza com macho j
solution = [2, 0, 1, 3, ...]  # Exemplo
```

## Fluxo de Execução

1. **Carregamento de Dados**:
   - Upload de arquivo CSV
   - Validação de formato
   - Processamento de dados

2. **Configuração de Parâmetros**:
   - Parâmetros manuais ou aleatórios
   - Número de execuções

3. **Execução do Algoritmo**:
   - Loop de múltiplas execuções
   - Cada execução: construção + busca local
   - Armazenamento de resultados

4. **Apresentação de Resultados**:
   - Redirecionamento para página de resultados
   - Visualizações e estatísticas
   - Downloads de arquivos

## Otimizações e Considerações

### Performance
- Uso de NumPy para operações matriciais
- Caching de cálculos de custo
- Estruturas de dados eficientes

### Escalabilidade
- Algoritmo O(n²) para construção
- Busca local O(n²) por iteração
- Memória O(n²) para matrizes

### Robustez
- Validação extensiva de dados
- Tratamento de erros
- Logs de execução

## Dependências

### Bibliotecas Python
- **streamlit**: Interface web
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **plotly**: Visualizações interativas
- **io**: Manipulação de arquivos em memória
- **time**: Medição de tempo e delays
- **random**: Geração de números aleatórios

### Requisitos de Sistema
- Python 3.8+
- Memória: Mínimo 512MB (recomendado 1GB+)
- Processador: Qualquer arquitetura moderna
- Sistema Operacional: Windows, Linux, macOS

## Configuração e Deployment

### Desenvolvimento Local
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 5000

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.port 5000
```

### Configuração Streamlit
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
maxUploadSize = 200

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## Testes e Validação

### Testes de Unidade
- Validação de dados
- Cálculos matemáticos
- Geração de matrizes

### Testes de Integração
- Fluxo completo de execução
- Interface web
- Geração de arquivos

### Testes de Performance
- Datasets de diferentes tamanhos
- Medição de tempo de execução
- Uso de memória

## Troubleshooting

### Problemas Comuns

1. **Erro de Importação**:
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solução**: Executar script de instalação

2. **Erro de Formato CSV**:
   ```
   Arquivo CSV deve conter colunas: Animal_1, Animal_2, Coef
   ```
   **Solução**: Verificar formato do arquivo

3. **Erro de Memória**:
   ```
   MemoryError: Unable to allocate array
   ```
   **Solução**: Reduzir tamanho do dataset ou aumentar memória

4. **Erro de Porta**:
   ```
   OSError: [Errno 98] Address already in use
   ```
   **Solução**: Mudar porta ou finalizar processo existente

### Logs de Debug
- Ativação via variável de ambiente: `STREAMLIT_LOGGER_LEVEL=debug`
- Logs disponíveis no console
- Monitoramento de performance via Streamlit

## Extensibilidade

### Novos Algoritmos
- Interface `BaseOptimizer` para implementar novos algoritmos
- Métodos padronizados para compatibilidade
- Testes automatizados

### Novos Formatos de Dados
- Extensão da classe `DataProcessor`
- Suporte para Excel, JSON, etc.
- Validadores personalizados

### Novas Visualizações
- Componentes Plotly customizados
- Dashboards interativos
- Exportação de relatórios

## Manutenção

### Atualizações Regulares
- Dependências Python
- Compatibilidade Streamlit
- Correções de bugs

### Monitoramento
- Logs de erro
- Performance metrics
- Feedback dos usuários

### Backup
- Versionamento de código
- Backup de configurações
- Documentação atualizada