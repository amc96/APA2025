# Manual do Usuário - Sistema de Otimização de Acasalamento Animal

## Introdução

Bem-vindo ao Sistema de Otimização de Acasalamento Animal! Este sistema foi desenvolvido para ajudar pesquisadores e criadores a encontrar os melhores cruzamentos entre animais, minimizando a coancestralidade (parentesco) entre a prole.

### O que é Coancestralidade?

A coancestralidade mede o grau de parentesco entre dois animais. Quanto menor o coeficiente de coancestralidade, menor o parentesco e melhor para a diversidade genética da prole.

### Como o Sistema Funciona?

O sistema usa um algoritmo inteligente chamado GRASP que:
1. Analisa todos os possíveis cruzamentos
2. Encontra as melhores combinações
3. Minimiza o parentesco entre os filhotes
4. Fornece resultados detalhados para sua análise

## Instalação e Execução Automática

### Para Windows

1. **Baixe os arquivos** do sistema para seu computador
2. **Execute o arquivo único**:
   - Clique duas vezes em `executar_sistema.bat`
   - O sistema verificará e instalará automaticamente todas as dependências
   - Abrirá o navegador automaticamente

### Para Linux/macOS

1. **Baixe os arquivos** do sistema para seu computador
2. **Abra o terminal** na pasta dos arquivos
3. **Execute o comando único**:
   ```bash
   ./executar_sistema.sh
   ```
4. **O sistema fará tudo automaticamente**:
   - Verificará se Python está instalado
   - Criará ambiente virtual
   - Instalará dependências
   - Iniciará a aplicação
   - Abrirá o navegador

### Requisitos

- **Python 3.8 ou superior** (será instalado automaticamente se necessário)
- **Conexão com internet** (apenas para instalação)
- **2GB de espaço livre** em disco
- **1GB de memória RAM** (recomendado)

## Como Usar o Sistema

### Passo 1: Iniciar o Sistema

#### Windows
- Clique duas vezes em `executar_sistema.bat`
- O sistema fará tudo automaticamente
- O navegador abrirá automaticamente em `http://localhost:5000`

#### Linux/macOS
- Abra o terminal na pasta do sistema
- Execute: `./executar_sistema.sh`
- O sistema fará tudo automaticamente
- O navegador abrirá automaticamente em `http://localhost:5000`

**Não é necessário configurar nada manualmente!**

### Passo 2: Preparar seus Dados

Seu arquivo CSV deve ter **exatamente** estas colunas:
- `Animal_1`: Par de animais no formato 'pai1_pai2'
- `Animal_2`: Par de animais no formato 'pai3_pai4'
- `Coef`: Coeficiente de coancestralidade (número decimal)

**Exemplo de arquivo CSV:**
```csv
Animal_1,Animal_2,Coef
parent1_parent2,parent3_parent4,0.328125
parent1_parent2,parent5_parent6,0.0
parent3_parent4,parent5_parent6,0.25
parent7_parent8,parent9_parent10,0.125
```

**Importante**: Os animais devem estar no formato 'pai1_pai2' onde cada par representa um cruzamento.

### Passo 3: Fazer Upload dos Dados

1. **Na barra lateral esquerda**, clique em "Fazer upload do arquivo CSV"
2. **Selecione seu arquivo** CSV com os dados de acasalamento
3. **Aguarde** o sistema processar e validar os dados
4. **Verifique** se apareceu a mensagem "Dados carregados com sucesso!"

### Passo 4: Configurar Parâmetros

Na barra lateral, você pode:

#### Usar Parâmetros Aleatórios (Recomendado)
- Deixe marcado "Usar parâmetros aleatórios"
- O sistema gerará automaticamente os melhores parâmetros para cada execução

#### Configurar Parâmetros Manualmente
- Desmarque "Usar parâmetros aleatórios"
- Ajuste os controles:
  - **Máximo de Iterações**: 50-1000 (quanto mais, melhor o resultado)
  - **Alpha**: 0.1-1.0 (controla aleatoriedade vs. precisão)
  - **Iterações de Busca Local**: 3-100 (refina o resultado)
  - **Número de Cruzamentos**: 1-1000 (quantos cruzamentos selecionar)

#### Número de Execuções
- Selecione quantas vezes executar o algoritmo (1-100)
- Mais execuções = resultados mais confiáveis

### Passo 5: Executar Otimização

1. **Clique em "Executar Otimização GRASP"**
2. **Aguarde** o processamento (pode demorar alguns minutos)
3. **Acompanhe o progresso** nas barras de progresso
4. **Aguarde** ser redirecionado automaticamente para os resultados

## Interpretando os Resultados

### Página de Resultados

Após a execução, você verá:

#### 1. Resumo das Execuções
- **Tabela com todos os resultados** de cada execução
- **Melhor custo** encontrado
- **Parâmetros usados** em cada execução
- **Tempo de execução**

#### 2. Gráficos de Convergência
- **Gráfico de linhas** mostrando como o algoritmo melhorou a cada iteração
- **Comparação entre execuções** diferentes
- **Identificação da melhor solução**

#### 3. Análise da Melhor Solução
- **Estatísticas detalhadas** da melhor solução encontrada
- **Distribuição dos valores** de coancestralidade
- **Gráficos de barras** com uso de machos e fêmeas

#### 4. Melhores Cruzamentos
- **Tabela detalhada** com todos os cruzamentos recomendados
- **Ordenação por coancestralidade** (menores valores primeiro)
- **IDs originais e novos** dos animais

### Downloads Disponíveis

#### 1. Resultados Completos (CSV)
- Todas as execuções e seus resultados
- Útil para análise posterior no Excel

#### 2. Melhores Cruzamentos (CSV)
- Apenas os cruzamentos da melhor solução
- Ordenados por coancestralidade
- Pronto para implementação

#### 3. Matriz de Cruzamentos (CSV)
- Matriz completa com todos os cruzamentos possíveis
- Destaque (★) nos cruzamentos recomendados
- Visão geral de todas as opções

## Entendendo os Valores

### Coeficiente de Coancestralidade
- **0.0**: Animais não aparentados (ideal)
- **0.125**: Primos (aceitável)
- **0.25**: Meio-irmãos (cuidado)
- **0.5**: Irmãos completos (evitar)
- **1.0**: Mesmo animal (impossível)

### Custo Total
- **Menor valor** = melhor solução
- **Soma de todos** os coeficientes dos cruzamentos
- **Objetivo**: minimizar este valor

### Estatísticas da Solução
- **Média**: Coancestralidade média dos cruzamentos
- **Mínima**: Menor coancestralidade encontrada
- **Máxima**: Maior coancestralidade aceita
- **Desvio Padrão**: Variação dos valores

## Dicas para Melhores Resultados

### 1. Prepare Bem os Dados
- Verifique se todos os animais estão no arquivo
- Confirme os valores de coancestralidade
- Remova linhas com dados incompletos

### 2. Execute Múltiplas Vezes
- Use pelo menos 3 execuções
- Para dados importantes, use 5-10 execuções
- Compare os resultados entre execuções

### 3. Analise os Resultados
- Não aceite automaticamente o resultado
- Verifique se faz sentido biologicamente
- Considere fatores não incluídos no algoritmo

### 4. Parâmetros Recomendados
- **Iniciantes**: Use parâmetros aleatórios
- **Experientes**: Teste diferentes configurações
- **Dados grandes**: Use mais iterações

## Solução de Problemas

### Problemas Comuns

#### "Erro ao processar arquivo CSV"
**Causa**: Formato incorreto do arquivo
**Solução**: 
- Verifique as colunas: Animal_1, Animal_2, Coef
- Confirme se não há células vazias
- Salve como CSV (separado por vírgulas)

#### "Python não encontrado"
**Causa**: Python não está instalado
**Solução**: 
- Execute novamente o arquivo de instalação
- Baixe Python de: https://www.python.org/downloads/
- Reinicie o computador após instalação

#### "Porta já em uso"
**Causa**: Sistema já está rodando
**Solução**: 
- Feche outras instâncias do sistema
- Aguarde 30 segundos e tente novamente
- Reinicie o computador se necessário

#### "Sem memória suficiente"
**Causa**: Dados muito grandes
**Solução**: 
- Reduza o número de animais
- Feche outros programas
- Use menos execuções simultâneas

### Otimização de Performance

#### Para Dados Grandes (>1000 animais)
- Use menos iterações (20-50)
- Execute menos vezes (1-3)
- Monitore o uso de memória

#### Para Dados Pequenos (<100 animais)
- Use mais iterações (80-100)
- Execute mais vezes (5-10)
- Teste diferentes parâmetros

## Interpretação Prática

### Exemplo Real

Imagine que você tem:
- 5 fêmeas (f1, f2, f3, f4, f5)
- 3 machos (m1, m2, m3)

O sistema pode recomendar:
- f1 cruza com m2 (coancestralidade: 0.125)
- f2 cruza com m1 (coancestralidade: 0.0625)
- f3 cruza com m3 (coancestralidade: 0.1875)
- f4 cruza com m2 (coancestralidade: 0.25)
- f5 cruza com m1 (coancestralidade: 0.1875)

### Implementação Prática

1. **Revise as recomendações** com seu veterinário
2. **Considere fatores não genéticos** (idade, saúde, comportamento)
3. **Implemente gradualmente** os cruzamentos
4. **Monitore os resultados** da prole
5. **Ajuste para futuras** gerações

## Suporte e Contato

### Problemas Técnicos
- Verifique este manual primeiro
- Teste com dados menores
- Reinicie o sistema

### Dúvidas sobre Interpretação
- Consulte um geneticista animal
- Revise literatura especializada
- Considere o contexto específico

### Melhorias e Sugestões
- Documente problemas encontrados
- Sugira novas funcionalidades
- Compartilhe casos de sucesso

## Glossário

**GRASP**: Algoritmo de otimização que combina técnicas gananciosas e aleatórias para encontrar boas soluções.

**Coancestralidade**: Medida do grau de parentesco entre dois animais, expressa como probabilidade de genes idênticos.

**Matriz de Cruzamentos**: Tabela que mostra todos os possíveis cruzamentos entre fêmeas e machos.

**Otimização**: Processo de encontrar a melhor solução possível dentro das restrições dadas.

**Convergência**: Processo pelo qual o algoritmo se aproxima da melhor solução a cada iteração.

**Busca Local**: Técnica que melhora uma solução testando pequenas modificações.

**Função Objetivo**: Critério usado para avaliar a qualidade de uma solução (no nosso caso, minimizar coancestralidade).

## Limitações do Sistema

### Fatores Não Considerados
- Características produtivas dos animais
- Compatibilidade física
- Fatores ambientais
- Aspectos econômicos
- Disponibilidade temporal

### Limitações Técnicas
- Máximo de 10.000 animais no arquivo
- Apenas formato CSV suportado
- Requer conexão com internet para instalação
- Resultados dependem da qualidade dos dados

### Recomendações Finais
- Use o sistema como ferramenta de apoio
- Sempre consulte especialistas
- Considere múltiplos critérios de seleção
- Monitore resultados práticos
- Mantenha registros detalhados

---

**Desenvolvido para o Projeto Acadêmico APA 2025/1**
*Sistema de Otimização de Acasalamento Animal - GRASP*

Para suporte técnico, consulte a documentação técnica ou entre em contato com a equipe de desenvolvimento.