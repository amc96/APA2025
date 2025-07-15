import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import random
from data_processor import DataProcessor
from grasp_algorithm import GRASPOptimizer

# Page configuration
st.set_page_config(
    page_title="Otimização de Acasalamento Animal - GRASP",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'multiple_results' not in st.session_state:
    st.session_state.multiple_results = []
if 'redirect_to_results' not in st.session_state:
    st.session_state.redirect_to_results = False
if 'force_data_page' not in st.session_state:
    st.session_state.force_data_page = False

def generate_random_params():
    """Gera parâmetros aleatórios para o GRASP com diversidade para evitar sequências"""
    # Usar distribuição não uniforme para maior diversidade no intervalo 50-1000
    random_max_iterations = random.choices(
        [50, 100, 200, 300, 500, 750, 1000], 
        weights=[1, 3, 5, 4, 3, 2, 1]  # Favorece valores médios
    )[0]
    
    # Alpha com distribuição que favorece valores mais equilibrados
    alpha_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    random_alpha = random.choices(
        alpha_options,
        weights=[1, 2, 3, 4, 5, 4, 3, 2, 1]  # Distribuição normal
    )[0]
    
    # Local search com variação para evitar padrões
    random_local_search = random.choices(
        [3, 5, 8, 10, 15, 20], 
        weights=[1, 3, 4, 5, 3, 1]  # Favorece valores médios
    )[0]
    
    return random_max_iterations, random_alpha, random_local_search

# Sidebar - Configuração de páginas
st.sidebar.header("📑 Navegação")
pages = ["Análise de Dados", "Resultados da Otimização"]

# Botão para limpar todos os dados e recomeçar
st.sidebar.markdown("---")
st.sidebar.header("🔄 Controles")

if st.sidebar.button("🆕 Começar Novo Projeto", type="primary", help="Limpa todos os dados e volta ao início"):
    # Limpar todos os dados da sessão
    st.session_state.data_processor = None
    st.session_state.optimization_results = None
    st.session_state.multiple_results = []
    st.session_state.redirect_to_results = False
    st.session_state.force_data_page = False
    st.success("✅ Todos os dados foram limpos! Você pode começar um novo projeto.")
    st.rerun()

# Mostrar status atual do projeto
if st.session_state.data_processor is not None:
    st.sidebar.success("✅ Dados carregados")
else:
    st.sidebar.info("📁 Aguardando dados")

if st.session_state.multiple_results:
    st.sidebar.success(f"🎯 {len(st.session_state.multiple_results)} otimizações concluídas")
else:
    st.sidebar.info("⚙️ Nenhuma otimização executada")

# Verificar se deve forçar ir para página de dados
if st.session_state.force_data_page:
    default_page = "Análise de Dados"
    st.session_state.force_data_page = False  # Reset flag
# Verificar se deve redirecionar automaticamente para resultados
elif st.session_state.redirect_to_results:
    default_page = "Resultados da Otimização"
    st.session_state.redirect_to_results = False  # Reset flag
else:
    # Se há resultados de otimização, manter na página de resultados por padrão
    if st.session_state.multiple_results:
        default_page = "Resultados da Otimização"
    else:
        default_page = pages[0]

# Encontrar o índice da página padrão
try:
    default_index = pages.index(default_page)
except ValueError:
    default_index = 0

selected_page = st.sidebar.selectbox("Selecione a Página", pages, index=default_index)

# Sidebar - Upload de arquivo
st.sidebar.header("📁 Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Fazer upload do arquivo CSV com dados de acasalamento", 
    type=['csv'],
    help="O arquivo CSV deve conter as colunas: Animal_1, Animal_2, Coef"
)

# Sidebar - Parâmetros GRASP
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parâmetros GRASP")

# Opção para parâmetros aleatórios
use_random_params = st.sidebar.checkbox("Usar parâmetros aleatórios", value=True)

if use_random_params:
    st.sidebar.info("Parâmetros serão gerados aleatoriamente para cada execução")
    max_iterations = None
    alpha = None
    local_search_iterations = None
else:
    max_iterations = st.sidebar.slider("Máximo de Iterações", 50, 1000, 200)  # Valores otimizados
    alpha = st.sidebar.slider("Alpha (Fator Ganancioso)", 0.1, 1.0, 0.3, 0.1)
    local_search_iterations = st.sidebar.slider("Iterações de Busca Local", 1, 30, 10)  # Valores otimizados

# Número de execuções
num_executions = st.sidebar.slider("Número de Execuções", 1, 100, 3)

# Configuração do número de cruzamentos
st.sidebar.markdown("---")
st.sidebar.header("🎯 Configuração de Cruzamentos")

# Opção para número automático ou manual
auto_crossings = st.sidebar.checkbox("Número automático de cruzamentos", value=True, help="MODIFICADO: Usa no mínimo 80% do total de pares disponíveis para otimização")

if not auto_crossings:
    manual_crossings = st.sidebar.slider("Número de cruzamentos a selecionar", 1, 1000, 15, help="Quantos cruzamentos o algoritmo deve selecionar")
else:
    manual_crossings = None

# Mostrar limite máximo sugerido se houver dados carregados
if st.session_state.data_processor is not None:
    total_pairs = len(st.session_state.data_processor.all_pairs)
    total_possible_pairs = (total_pairs * (total_pairs - 1)) // 2
    min_crossings_80_percent = int(total_possible_pairs * 0.8)
    
    st.sidebar.info(f"Total de pares disponíveis: {total_pairs}")
    st.sidebar.info(f"Total de combinações possíveis: {total_possible_pairs}")
    st.sidebar.info(f"🎯 Mínimo 80% da matriz: {min_crossings_80_percent} cruzamentos")
    st.sidebar.info(f"Máximo possível: {total_possible_pairs} cruzamentos")
else:
    st.sidebar.info("Carregue dados para ver sugestões")

# Title and description
st.title("🐄 Sistema de Otimização de Acasalamento Animal")
st.markdown("### Usando Meta-heurística GRASP para minimizar coeficientes de coancestralidade")
st.markdown("---")

# =============================================================================
# PÁGINA DE ANÁLISE DE DADOS
# =============================================================================
if selected_page == "Análise de Dados":
    if uploaded_file is not None:
        try:
            # Load and process data
            with st.spinner("Carregando e processando dados..."):
                st.session_state.data_processor = DataProcessor(uploaded_file)
                
            dp = st.session_state.data_processor
            
            # Display data statistics
            st.header("📊 Análise dos Dados")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Registros", len(dp.df))
            
            with col2:
                st.metric("Fêmeas Distintas", dp.num_females)
            
            with col3:
                st.metric("Machos Distintos", dp.num_males)
            
            with col4:
                st.metric("Total de Animais", dp.num_females + dp.num_males)
            
            # Coefficient statistics
            st.subheader("🔢 Estatísticas dos Coeficientes")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Coeficiente Médio", f"{dp.coef_mean:.6f}")
            
            with col2:
                st.metric("Coeficiente Máximo", f"{dp.coef_max:.6f}")
            
            with col3:
                st.metric("Coeficiente Mínimo", f"{dp.coef_min:.6f}")
            
            # Coefficient distribution plot
            st.subheader("📈 Distribuição dos Coeficientes")
            fig_hist = px.histogram(
                dp.df, 
                x='Coef', 
                nbins=50,
                title="Distribuição dos Coeficientes de Coancestralidade",
                labels={'Coef': 'Coeficiente de Coancestralidade', 'count': 'Frequência'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Animal mapping
            st.header("🏷️ Identificação dos Animais")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fêmeas")
                female_df = pd.DataFrame({
                    'ID Original': dp.females,
                    'Novo ID': [f'f{i+1}' for i in range(len(dp.females))]
                })
                st.dataframe(female_df, use_container_width=True, height=300)
            
            with col2:
                st.subheader("Machos")
                male_df = pd.DataFrame({
                    'ID Original': dp.males,
                    'Novo ID': [f'm{i+1}' for i in range(len(dp.males))]
                })
                st.dataframe(male_df, use_container_width=True, height=300)
            
            # Breeding matrix - Matriz completa de todos os cruzamentos
            st.header("🔢 Matriz de Cruzamentos")
            st.subheader("Matriz de Coancestralidade (Todos os Cruzamentos do Arquivo CSV)")
            
            # Create matrix visualization
            matrix_fig = px.imshow(
                dp.coancestry_matrix,
                labels=dict(x="Pares de Animais", y="Pares de Animais", color="Coancestralidade"),
                x=[f'P{i+1}' for i in range(len(dp.all_pairs))],
                y=[f'P{i+1}' for i in range(len(dp.all_pairs))],
                color_continuous_scale="Viridis"
            )
            matrix_fig.update_layout(
                title="Mapa de Calor da Matriz de Coancestralidade (Todos os Cruzamentos)",
                xaxis_title="Pares de Animais",
                yaxis_title="Pares de Animais"
            )
            st.plotly_chart(matrix_fig, use_container_width=True)
            
            # Show matrix as table (first 20x20 for readability)
            st.subheader("Valores da Matriz (Primeiros 20×20)")
            display_size = min(20, len(dp.all_pairs))
            matrix_display = pd.DataFrame(
                dp.coancestry_matrix[:display_size, :display_size],
                index=[f'P{i+1}' for i in range(display_size)],
                columns=[f'P{i+1}' for i in range(display_size)]
            )
            st.dataframe(matrix_display, use_container_width=True)
            
            # Mostrar correspondência entre P1, P2, etc. e os pares reais
            st.subheader("Correspondência dos Pares")
            pairs_df = pd.DataFrame({
                'ID': [f'P{i+1}' for i in range(len(dp.all_pairs))],
                'Par de Animais': dp.all_pairs
            })
            st.dataframe(pairs_df, use_container_width=True, height=300)
            
            # Matriz de Breeding (Fêmeas x Machos) para otimização
            st.header("🔍 Matriz de Breeding (Para Otimização)")
            st.subheader("Matriz Fêmeas × Machos")
            
            # Create breeding matrix visualization
            breeding_fig = px.imshow(
                dp.breeding_matrix,
                labels=dict(x="Machos", y="Fêmeas", color="Coancestralidade"),
                x=[f'm{i+1}' for i in range(len(dp.males))],
                y=[f'f{i+1}' for i in range(len(dp.females))],
                color_continuous_scale="Viridis"
            )
            breeding_fig.update_layout(
                title="Matriz de Breeding (Fêmeas × Machos)",
                xaxis_title="Machos",
                yaxis_title="Fêmeas"
            )
            st.plotly_chart(breeding_fig, use_container_width=True)
            
            # GRASP Optimization
            st.header("🔍 Otimização GRASP")
            
            # Mostrar resultados existentes se houver
            if st.session_state.multiple_results:
                st.info(f"🎯 Você já possui {len(st.session_state.multiple_results)} execuções salvas. Use os botões abaixo para executar mais ou limpar os resultados.")
                
                col_exec1, col_exec2 = st.columns(2)
                
                with col_exec1:
                    execute_new = st.button("▶️ Executar Mais Otimizações", type="primary", help="Adiciona novas execuções aos resultados existentes")
                
                with col_exec2:
                    if st.button("🗑️ Limpar e Recomeçar", help="Remove resultados anteriores e executa novas otimizações"):
                        st.session_state.multiple_results = []
                        st.session_state.optimization_results = None
                        execute_new = True
                        st.success("✅ Resultados anteriores removidos!")
                        st.rerun()
                    else:
                        execute_new = False
            else:
                execute_new = st.button("▶️ Executar Algoritmo GRASP", type="primary")
            
            if execute_new:
                try:
                    # Só limpar se não existirem resultados anteriores ou se foi solicitado explicitamente
                    if not st.session_state.multiple_results:
                        st.session_state.multiple_results = []
                    
                    # Progress tracking
                    overall_progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Calcular o número base de execuções já existentes
                    existing_executions = len(st.session_state.multiple_results)
                    
                    # Inicializar progresso detalhado
                    progress_container = st.container()
                    with progress_container:
                        progress_detail = st.empty()
                    
                    # Verificar se há dados válidos antes de executar
                    if not hasattr(dp, 'all_pairs') or len(dp.all_pairs) == 0:
                        st.error("❌ Erro: Não há dados válidos para processar")
                        st.stop()
                    
                    st.success(f"✅ Processando {len(dp.all_pairs)} pares de animais")
                    
                    for execution in range(num_executions):
                        # Generate parameters for this execution
                        if use_random_params:
                            curr_max_iterations, curr_alpha, curr_local_search = generate_random_params()
                        else:
                            curr_max_iterations = max_iterations
                            curr_alpha = alpha
                            curr_local_search = local_search_iterations
                        
                        current_exec_num = existing_executions + execution + 1
                        total_executions = existing_executions + num_executions
                        
                        # MODIFICADO: Calcular número de cruzamentos com base em 80% dos pares disponíveis no CSV
                        available_pairs = len(dp.all_pairs)  # Pares disponíveis no arquivo CSV
                        min_crossings_80_percent = int(available_pairs * 0.8)
                        
                        if auto_crossings:
                            preview_crossings = max(min_crossings_80_percent, min(available_pairs, len(dp.all_pairs)))
                        else:
                            preview_crossings = max(min_crossings_80_percent, min(manual_crossings, available_pairs))
                        
                        status_text.text(f"Executando otimização {current_exec_num}/{total_executions} - Selecionando {preview_crossings} cruzamentos")
                        
                        # Progresso detalhado dentro de cada execução
                        exec_progress = st.progress(0)
                        
                        with st.spinner(f"Executando otimização {current_exec_num}/{total_executions} com {preview_crossings} cruzamentos..."):
                            try:
                                # Usar a matriz de coancestralidade completa (todos os cruzamentos)
                                optimizer = GRASPOptimizer(
                                    dp.coancestry_matrix,
                                    max_iterations=curr_max_iterations,
                                    alpha=curr_alpha,
                                    local_search_iterations=curr_local_search,
                                    pair_names=dp.all_pairs
                                )
                                
                                # MODIFICADO: Definir número de cruzamentos para usar no mínimo 80% dos pares disponíveis
                                available_pairs = len(dp.all_pairs)  # Pares disponíveis no arquivo CSV
                                min_crossings_80_percent = int(available_pairs * 0.8)
                                
                                if auto_crossings:
                                    # Modo automático: usa no mínimo 80% dos pares disponíveis
                                    num_crossings = max(min_crossings_80_percent, min(available_pairs, len(dp.all_pairs)))
                                else:
                                    # Modo manual: usa valor definido pelo usuário, mas garante no mínimo 80%
                                    num_crossings = max(min_crossings_80_percent, min(manual_crossings, available_pairs))
                                
                                st.info(f"🔄 Processando {num_crossings} cruzamentos de {available_pairs} pares disponíveis ({num_crossings/available_pairs*100:.1f}%)")
                                
                                start_time = time.time()
                                
                                # Criar callback de progresso
                                def progress_callback(current_iteration, total_iterations):
                                    progress = current_iteration / total_iterations
                                    exec_progress.progress(progress)
                                    return True  # Continue execution
                                
                                best_solution, best_cost, iteration_costs = optimizer.optimize(
                                    num_crossings=num_crossings,
                                    progress_callback=progress_callback
                                )
                                end_time = time.time()
                                
                                # Verificar se obtivemos uma solução válida
                                if best_solution is None or len(best_solution) == 0:
                                    st.error(f"❌ Erro: Nenhuma solução válida encontrada na execução {current_exec_num}")
                                    continue
                                
                                st.success(f"✅ Execução {current_exec_num} concluída com sucesso!")
                                
                            except Exception as e:
                                st.error(f"❌ Erro na execução {current_exec_num}: {str(e)}")
                                continue
                        
                            # Store results
                            result = {
                                'execution': current_exec_num,
                                'best_solution': best_solution,
                                'best_cost': best_cost,
                                'iteration_costs': iteration_costs,
                                'execution_time': end_time - start_time,
                                'max_iterations': curr_max_iterations,
                                'alpha': curr_alpha,
                                'local_search_iterations': curr_local_search,
                                'final_iterations': len(iteration_costs),
                                'best_crossings': optimizer.best_crossings,
                                'num_crossings_selected': len(best_solution),
                                'optimizer': optimizer  # Guardar para análises posteriores
                            }
                            
                            st.session_state.multiple_results.append(result)
                        
                        # Update progress
                        exec_progress.progress(1.0)  # Completa a barra desta execução
                        overall_progress.progress((execution + 1) / num_executions)
                
                    status_text.text("Todas as otimizações concluídas!")
                    st.success(f"Concluídas {num_executions} execuções do algoritmo GRASP!")
                    
                    # Store the best overall result
                    best_execution = min(st.session_state.multiple_results, key=lambda x: x['best_cost'])
                    st.session_state.optimization_results = best_execution
                    
                    # Redirecionar para a página de resultados
                    st.info("🎯 Redirecionando para a página de resultados...")
                    time.sleep(1)  # Pequena pausa para mostrar a mensagem
                    st.session_state.redirect_to_results = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erro geral durante a execução: {str(e)}")
                    with st.expander("🔍 Detalhes do Erro"):
                        st.code(str(e))
                        import traceback
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
            st.error("Certifique-se de que seu arquivo CSV possui as colunas requeridas: Animal_1, Animal_2, Coef")
            
            # Mostrar detalhes do erro para debug
            with st.expander("🔍 Detalhes do Erro (para debug)"):
                st.code(f"Tipo do erro: {type(e).__name__}")
                st.code(f"Mensagem: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:
        st.info("👆 Por favor, faça upload de um arquivo CSV para começar a análise")
        
        # Show example format
        st.subheader("📋 Formato de Arquivo Esperado")
        st.code("""
Animal_1,Animal_2,Coef
parent1_parent2,parent3_parent4,0.328125
parent1_parent2,parent5_parent6,0.0
parent3_parent4,parent5_parent6,0.25
...
        """)
        
        st.markdown("""
        **Requisitos:**
        - Arquivo CSV com três colunas: `Animal_1`, `Animal_2`, `Coef`
        - `Animal_1` e `Animal_2` representam pares de acasalamento no formato `parent1_parent2`
        - `Coef` representa o coeficiente de coancestralidade entre os pares
        - O sistema identificará automaticamente e categorizará os animais como fêmeas e machos
        """)

# =============================================================================
# PÁGINA DE RESULTADOS DA OTIMIZAÇÃO
# =============================================================================
elif selected_page == "Resultados da Otimização":
    if st.session_state.data_processor is None:
        st.warning("⚠️ Nenhum dado foi carregado. Por favor, vá para a página 'Análise de Dados' e faça upload de um arquivo CSV.")
    elif not st.session_state.multiple_results:
        st.info("ℹ️ Nenhuma otimização foi executada ainda. Por favor, execute o algoritmo GRASP na página 'Análise de Dados'.")
    else:
        dp = st.session_state.data_processor
        results = st.session_state.multiple_results
        
        st.header("🎯 Resultados da Otimização GRASP")
        
        # Botão para executar nova otimização na página de resultados
        col_button1, col_button2, col_button3 = st.columns([1, 1, 2])
        
        with col_button1:
            if st.button("🔄 Nova Otimização", help="Executa uma nova otimização com os mesmos dados"):
                # Resetar flag de redirecionamento para forçar ida para análise de dados
                st.session_state.redirect_to_results = False
                st.session_state.force_data_page = True
                st.rerun()
        
        with col_button2:
            if st.button("📊 Voltar aos Dados", help="Voltar para página de análise dos dados"):
                # Definir página padrão como análise de dados
                st.session_state.redirect_to_results = False
                st.session_state.force_data_page = True
                st.rerun()
        
        with col_button3:
            st.info(f"Mostrando resultados de {len(results)} execuções")
        
        st.markdown("---")
        
        # Summary statistics
        st.subheader("📊 Resumo das Execuções")
        
        costs = [r['best_cost'] for r in results]
        times = [r['execution_time'] for r in results]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Número de Execuções", len(results))
        
        with col2:
            st.metric("Melhor Custo", f"{min(costs):.6f}")
        
        with col3:
            st.metric("Custo Médio", f"{np.mean(costs):.6f}")
        
        with col4:
            st.metric("Tempo Médio", f"{np.mean(times):.2f}s")
        
        # Comparison chart of all executions
        st.subheader("📈 Comparação das Execuções")
        
        # Create comparison data
        comparison_data = []
        for i, result in enumerate(results):
            comparison_data.append({
                'Execução': f"Execução {result['execution']}",
                'Custo Total': result['best_cost'],
                'Tempo (s)': result['execution_time'],
                'Iterações': result['max_iterations'],
                'Alpha': result['alpha'],
                'Busca Local': result['local_search_iterations']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Bar chart of costs
        fig_costs = px.bar(
            comparison_df,
            x='Execução',
            y='Custo Total',
            title="Custo Total de Coancestralidade por Execução",
            labels={'Custo Total': 'Custo Total de Coancestralidade'}
        )
        fig_costs.update_layout(showlegend=False)
        st.plotly_chart(fig_costs, use_container_width=True)
        
        # Parameters comparison
        st.subheader("⚙️ Parâmetros Utilizados")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Convergence comparison
        st.subheader("🔄 Convergência dos Algoritmos")
        
        fig_convergence = go.Figure()
        
        for result in results:
            fig_convergence.add_trace(go.Scatter(
                x=list(range(len(result['iteration_costs']))),
                y=result['iteration_costs'],
                mode='lines',
                name=f"Execução {result['execution']}",
                line=dict(width=2)
            ))
        
        fig_convergence.update_layout(
            title="Convergência de Todas as Execuções",
            xaxis_title="Iteração",
            yaxis_title="Melhor Custo Encontrado",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_convergence, use_container_width=True)
        
        # Best solution details
        best_result = min(results, key=lambda x: x['best_cost'])
        st.subheader("🏆 Melhor Solução Encontrada")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Execução", f"Execução {best_result['execution']}")
        
        with col2:
            st.metric("Custo Total", f"{best_result['best_cost']:.6f}")
        
        with col3:
            st.metric("Tempo de Execução", f"{best_result['execution_time']:.2f}s")
        
        # Best crossings from optimization
        st.subheader("🎯 Melhores Cruzamentos Encontrados")
        
        if 'best_crossings' in best_result and best_result['best_crossings']:
            best_crossings_df = pd.DataFrame(best_result['best_crossings'])
            best_crossings_df = best_crossings_df.rename(columns={
                'pair1_name': 'Par Animal 1',
                'pair2_name': 'Par Animal 2', 
                'coancestry': 'Coeficiente de Coancestralidade'
            })
            
            # Destacar os melhores (menores coeficientes)
            st.dataframe(
                best_crossings_df[['Par Animal 1', 'Par Animal 2', 'Coeficiente de Coancestralidade']],
                use_container_width=True
            )
            
            # Estatísticas dos melhores cruzamentos
            st.subheader("📊 Estatísticas dos Melhores Cruzamentos")
            
            col1, col2, col3, col4 = st.columns(4)
            
            coef_values = [c['coancestry'] for c in best_result['best_crossings']]
            
            with col1:
                st.metric("Média", f"{np.mean(coef_values):.6f}")
            
            with col2:
                st.metric("Mínimo", f"{min(coef_values):.6f}")
            
            with col3:
                st.metric("Máximo", f"{max(coef_values):.6f}")
            
            with col4:
                st.metric("Desvio Padrão", f"{np.std(coef_values):.6f}")
        
        else:
            # Fallback para compatibilidade com versão anterior
            st.warning("⚠️ Dados de cruzamentos detalhados não disponíveis. Execute a otimização novamente.")
            
            breeding_pairs = []
            for i, male_idx in enumerate(best_result['best_solution']):
                if isinstance(male_idx, tuple):
                    # Nova versão com pares de cruzamentos
                    pair1_idx, pair2_idx = male_idx
                    breeding_pairs.append({
                        'Par 1': dp.all_pairs[pair1_idx],
                        'Par 2': dp.all_pairs[pair2_idx],
                        'Coancestralidade': dp.coancestry_matrix[pair1_idx, pair2_idx]
                    })
                else:
                    # Versão antiga com machos/fêmeas
                    female_id = f'f{i+1}'
                    male_id = f'm{male_idx+1}'
                    coancestry = dp.breeding_matrix[i, male_idx] if hasattr(dp, 'breeding_matrix') else 0
                    breeding_pairs.append({
                        'Fêmea': female_id,
                        'Macho': male_id,
                        'ID Original Fêmea': dp.females[i] if hasattr(dp, 'females') else f'f{i+1}',
                        'ID Original Macho': dp.males[male_idx] if hasattr(dp, 'males') else f'm{male_idx+1}',
                        'Coancestralidade': coancestry
                    })
            
            breeding_df = pd.DataFrame(breeding_pairs)
            st.dataframe(breeding_df, use_container_width=True)
        
        # Matrix visualization with best crossings
        st.subheader("🗃️ Matriz de Cruzamentos com Destaques")
        
        if 'optimizer' in best_result and hasattr(best_result['optimizer'], 'get_best_crossings_matrix'):
            # Obter matriz com melhores cruzamentos destacados
            best_crossings_matrix = best_result['optimizer'].get_best_crossings_matrix()
            
            # Criar visualização da matriz original com destaques
            fig_matrix = px.imshow(
                dp.coancestry_matrix,
                labels=dict(x="Pares de Animais", y="Pares de Animais", color="Coancestralidade"),
                x=[f'P{i+1}' for i in range(len(dp.all_pairs))],
                y=[f'P{i+1}' for i in range(len(dp.all_pairs))],
                color_continuous_scale="Viridis"
            )
            
            # Adicionar marcadores para os melhores cruzamentos
            best_x, best_y = np.where(best_crossings_matrix == 1)
            
            fig_matrix.add_scatter(
                x=best_x,
                y=best_y,
                mode='markers',
                marker=dict(symbol='star', size=10, color='red'),
                name='Melhores Cruzamentos',
                showlegend=True
            )
            
            fig_matrix.update_layout(
                title="Matriz de Coancestralidade com Melhores Cruzamentos Destacados (★)",
                xaxis_title="Pares de Animais",
                yaxis_title="Pares de Animais"
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Solution analysis
        st.subheader("📈 Análise da Solução")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'best_crossings' in best_result and best_result['best_crossings']:
                st.write("**Distribuição de Coancestralidade dos Melhores Cruzamentos:**")
                coef_values = [c['coancestry'] for c in best_result['best_crossings']]
                
                coancestry_fig = px.histogram(
                    x=coef_values,
                    nbins=15,
                    title="Distribuição da Coancestralidade nos Melhores Cruzamentos",
                    labels={'x': 'Coeficiente de Coancestralidade', 'y': 'Frequência'}
                )
                st.plotly_chart(coancestry_fig, use_container_width=True)
            else:
                st.write("**Distribuição de Uso dos Machos:**")
                if all(isinstance(x, int) for x in best_result['best_solution']):
                    male_usage = pd.Series(best_result['best_solution']).value_counts().sort_index()
                    male_usage.index = [f'm{i+1}' for i in male_usage.index]
                    usage_fig = px.bar(
                        x=male_usage.index,
                        y=male_usage.values,
                        title="Número de Fêmeas por Macho",
                        labels={'x': 'Macho', 'y': 'Número de Fêmeas'}
                    )
                    st.plotly_chart(usage_fig, use_container_width=True)
                else:
                    st.info("Análise de uso de machos não disponível para este tipo de solução.")
        
        with col2:
            st.write("**Qualidade da Solução vs. Todas as Possibilidades:**")
            
            # Comparar com todos os cruzamentos possíveis
            if 'optimizer' in best_result and hasattr(best_result['optimizer'], 'get_all_crossings_ranked'):
                all_crossings = best_result['optimizer'].get_all_crossings_ranked()
                all_coef = [c['coancestry'] for c in all_crossings]
                selected_coef = [c['coancestry'] for c in all_crossings if c['selected']]
                
                comparison_data = {
                    'Todos os Cruzamentos': all_coef,
                    'Cruzamentos Selecionados': selected_coef
                }
                
                fig_comparison = px.box(
                    [all_coef, selected_coef],
                    title="Comparação: Todos vs. Selecionados"
                )
                fig_comparison.update_layout(
                    xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Todos', 'Selecionados'])
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.info("Comparação detalhada não disponível.")
        
        # Download results
        st.subheader("💾 Download dos Resultados")
        
        # Create comprehensive results CSV
        all_results_data = []
        for result in results:
            base_data = {
                'Execução': result['execution'],
                'Custo_Total': result['best_cost'],
                'Tempo_Execucao': result['execution_time'],
                'Max_Iteracoes': result['max_iterations'],
                'Alpha': result['alpha'],
                'Busca_Local': result['local_search_iterations'],
                'Iteracoes_Realizadas': result['final_iterations']
            }
            
            # Add crossings data for this execution
            if 'best_crossings' in result and result['best_crossings']:
                # Novos dados de cruzamentos
                for idx, crossing in enumerate(result['best_crossings']):
                    row_data = base_data.copy()
                    row_data.update({
                        'Cruzamento_ID': idx + 1,
                        'Par_Animal_1': crossing['pair1_name'],
                        'Par_Animal_2': crossing['pair2_name'],
                        'Coancestralidade': crossing['coancestry']
                    })
                    all_results_data.append(row_data)
            else:
                # Compatibilidade com versão anterior (fêmeas/machos)
                for i, solution_item in enumerate(result['best_solution']):
                    row_data = base_data.copy()
                    if isinstance(solution_item, tuple):
                        # Nova versão com pares
                        pair1_idx, pair2_idx = solution_item
                        row_data.update({
                            'Cruzamento_ID': i + 1,
                            'Par_Animal_1': dp.all_pairs[pair1_idx],
                            'Par_Animal_2': dp.all_pairs[pair2_idx],
                            'Coancestralidade': dp.coancestry_matrix[pair1_idx, pair2_idx]
                        })
                    else:
                        # Versão antiga com machos/fêmeas
                        male_idx = solution_item
                        row_data.update({
                            'Femea_ID': f'f{i+1}',
                            'Macho_ID': f'm{male_idx+1}',
                            'Femea_Original': dp.females[i] if hasattr(dp, 'females') else f'f{i+1}',
                            'Macho_Original': dp.males[male_idx] if hasattr(dp, 'males') else f'm{male_idx+1}',
                            'Coancestralidade': dp.breeding_matrix[i, male_idx] if hasattr(dp, 'breeding_matrix') else 0
                        })
                    all_results_data.append(row_data)
        
        results_df = pd.DataFrame(all_results_data)
        
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Resultados Completos (CSV)",
            data=csv_data,
            file_name=f"resultados_grasp_{int(time.time())}.csv",
            mime="text/csv"
        )
        
        # Gerar arquivos especiais para a melhor solução
        st.subheader("🎯 Arquivos da Melhor Solução")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV com os melhores cruzamentos
            if 'best_crossings' in best_result and best_result['best_crossings']:
                # Novos dados de cruzamentos
                best_crossings_df = pd.DataFrame(best_result['best_crossings'])
                best_crossings_df = best_crossings_df.rename(columns={
                    'pair1_name': 'Par_Animal_1',
                    'pair2_name': 'Par_Animal_2',
                    'coancestry': 'Coeficiente_Coancestralidade'
                })
                best_crossings_df = best_crossings_df[['Par_Animal_1', 'Par_Animal_2', 'Coeficiente_Coancestralidade']]
                
                # Já está ordenado por coancestralidade (menor para maior)
                csv_best_buffer = io.StringIO()
                best_crossings_df.to_csv(csv_best_buffer, index=False)
                csv_best_data = csv_best_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download Melhores Cruzamentos (CSV)",
                    data=csv_best_data,
                    file_name=f"melhores_cruzamentos_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
                # Mostrar preview dos melhores cruzamentos
                st.write("**Preview dos Melhores Cruzamentos:**")
                st.dataframe(best_crossings_df.head(10), use_container_width=True)
                
            else:
                # Compatibilidade com versão anterior
                best_breeding_data = []
                for i, solution_item in enumerate(best_result['best_solution']):
                    if isinstance(solution_item, tuple):
                        pair1_idx, pair2_idx = solution_item
                        best_breeding_data.append({
                            'Par_Animal_1': dp.all_pairs[pair1_idx],
                            'Par_Animal_2': dp.all_pairs[pair2_idx],
                            'Coeficiente_Coancestralidade': dp.coancestry_matrix[pair1_idx, pair2_idx]
                        })
                    else:
                        male_idx = solution_item
                        best_breeding_data.append({
                            'Femea_ID': f'f{i+1}',
                            'Macho_ID': f'm{male_idx+1}',
                            'Femea_Original': dp.females[i] if hasattr(dp, 'females') else f'f{i+1}',
                            'Macho_Original': dp.males[male_idx] if hasattr(dp, 'males') else f'm{male_idx+1}',
                            'Coancestralidade': dp.breeding_matrix[i, male_idx] if hasattr(dp, 'breeding_matrix') else 0
                        })
                
                best_breeding_df = pd.DataFrame(best_breeding_data)
                
                # Ordenar por coancestralidade (do menor para o maior)
                sort_column = 'Coeficiente_Coancestralidade' if 'Coeficiente_Coancestralidade' in best_breeding_df.columns else 'Coancestralidade'
                best_breeding_df = best_breeding_df.sort_values(sort_column)
                
                csv_best_buffer = io.StringIO()
                best_breeding_df.to_csv(csv_best_buffer, index=False)
                csv_best_data = csv_best_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download Melhores Cruzamentos (CSV)",
                    data=csv_best_data,
                    file_name=f"melhores_cruzamentos_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
                # Mostrar preview dos melhores cruzamentos
                st.write("**Preview dos Melhores Cruzamentos:**")
                st.dataframe(best_breeding_df.head(10), use_container_width=True)
        
        with col2:
            # Matriz de melhores cruzamentos por fêmea
            if hasattr(dp, 'breeding_matrix') and hasattr(dp, 'females') and hasattr(dp, 'males'):
                st.write("**Melhores Cruzamentos por Fêmea:**")
                
                # Encontrar o melhor macho para cada fêmea
                best_males_per_female = []
                for i in range(len(dp.females)):
                    # Encontrar o macho com menor coancestralidade para esta fêmea
                    best_male_idx = np.argmin(dp.breeding_matrix[i, :])
                    best_coancestry = dp.breeding_matrix[i, best_male_idx]
                    
                    best_males_per_female.append({
                        'Fêmea': f'f{i+1}',
                        'Fêmea_Original': dp.females[i],
                        'Melhor_Macho': f'm{best_male_idx+1}',
                        'Macho_Original': dp.males[best_male_idx],
                        'Coancestralidade': best_coancestry
                    })
                
                best_males_df = pd.DataFrame(best_males_per_female)
                st.dataframe(best_males_df, use_container_width=True)
                
                # Download dos melhores cruzamentos por fêmea
                csv_best_males_buffer = io.StringIO()
                best_males_df.to_csv(csv_best_males_buffer, index=False)
                csv_best_males_data = csv_best_males_buffer.getvalue()
                
                st.download_button(
                    label="📋 Download Melhores Cruzamentos por Fêmea (CSV)",
                    data=csv_best_males_data,
                    file_name=f"melhores_cruzamentos_por_femea_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
                # Criar matriz com destaques dos melhores cruzamentos
                breeding_matrix_for_females = pd.DataFrame(
                    dp.breeding_matrix,
                    index=[f'f{i+1} ({dp.females[i]})' for i in range(len(dp.females))],
                    columns=[f'm{i+1} ({dp.males[i]})' for i in range(len(dp.males))]
                )
                
                # Destacar os melhores cruzamentos na matriz (apenas se for compatível)
                matrix_with_best = breeding_matrix_for_females.copy().astype(str)
                
                # Destacar melhor macho para cada fêmea
                for i in range(len(dp.females)):
                    best_male_idx = np.argmin(dp.breeding_matrix[i, :])
                    original_value = breeding_matrix_for_females.iloc[i, best_male_idx]
                    matrix_with_best.iloc[i, best_male_idx] = f"★ {original_value}"
                
                # Se há solução da otimização, destacar também
                if 'best_solution' in best_result and all(isinstance(x, int) for x in best_result['best_solution']):
                    for i, male_idx in enumerate(best_result['best_solution']):
                        if i < len(dp.females) and male_idx < len(dp.males):
                            original_value = breeding_matrix_for_females.iloc[i, male_idx]
                            matrix_with_best.iloc[i, male_idx] = f"🎯 {original_value}"
                
                csv_matrix_buffer = io.StringIO()
                matrix_with_best.to_csv(csv_matrix_buffer)
                csv_matrix_data = csv_matrix_buffer.getvalue()
                
                st.download_button(
                    label="📋 Download Matriz de Cruzamentos (CSV)",
                    data=csv_matrix_data,
                    file_name=f"matriz_cruzamentos_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
                # Mostrar preview da matriz
                st.write("**Preview da Matriz de Cruzamentos:**")
                st.write("(★ = melhor para cada fêmea, 🎯 = selecionado pela otimização)")
                st.dataframe(matrix_with_best.head(10), use_container_width=True)
            else:
                st.info("Matriz de breeding tradicional não disponível com a nova implementação.")
        
        # Matriz de Indicações: 5 melhores cruzamentos para cada fêmea
        st.header("🎯 Matriz de Indicações: Melhores Cruzamentos por Fêmea")
        
        # Configurações da matriz
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            num_recommendations = st.slider(
                "Número de indicações por fêmea:",
                min_value=1,
                max_value=10,
                value=5,
                help="Quantas indicações de machos mostrar para cada fêmea"
            )
        
        with col_config2:
            allow_reuse = st.checkbox(
                "Permitir reutilização de machos",
                value=False,
                help="Se marcado, um macho pode ser indicado para múltiplas fêmeas"
            )
        
        with col_config3:
            sort_by_quality = st.selectbox(
                "Ordenar fêmeas por:",
                options=["Ordem Original", "Melhor Coancestralidade", "Pior Coancestralidade"],
                index=1,
                help="Como ordenar as fêmeas na matriz"
            )
        
        st.info(f"Matriz mostrando as {num_recommendations} melhores indicações para cada fêmea, priorizando menor coancestralidade.")
        
        if hasattr(dp, 'breeding_matrix') and hasattr(dp, 'females') and hasattr(dp, 'males'):
            # Criar matriz de indicações
            num_females = len(dp.females)
            num_males = len(dp.males)
            
            # Algoritmo melhorado para distribuir machos para fêmeas
            def distribute_males_improved(breeding_matrix, females, males, max_recommendations_per_female=5, allow_reuse=False):
                """
                Distribui machos para fêmeas com configurações flexíveis.
                Prioriza melhores cruzamentos (menor coancestralidade).
                """
                num_females = len(females)
                num_males = len(males)
                
                # Criar lista de todos os cruzamentos possíveis
                all_crossings = []
                for i in range(num_females):
                    for j in range(num_males):
                        all_crossings.append({
                            'female_idx': i,
                            'male_idx': j,
                            'coancestry': breeding_matrix[i, j]
                        })
                
                # Ordenar por coancestralidade (melhor primeiro)
                all_crossings.sort(key=lambda x: x['coancestry'])
                
                # Inicializar matriz de recomendações
                female_recommendations = [[] for _ in range(num_females)]
                used_males = set()
                
                if allow_reuse:
                    # Permitir reutilização de machos - simplesmente pegar os melhores para cada fêmea
                    for i in range(num_females):
                        female_crossings = [c for c in all_crossings if c['female_idx'] == i]
                        female_recommendations[i] = female_crossings[:max_recommendations_per_female]
                else:
                    # Distribuir sem reutilização
                    # Primeira rodada: melhor para cada fêmea
                    for crossing in all_crossings:
                        female_idx = crossing['female_idx']
                        male_idx = crossing['male_idx']
                        
                        if len(female_recommendations[female_idx]) == 0 and male_idx not in used_males:
                            female_recommendations[female_idx].append(crossing)
                            used_males.add(male_idx)
                    
                    # Rodadas subsequentes: completar recomendações
                    for round_num in range(1, max_recommendations_per_female):
                        random.shuffle(all_crossings)  # Embaralhar para diversidade
                        for crossing in all_crossings:
                            female_idx = crossing['female_idx']
                            male_idx = crossing['male_idx']
                            
                            if (len(female_recommendations[female_idx]) == round_num and 
                                male_idx not in used_males):
                                female_recommendations[female_idx].append(crossing)
                                used_males.add(male_idx)
                                
                                if len(used_males) >= num_males:
                                    break
                        
                        if len(used_males) >= num_males:
                            break
                
                return female_recommendations
            
            # Gerar recomendações com configurações personalizadas
            female_recommendations = distribute_males_improved(
                dp.breeding_matrix, dp.females, dp.males, 
                num_recommendations, allow_reuse
            )
            
            # Criar matriz de indicações
            recommendations_matrix = []
            
            for i in range(num_females):
                # Criar linha da matriz
                female_row = {
                    'Fêmea': f'f{i+1}',
                    'ID_Original': dp.females[i]
                }
                
                # Calcular estatísticas da fêmea
                if female_recommendations[i]:
                    best_coancestry = female_recommendations[i][0]['coancestry']
                    avg_coancestry = np.mean([c['coancestry'] for c in female_recommendations[i]])
                    female_row['Melhor_Coancestralidade'] = best_coancestry
                    female_row['Média_Coancestralidade'] = avg_coancestry
                else:
                    female_row['Melhor_Coancestralidade'] = 1.0
                    female_row['Média_Coancestralidade'] = 1.0
                
                # Adicionar as recomendações para esta fêmea
                for j in range(num_recommendations):
                    if j < len(female_recommendations[i]):
                        crossing = female_recommendations[i][j]
                        male_idx = crossing['male_idx']
                        coancestry = crossing['coancestry']
                        
                        # Classificar qualidade do cruzamento
                        if coancestry < 0.05:
                            quality = "🟢 Excelente"
                        elif coancestry < 0.15:
                            quality = "🟡 Bom"
                        elif coancestry < 0.3:
                            quality = "🟠 Regular"
                        else:
                            quality = "🔴 Evitar"
                        
                        male_info = f"m{male_idx+1} ({dp.males[male_idx]}) - {coancestry:.6f} {quality}"
                        female_row[f'1° Melhor' if j == 0 else f'{j+1}° Melhor'] = male_info
                    else:
                        female_row[f'{j+1}° Melhor'] = "N/A - Sem machos disponíveis"
                
                recommendations_matrix.append(female_row)
            
            # Converter para DataFrame
            recommendations_df = pd.DataFrame(recommendations_matrix)
            
            # Ordenar o DataFrame conforme solicitado
            if sort_by_quality == "Melhor Coancestralidade":
                recommendations_df = recommendations_df.sort_values('Melhor_Coancestralidade').reset_index(drop=True)
            elif sort_by_quality == "Pior Coancestralidade":
                recommendations_df = recommendations_df.sort_values('Melhor_Coancestralidade', ascending=False).reset_index(drop=True)
            
            # Mostrar estatísticas detalhadas
            st.subheader("📊 Estatísticas da Matriz de Indicações")
            
            # Calcular estatísticas
            all_coancestries = []
            best_coancestries = []
            males_used = set()
            quality_counts = {"🟢 Excelente": 0, "🟡 Bom": 0, "🟠 Regular": 0, "🔴 Evitar": 0}
            
            for i in range(num_females):
                if female_recommendations[i]:
                    best_coancestries.append(female_recommendations[i][0]['coancestry'])
                    for rec in female_recommendations[i]:
                        all_coancestries.append(rec['coancestry'])
                        males_used.add(rec['male_idx'])
                        
                        # Contar qualidade
                        if rec['coancestry'] < 0.05:
                            quality_counts["🟢 Excelente"] += 1
                        elif rec['coancestry'] < 0.15:
                            quality_counts["🟡 Bom"] += 1
                        elif rec['coancestry'] < 0.3:
                            quality_counts["🟠 Regular"] += 1
                        else:
                            quality_counts["🔴 Evitar"] += 1
            
            # Primeira linha de estatísticas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Fêmeas", num_females)
            
            with col2:
                st.metric("Total de Machos", num_males)
            
            with col3:
                males_used_count = len(males_used)
                usage_percentage = (males_used_count / num_males) * 100 if num_males > 0 else 0
                st.metric("Machos Utilizados", f"{males_used_count} ({usage_percentage:.1f}%)")
            
            with col4:
                avg_best = np.mean(best_coancestries) if best_coancestries else 0
                st.metric("Coancestralidade Média (Melhores)", f"{avg_best:.6f}")
            
            # Segunda linha de estatísticas
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("🟢 Excelentes", quality_counts["🟢 Excelente"])
            
            with col6:
                st.metric("🟡 Bons", quality_counts["🟡 Bom"])
            
            with col7:
                st.metric("🟠 Regulares", quality_counts["🟠 Regular"])
            
            with col8:
                st.metric("🔴 A Evitar", quality_counts["🔴 Evitar"])
            
            # Mostrar a matriz com estilo
            st.subheader("📋 Matriz de Indicações")
            
            # Criar visualização da matriz
            display_df = recommendations_df.copy()
            
            # Remover colunas auxiliares para exibição
            cols_to_remove = ['Melhor_Coancestralidade', 'Média_Coancestralidade']
            display_df = display_df.drop(columns=[col for col in cols_to_remove if col in display_df.columns])
            
            # Mostrar informações adicionais
            st.info(f"🔍 Ordenação: {sort_by_quality} | 📊 Recomendações: {num_recommendations} por fêmea | 🔄 Reutilização: {'Permitida' if allow_reuse else 'Não permitida'}")
            
            # Criar abas para diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["📋 Matriz Completa", "📊 Análise de Qualidade", "📈 Gráficos"])
            
            with tab1:
                st.dataframe(display_df, use_container_width=True, height=400)
            
            with tab2:
                # Análise de qualidade por fêmea
                quality_analysis = []
                for i in range(num_females):
                    if female_recommendations[i]:
                        female_quality = {
                            'Fêmea': f'f{i+1}',
                            'ID_Original': dp.females[i],
                            'Melhor_Coancestralidade': female_recommendations[i][0]['coancestry'],
                            'Média_Coancestralidade': np.mean([c['coancestry'] for c in female_recommendations[i]]),
                            'Num_Excelentes': sum(1 for c in female_recommendations[i] if c['coancestry'] < 0.05),
                            'Num_Bons': sum(1 for c in female_recommendations[i] if 0.05 <= c['coancestry'] < 0.15),
                            'Num_Regulares': sum(1 for c in female_recommendations[i] if 0.15 <= c['coancestry'] < 0.3),
                            'Num_Evitar': sum(1 for c in female_recommendations[i] if c['coancestry'] >= 0.3)
                        }
                        quality_analysis.append(female_quality)
                
                quality_df = pd.DataFrame(quality_analysis)
                st.dataframe(quality_df, use_container_width=True)
            
            with tab3:
                # Gráficos de análise
                if all_coancestries:
                    col_graph1, col_graph2 = st.columns(2)
                    
                    with col_graph1:
                        # Histograma de coancestralidade
                        fig_hist = px.histogram(
                            x=all_coancestries,
                            nbins=20,
                            title="Distribuição da Coancestralidade nas Indicações",
                            labels={'x': 'Coancestralidade', 'y': 'Frequência'},
                            color_discrete_sequence=['#2E86AB']
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_graph2:
                        # Gráfico de barras da qualidade
                        quality_data = pd.DataFrame(list(quality_counts.items()), columns=['Qualidade', 'Contagem'])
                        fig_quality = px.bar(
                            quality_data,
                            x='Qualidade',
                            y='Contagem',
                            title="Distribuição da Qualidade das Indicações",
                            color='Qualidade',
                            color_discrete_map={
                                '🟢 Excelente': '#28a745',
                                '🟡 Bom': '#ffc107',
                                '🟠 Regular': '#fd7e14',
                                '🔴 Evitar': '#dc3545'
                            }
                        )
                        st.plotly_chart(fig_quality, use_container_width=True)
            
            # Criar matriz melhorada para download
            st.subheader("💾 Download da Matriz de Indicações")
            
            # Preparar dados para download
            download_matrix = []
            for i in range(num_females):
                base_row = {
                    'Femea': f'f{i+1}',
                    'Femea_Original': dp.females[i],
                    'Melhor_Coancestralidade': female_recommendations[i][0]['coancestry'] if female_recommendations[i] else 1.0,
                    'Media_Coancestralidade': np.mean([c['coancestry'] for c in female_recommendations[i]]) if female_recommendations[i] else 1.0,
                    'Num_Indicacoes': len(female_recommendations[i])
                }
                
                # Adicionar as recomendações desta fêmea
                for j in range(num_recommendations):
                    if j < len(female_recommendations[i]):
                        crossing = female_recommendations[i][j]
                        male_idx = crossing['male_idx']
                        coancestry = crossing['coancestry']
                        
                        # Classificar qualidade
                        if coancestry < 0.05:
                            quality = "Excelente"
                        elif coancestry < 0.15:
                            quality = "Bom"
                        elif coancestry < 0.3:
                            quality = "Regular"
                        else:
                            quality = "Evitar"
                        
                        base_row[f'Macho_{j+1}'] = f"m{male_idx+1}"
                        base_row[f'Macho_Original_{j+1}'] = dp.males[male_idx]
                        base_row[f'Coancestralidade_{j+1}'] = coancestry
                        base_row[f'Qualidade_{j+1}'] = quality
                    else:
                        base_row[f'Macho_{j+1}'] = "N/A"
                        base_row[f'Macho_Original_{j+1}'] = "N/A"
                        base_row[f'Coancestralidade_{j+1}'] = "N/A"
                        base_row[f'Qualidade_{j+1}'] = "N/A"
                
                download_matrix.append(base_row)
            
            download_df = pd.DataFrame(download_matrix)
            
            # Opções de download
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download completo com todas as colunas
                csv_buffer = io.StringIO()
                download_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="💾 Download Matriz Completa (CSV)",
                    data=csv_data,
                    file_name=f"matriz_indicacoes_completa_{int(time.time())}.csv",
                    mime="text/csv",
                    help="Inclui todas as colunas com estatísticas e qualidade"
                )
            
            with col_dl2:
                # Download simplificado apenas com indicações
                simple_cols = ['Femea', 'Femea_Original'] + [f'Macho_Original_{j+1}' for j in range(num_recommendations)] + [f'Coancestralidade_{j+1}' for j in range(num_recommendations)]
                simple_df = download_df[simple_cols]
                
                csv_buffer_simple = io.StringIO()
                simple_df.to_csv(csv_buffer_simple, index=False)
                csv_simple_data = csv_buffer_simple.getvalue()
                
                st.download_button(
                    label="💾 Download Matriz Simples (CSV)",
                    data=csv_simple_data,
                    file_name=f"matriz_indicacoes_simples_{int(time.time())}.csv",
                    mime="text/csv",
                    help="Apenas fêmeas, machos e coancestralidade"
                )
            
            # Criar função para destacar melhores recomendações
            def create_best_crossings_matrix(female_recommendations, num_females, num_males):
                """
                Cria uma matriz reduzida contendo apenas os melhores cruzamentos identificados.
                """
                best_matrix = np.ones((num_females, num_males)) * np.nan
                
                for i in range(num_females):
                    if female_recommendations[i]:
                        best_crossing = female_recommendations[i][0]
                        best_matrix[i, best_crossing['male_idx']] = best_crossing['coancestry']
                
                return best_matrix
            
            # Criar matriz dos melhores cruzamentos
            best_matrix = create_best_crossings_matrix(female_recommendations, num_females, num_males)
            
            # Visualização da matriz otimizada
            st.subheader("🗺️ Visualização da Matriz Otimizada")
            
            # Mostrar apenas uma parte da matriz se for muito grande
            matrix_size = min(num_females, num_males, 20)  # Limitar para visualização
            
            if matrix_size < min(num_females, num_males):
                st.info(f"Mostrando apenas os primeiros {matrix_size}x{matrix_size} elementos da matriz para melhor visualização")
            
            # Criar heatmap da matriz breeding reduzida
            breeding_subset = dp.breeding_matrix[:matrix_size, :matrix_size]
            best_subset = best_matrix[:matrix_size, :matrix_size]
            
            fig_heatmap = px.imshow(
                breeding_subset,
                labels=dict(x="Machos", y="Fêmeas", color="Coancestralidade"),
                x=[f'm{i+1}' for i in range(matrix_size)],
                y=[f'f{i+1}' for i in range(matrix_size)],
                color_continuous_scale="RdYlBu_r",
                title="Matriz de Coancestralidade com Melhores Indicações"
            )
            
            # Adicionar marcadores para os melhores cruzamentos
            for i in range(matrix_size):
                for j in range(matrix_size):
                    if not np.isnan(best_subset[i, j]):
                        fig_heatmap.add_scatter(
                            x=[j],
                            y=[i],
                            mode='markers',
                            marker=dict(symbol='star', size=15, color='red', line=dict(width=2, color='white')),
                            name='Melhor Indicação',
                            showlegend=False
                        )
            
            fig_heatmap.update_layout(
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Tabela resumo das fêmeas com melhor potencial
            st.subheader("🏆 Top 10 Fêmeas com Melhor Potencial")
            
            female_potential = []
            for i in range(num_females):
                if female_recommendations[i]:
                    best_crossing = female_recommendations[i][0]
                    best_coancestry = best_crossing['coancestry']
                    best_male_idx = best_crossing['male_idx']
                    
                    female_potential.append({
                        'Posição': 0,
                        'Fêmea': f'f{i+1}',
                        'ID_Original': dp.females[i],
                        'Melhor_Coancestralidade': best_coancestry,
                        'Melhor_Macho': f"m{best_male_idx+1}",
                        'Macho_Original': dp.males[best_male_idx],
                        'Qualidade': 'Excelente' if best_coancestry < 0.1 else 'Boa' if best_coancestry < 0.2 else 'Regular',
                        'Total_Indicacoes': len(female_recommendations[i])
                    })
                else:  # Se não há recomendações
                    female_potential.append({
                        'Posição': 0,
                        'Fêmea': f'f{i+1}',
                        'ID_Original': dp.females[i],
                        'Melhor_Coancestralidade': float('inf'),
                        'Melhor_Macho': "N/A",
                        'Macho_Original': "N/A",
                        'Qualidade': 'Sem indicação',
                        'Total_Indicacoes': 0
                    })
            
            # Ordenar por melhor coancestralidade
            female_potential.sort(key=lambda x: x['Melhor_Coancestralidade'])
            
            # Adicionar posição
            for i, fp in enumerate(female_potential):
                if fp['Melhor_Coancestralidade'] != float('inf'):
                    fp['Posição'] = i + 1
                else:
                    fp['Posição'] = '-'
            
            # Mostrar top 10
            top_10_df = pd.DataFrame(female_potential[:10])
            st.dataframe(top_10_df, use_container_width=True)
            
            # Informações sobre distribuição única
            total_males_used = len(set(crossing['male_idx'] 
                                     for recommendations in female_recommendations 
                                     for crossing in recommendations))
            st.info(f"🎯 **Distribuição Única**: {total_males_used} machos diferentes foram utilizados para todas as recomendações, evitando repetições.")
            
            # GRASP na Matriz de Melhores Cruzamentos
            st.header("🔬 Otimização GRASP dos Melhores Cruzamentos")
            st.info("Aplicando algoritmo GRASP especificamente na matriz de melhores cruzamentos para encontrar a solução ótima.")
            
            # Criar matriz reduzida com apenas os melhores cruzamentos
            def create_best_crossings_matrix(female_recommendations, num_females, num_males):
                """
                Cria uma matriz reduzida contendo apenas os melhores cruzamentos identificados.
                """
                # Coletar todos os cruzamentos únicos das recomendações
                all_crossings = []
                for i in range(num_females):
                    for crossing in female_recommendations[i]:
                        # Adicionar todos os cruzamentos (não apenas únicos por macho)
                        all_crossings.append({
                            'female_idx': crossing['female_idx'],
                            'male_idx': crossing['male_idx'],
                            'coancestry': crossing['coancestry'],
                            'pair_name': f"f{crossing['female_idx']+1}_m{crossing['male_idx']+1}"
                        })
                
                # Remover duplicatas mantendo o melhor para cada par fêmea-macho
                unique_crossings = {}
                for crossing in all_crossings:
                    key = f"{crossing['female_idx']}_{crossing['male_idx']}"
                    if key not in unique_crossings or crossing['coancestry'] < unique_crossings[key]['coancestry']:
                        unique_crossings[key] = crossing
                
                best_crossings_data = list(unique_crossings.values())
                
                # Criar matriz quadrada de coancestralidade para GRASP
                matrix_size = len(best_crossings_data)
                grasp_matrix = np.full((matrix_size, matrix_size), 1.0)  # Inicializar com valor alto
                
                # Diagonal principal com os coeficientes de coancestralidade
                for i in range(matrix_size):
                    grasp_matrix[i, i] = best_crossings_data[i]['coancestry']
                
                # Penalizar cruzamentos que usam a mesma fêmea ou mesmo macho
                for i in range(matrix_size):
                    for j in range(i + 1, matrix_size):
                        crossing_i = best_crossings_data[i]
                        crossing_j = best_crossings_data[j]
                        
                        # Se usam a mesma fêmea ou mesmo macho, aumentar penalidade
                        if (crossing_i['female_idx'] == crossing_j['female_idx'] or 
                            crossing_i['male_idx'] == crossing_j['male_idx']):
                            penalty = max(crossing_i['coancestry'], crossing_j['coancestry']) * 10
                            grasp_matrix[i, j] = penalty
                            grasp_matrix[j, i] = penalty
                        else:
                            # Cruzamentos compatíveis têm menor penalidade
                            avg_coancestry = (crossing_i['coancestry'] + crossing_j['coancestry']) / 2
                            grasp_matrix[i, j] = avg_coancestry
                            grasp_matrix[j, i] = avg_coancestry
                
                return grasp_matrix, best_crossings_data
            
            # Configuração do GRASP
            col_grasp1, col_grasp2, col_grasp3 = st.columns(3)
            
            with col_grasp1:
                grasp_iterations = st.slider(
                    "Iterações GRASP:",
                    min_value=10,
                    max_value=200,
                    value=100,
                    help="Número de iterações para o algoritmo GRASP"
                )
            
            with col_grasp2:
                grasp_alpha = st.slider(
                    "Alpha (Greedy Factor):",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="0.1 = mais greedy, 1.0 = mais randômico"
                )
            
            with col_grasp3:
                max_range = max(10, min(num_females, total_males_used))  # Garantir mínimo de 10
                max_selected_crossings = st.slider(
                    "Máximo de cruzamentos:",
                    min_value=3,
                    max_value=max_range,
                    value=min(50, max_range),
                    help="Número máximo de cruzamentos para selecionar"
                )
            
            if st.button("🚀 Executar GRASP nos Melhores Cruzamentos"):
                with st.spinner("Executando otimização GRASP..."):
                    # Criar matriz dos melhores cruzamentos
                    grasp_matrix, best_crossings_data = create_best_crossings_matrix(
                        female_recommendations, num_females, num_males
                    )
                    
                    st.write(f"Debug: Matriz GRASP criada com {len(best_crossings_data)} cruzamentos")
                    
                    if len(best_crossings_data) > 0:
                        # Implementar GRASP especializado para seleção de cruzamentos
                        def grasp_crossing_selection(crossings_data, matrix, max_selections, iterations, alpha):
                            """
                            GRASP especializado para seleção de melhores cruzamentos por fêmea.
                            """
                            best_solution = None
                            best_cost = float('inf')
                            iteration_costs = []
                            
                            for iteration in range(iterations):
                                # Fase de Construção Greedy Randomizada
                                selected_indices = []
                                used_females = set()
                                used_males = set()
                                current_cost = 0
                                
                                # Criar lista de candidatos
                                candidates = list(range(len(crossings_data)))
                                
                                while len(selected_indices) < max_selections and candidates:
                                    # Calcular custos para candidatos válidos
                                    valid_candidates = []
                                    for idx in candidates:
                                        crossing = crossings_data[idx]
                                        # Verificar se não conflita com seleções anteriores
                                        if (crossing['female_idx'] not in used_females and 
                                            crossing['male_idx'] not in used_males):
                                            valid_candidates.append((idx, crossing['coancestry']))
                                    
                                    if not valid_candidates:
                                        break
                                    
                                    # Ordenar por coancestralidade (menor é melhor)
                                    valid_candidates.sort(key=lambda x: x[1])
                                    
                                    # Seleção greedy randomizada
                                    rcl_size = max(1, int(len(valid_candidates) * alpha))
                                    rcl = valid_candidates[:rcl_size]
                                    
                                    # Escolher aleatoriamente da RCL
                                    selected_idx, cost = rcl[np.random.randint(len(rcl))]
                                    selected_indices.append(selected_idx)
                                    
                                    crossing = crossings_data[selected_idx]
                                    used_females.add(crossing['female_idx'])
                                    used_males.add(crossing['male_idx'])
                                    current_cost += cost
                                    
                                    # Remover candidato selecionado
                                    candidates.remove(selected_idx)
                                
                                # Busca Local simples - tentar trocar cruzamentos
                                improved = True
                                while improved:
                                    improved = False
                                    for i in range(len(selected_indices)):
                                        current_idx = selected_indices[i]
                                        current_crossing = crossings_data[current_idx]
                                        
                                        # Tentar substituir por um melhor cruzamento para a mesma fêmea
                                        for j, crossing in enumerate(crossings_data):
                                            if (j not in selected_indices and 
                                                crossing['female_idx'] == current_crossing['female_idx'] and
                                                crossing['male_idx'] not in [crossings_data[idx]['male_idx'] for k, idx in enumerate(selected_indices) if k != i] and
                                                crossing['coancestry'] < current_crossing['coancestry']):
                                                
                                                # Fazer a troca
                                                new_cost = current_cost - current_crossing['coancestry'] + crossing['coancestry']
                                                if new_cost < current_cost:
                                                    selected_indices[i] = j
                                                    current_cost = new_cost
                                                    improved = True
                                                    break
                                
                                iteration_costs.append(current_cost)
                                
                                # Verificar se é a melhor solução
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_solution = selected_indices.copy()
                            
                            return best_solution, best_cost, iteration_costs
                        
                        # Executar GRASP personalizado
                        best_solution, best_cost, iteration_costs = grasp_crossing_selection(
                            best_crossings_data, grasp_matrix, max_selected_crossings, grasp_iterations, grasp_alpha
                        )
                        
                        # Mostrar resultados
                        st.success(f"✅ Otimização concluída! Custo final: {best_cost:.6f}")
                        
                        # Converter solução para detalhes de cruzamentos
                        selected_crossings_details = []
                        if best_solution:
                            for i, crossing_idx in enumerate(best_solution):
                                crossing = best_crossings_data[crossing_idx]
                                selected_crossings_details.append({
                                    'Posição': i + 1,
                                    'Fêmea': f"f{crossing['female_idx']+1}",
                                    'Fêmea_Original': dp.females[crossing['female_idx']],
                                    'Macho': f"m{crossing['male_idx']+1}",
                                    'Macho_Original': dp.males[crossing['male_idx']],
                                    'Coancestralidade': crossing['coancestry'],
                                    'Qualidade': 'Excelente' if crossing['coancestry'] < 0.1 else 'Boa' if crossing['coancestry'] < 0.2 else 'Regular'
                                })
                        
                        # Mostrar tabela de resultados
                        st.subheader("🏆 Cruzamentos Selecionados pelo GRASP")
                        
                        if selected_crossings_details:
                            results_df = pd.DataFrame(selected_crossings_details)
                            
                            # Destacar por qualidade
                            def highlight_grasp_quality(row):
                                if row['Qualidade'] == 'Excelente':
                                    return ['background-color: #d4edda'] * len(row)
                                elif row['Qualidade'] == 'Boa':
                                    return ['background-color: #fff3cd'] * len(row)
                                else:
                                    return ['background-color: #f8d7da'] * len(row)
                            
                            styled_results_df = results_df.style.apply(highlight_grasp_quality, axis=1)
                            st.dataframe(styled_results_df, use_container_width=True)
                            
                            # Estatísticas da solução GRASP
                            st.subheader("📊 Estatísticas da Solução GRASP")
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            
                            with col_stat1:
                                st.metric("Cruzamentos Selecionados", len(selected_crossings_details))
                            
                            with col_stat2:
                                avg_coancestry = np.mean([c['Coancestralidade'] for c in selected_crossings_details])
                                st.metric("Coancestralidade Média", f"{avg_coancestry:.6f}")
                            
                            with col_stat3:
                                excellent_count = sum(1 for c in selected_crossings_details if c['Qualidade'] == 'Excelente')
                                st.metric("Cruzamentos Excelentes", excellent_count)
                            
                            with col_stat4:
                                unique_females = len(set(c['Fêmea'] for c in selected_crossings_details))
                                st.metric("Fêmeas Contempladas", unique_females)
                            
                            # Gráfico de convergência
                            st.subheader("📈 Convergência do GRASP")
                            
                            convergence_fig = px.line(
                                x=list(range(1, len(iteration_costs) + 1)),
                                y=iteration_costs,
                                title="Evolução do Custo por Iteração",
                                labels={'x': 'Iteração', 'y': 'Custo (Coancestralidade)'}
                            )
                            convergence_fig.update_layout(showlegend=False)
                            st.plotly_chart(convergence_fig, use_container_width=True)
                            
                            # Download da solução GRASP
                            csv_grasp_buffer = io.StringIO()
                            results_df.to_csv(csv_grasp_buffer, index=False)
                            csv_grasp_data = csv_grasp_buffer.getvalue()
                            
                            st.download_button(
                                label="📊 Download Solução GRASP (CSV)",
                                data=csv_grasp_data,
                                file_name=f"solucao_grasp_melhores_cruzamentos_{int(time.time())}.csv",
                                mime="text/csv"
                            )
                        
                        else:
                            st.warning("Nenhum cruzamento foi selecionado pela otimização.")
                    
                    else:
                        st.error("Não foi possível criar matriz de melhores cruzamentos.")
        
        else:
            st.error("Dados de breeding matrix não disponíveis.")
        
        # Resumo estatístico dos melhores cruzamentos
        st.subheader("📊 Resumo Estatístico dos Melhores Cruzamentos")
        
        # Determinar qual dataset usar para estatísticas
        if 'best_crossings' in best_result and best_result['best_crossings']:
            stats_data = [c['coancestry'] for c in best_result['best_crossings']]
            coef_column = 'coancestry'
        elif hasattr(dp, 'breeding_matrix') and hasattr(dp, 'females'):
            stats_data = [np.min(dp.breeding_matrix[i, :]) for i in range(len(dp.females))]
            coef_column = 'min_coancestry'
        else:
            stats_data = [0]  # Fallback
            coef_column = 'fallback'
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Coancestralidade Média",
                f"{np.mean(stats_data):.6f}"
            )
        
        with col2:
            st.metric(
                "Coancestralidade Mínima",
                f"{np.min(stats_data):.6f}"
            )
        
        with col3:
            st.metric(
                "Coancestralidade Máxima",
                f"{np.max(stats_data):.6f}"
            )
        
        with col4:
            st.metric(
                "Desvio Padrão",
                f"{np.std(stats_data):.6f}"
            )
        
        # Gráfico da distribuição dos melhores cruzamentos
        st.subheader("📈 Distribuição dos Coeficientes de Coancestralidade")
        
        dist_fig = px.histogram(
            x=stats_data,
            nbins=15,
            title="Distribuição dos Coeficientes na Melhor Solução",
            labels={'x': 'Coeficiente de Coancestralidade', 'y': 'Frequência'}
        )
        dist_fig.update_layout(showlegend=False)
        st.plotly_chart(dist_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Sistema de Otimização de Acasalamento Animal** - Desenvolvido para o Projeto Acadêmico APA 2025/1")
st.markdown("*Usando GRASP (Greedy Randomized Adaptive Search Procedure) Meta-heurística*")