import numpy as np
import random
from typing import List, Tuple, Callable, Optional

class GRASPOptimizer:
    """
    Otimizador GRASP (Greedy Randomized Adaptive Search Procedure) para 
    otimização de acasalamento animal para minimizar coancestralidade trabalhando diretamente na matriz de cruzamentos.
    """
    
    def __init__(self, coancestry_matrix: np.ndarray, max_iterations: int = 200, 
                 alpha: float = 0.3, local_search_iterations: int = 30, 
                 pair_names: list = None):
        """
        Inicializa o otimizador GRASP.
        
        Args:
            coancestry_matrix: Matriz quadrada de valores de coancestralidade entre todos os pares de cruzamento
            max_iterations: Número máximo de iterações GRASP
            alpha: Parâmetro guloso (0 = puramente guloso, 1 = puramente aleatório)
            local_search_iterations: Número de iterações de busca local
            pair_names: Lista de nomes de pares correspondentes aos índices da matriz
        """
        self.coancestry_matrix = coancestry_matrix
        self.matrix_size = coancestry_matrix.shape[0]
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.local_search_iterations = local_search_iterations
        self.pair_names = pair_names if pair_names else [f'P{i+1}' for i in range(self.matrix_size)]
        
        # Para rastrear convergência
        self.iteration_costs = []
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_crossings = []
    
    def calculate_total_cost(self, selected_crossings: List[int]) -> float:
        """
        Calcula o custo total de coancestralidade para os cruzamentos selecionados da matriz.
        
        Args:
            selected_crossings: Lista de índices representando pares de cruzamento selecionados
            
        Returns:
            Custo total de coancestralidade (soma dos coeficientes de coancestralidade selecionados)
        """
        total_cost = 0.0
        
        # Somar os coeficientes de coancestralidade dos cruzamentos selecionados
        for crossing_idx in selected_crossings:
            # Usar apenas a diagonal inferior para evitar duplicatas
            row = crossing_idx // self.matrix_size
            col = crossing_idx % self.matrix_size
            
            # Só considerar se não for a diagonal principal (mesmo animal)
            if row != col:
                total_cost += self.coancestry_matrix[row, col]
        
        return total_cost
    
    def calculate_crossing_matrix_cost(self, solution_matrix: np.ndarray) -> float:
        """
        Calcula o custo diretamente de uma matriz solução onde 1 indica cruzamentos selecionados.
        
        Args:
            solution_matrix: Matriz binária indicando cruzamentos selecionados
            
        Returns:
            Custo total de coancestralidade
        """
        total_cost = 0.0
        
        # Somar apenas os valores onde a matriz solução indica 1
        for i in range(self.matrix_size):
            for j in range(i + 1, self.matrix_size):  # Usar apenas triangular superior
                if solution_matrix[i, j] == 1:
                    total_cost += self.coancestry_matrix[i, j]
        
        return total_cost
    
    def greedy_randomized_construction(self, num_crossings: int = None) -> List[Tuple[int, int]]:
        """
        Constrói uma solução usando construção gulosa randomizada trabalhando na matriz de cruzamentos.
        Versão otimizada com cruzamentos pré-ordenados e randomização melhorada.
        MODIFICADO: Usa no mínimo 80% dos pares disponíveis no CSV.
        
        Args:
            num_crossings: Número de cruzamentos a selecionar (padrão: 80% dos pares disponíveis)
            
        Returns:
            Uma solução como lista de pares de cruzamento (linha, coluna)
        """
        # Usar os pares disponíveis no CSV (não a matriz completa)
        available_pairs = len(self.pair_names) if self.pair_names else self.matrix_size
        
        if num_crossings is None:
            # Usar no mínimo 80% dos pares disponíveis
            num_crossings = max(3, int(available_pairs * 0.8))
        else:
            # Garantir que seja pelo menos 80% dos pares disponíveis ou o valor solicitado (o que for maior)
            min_crossings = int(available_pairs * 0.8)
            num_crossings = max(min_crossings, num_crossings)
        
        solution = []
        used_pairs = set()
        used_animals = set()  # Para evitar sequência de animais
        
        # Pré-calcular todos os cruzamentos possíveis ordenados (cache)
        if not hasattr(self, '_sorted_crossings'):
            all_crossings = []
            for i in range(self.matrix_size):
                for j in range(i + 1, self.matrix_size):
                    all_crossings.append((i, j, self.coancestry_matrix[i, j]))
            
            # Ordenar por coeficiente de coancestralidade (menor é melhor)
            all_crossings.sort(key=lambda x: x[2])
            self._sorted_crossings = all_crossings
        
        # Embaralhar os candidatos para evitar sequência
        candidate_pool = self._sorted_crossings.copy()
        random.shuffle(candidate_pool)
        
        # MODIFICADO: Usar todos os candidatos disponíveis para garantir 80% de uso da matriz
        # Não limitar o número de candidatos para permitir seleção de 80% dos pares
        max_candidates = len(candidate_pool)  # Usar todos os candidatos disponíveis
        candidate_pool = candidate_pool[:max_candidates]
        
        # Reordenar por coancestralidade após embaralhamento
        candidate_pool.sort(key=lambda x: x[2])
        
        attempts = 0
        max_attempts = num_crossings * 2  # Permitir mais tentativas
        
        while len(solution) < num_crossings and attempts < max_attempts:
            attempts += 1
            
            # Criar lista de candidatos ainda não utilizados
            candidates = []
            for i, j, coef in candidate_pool:
                # Priorizar pares não utilizados
                if (i, j) not in used_pairs:
                    candidates.append((i, j, coef))
            
            # Se não há mais candidatos disponíveis, parar
            if not candidates:
                break
            
            # MODIFICADO: Usar mais candidatos para garantir 80% de uso da matriz
            # Usar mais candidatos para ter maior diversidade e alcançar 80% de uso
            num_rcl_candidates = min(len(candidates), max(200, len(candidates) // 2))
            top_candidates = candidates[:num_rcl_candidates]
            
            # Criar lista restrita de candidatos (RCL) com diversidade
            min_cost = top_candidates[0][2]
            max_cost = top_candidates[-1][2]
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            
            rcl = [(i, j) for i, j, coef in top_candidates if coef <= threshold]
            
            # Adicionar algumas opções aleatórias para diversidade
            if len(rcl) < 10 and len(candidates) > len(rcl):
                additional_candidates = random.sample(candidates[len(rcl):], 
                                                    min(5, len(candidates) - len(rcl)))
                rcl.extend([(i, j) for i, j, coef in additional_candidates])
            
            # Selecionar aleatoriamente da RCL
            selected_crossing = random.choice(rcl)
            solution.append(selected_crossing)
            used_pairs.add(selected_crossing)
            
            # MODIFICADO: Não restringir animais para permitir usar 80% da matriz
            # Remover restrições de animais usados para permitir maior cobertura da matriz
        
        return solution
    
    def get_matrix_usage_statistics(self, solution: List[Tuple[int, int]]) -> dict:
        """
        Calcula estatísticas sobre o uso da matriz de pares.
        
        Args:
            solution: Lista de pares selecionados
            
        Returns:
            Dicionário com estatísticas de uso da matriz
        """
        total_possible_pairs = (self.matrix_size * (self.matrix_size - 1)) // 2
        pairs_used = len(solution)
        usage_percentage = (pairs_used / total_possible_pairs) * 100
        
        return {
            'total_possible_pairs': total_possible_pairs,
            'pairs_used': pairs_used,
            'usage_percentage': usage_percentage,
            'meets_80_percent_target': usage_percentage >= 80.0
        }
    
    def local_search(self, solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Perform local search to improve the solution working on crossing pairs.
        Optimized version with limited search space and randomization.
        
        Args:
            solution: Current solution as list of crossing pairs
            
        Returns:
            Improved solution
        """
        current_solution = solution.copy()
        current_cost = self.calculate_crossing_cost(current_solution)
        
        improved = True
        iterations = 0
        
        # Pré-calcular todos os cruzamentos possíveis ordenados por coancestralidade
        if not hasattr(self, '_sorted_crossings'):
            all_crossings = []
            for i in range(self.matrix_size):
                for j in range(i + 1, self.matrix_size):
                    all_crossings.append((i, j, self.coancestry_matrix[i, j]))
            
            # Ordenar por coancestralidade (menor primeiro)
            all_crossings.sort(key=lambda x: x[2])
            self._sorted_crossings = all_crossings
        
        # Embaralhar candidatos para evitar padrões sequenciais
        candidate_pool = self._sorted_crossings.copy()
        random.shuffle(candidate_pool)
        
        # MODIFICADO: Usar todos os candidatos disponíveis para busca local ampla
        # Usar todos os candidatos para permitir busca mais abrangente
        max_candidates = len(candidate_pool)  # Usar todos os candidatos disponíveis
        top_candidates = candidate_pool[:max_candidates]
        
        # Reordenar por coancestralidade após embaralhamento
        top_candidates.sort(key=lambda x: x[2])
        
        while improved and iterations < self.local_search_iterations:
            improved = False
            iterations += 1
            
            # Embaralhar a ordem de verificação das soluções atuais
            solution_indices = list(range(len(current_solution)))
            random.shuffle(solution_indices)
            
            # Tentar trocar cruzamentos da solução atual por candidatos melhores
            for idx in solution_indices:
                current_crossing = current_solution[idx]
                current_i, current_j = current_crossing
                current_crossing_cost = self.coancestry_matrix[current_i, current_j]
                
                # Embaralhar candidatos para cada verificação
                random.shuffle(top_candidates)
                
                # Tentar substituir por candidatos de baixa coancestralidade
                for new_i, new_j, new_cost in top_candidates:
                    if (new_i, new_j) not in current_solution and new_cost < current_crossing_cost:
                        # Verificar se não cria sequência de animais
                        current_animals = set()
                        for ci, cj in current_solution:
                            if (ci, cj) != current_crossing:
                                current_animals.add(ci)
                                current_animals.add(cj)
                        
                        # Aceitar substituição se não cria muita sobreposição
                        if new_i not in current_animals or new_j not in current_animals:
                            new_solution = current_solution.copy()
                            new_solution[idx] = (new_i, new_j)
                            
                            total_new_cost = self.calculate_crossing_cost(new_solution)
                            
                            if total_new_cost < current_cost:
                                current_solution = new_solution
                                current_cost = total_new_cost
                                improved = True
                                break
                
                if improved:
                    break
        
        return current_solution
    
    def calculate_crossing_cost(self, solution: List[Tuple[int, int]]) -> float:
        """
        Calculate cost for a solution of crossing pairs.
        
        Args:
            solution: List of crossing pairs (row, col)
            
        Returns:
            Total coancestry cost
        """
        total_cost = 0.0
        for i, j in solution:
            total_cost += self.coancestry_matrix[i, j]
        return total_cost
    
    def optimize(self, num_crossings: int = None, progress_callback: Optional[Callable] = None) -> Tuple[List[Tuple[int, int]], float, List[float]]:
        """
        Run the GRASP optimization algorithm working on crossing matrix.
        
        Args:
            progress_callback: Optional callback function for progress updates
            num_crossings: Number of crossings to select in each solution
            
        Returns:
            Tuple of (best_solution, best_cost, iteration_costs)
        """
        self.iteration_costs = []
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_crossings = []
        
        no_improvement_count = 0
        max_no_improvement = min(50, self.max_iterations // 10)  # Convergência antecipada adaptativa
        
        for iteration in range(self.max_iterations):
            # Construction phase - selecionar cruzamentos da matriz
            solution = self.greedy_randomized_construction(num_crossings)
            
            # Local search phase - aplicar com menos frequência para iterações altas
            if self.max_iterations > 500 and iteration % 3 == 0:
                solution = self.local_search(solution)  # Busca local esporádica
            elif self.max_iterations <= 500:
                solution = self.local_search(solution)  # Busca local normal
            
            # Evaluate solution
            cost = self.calculate_crossing_cost(solution)
            
            # Update best solution
            if cost < self.best_cost:
                self.best_cost = cost
                # Diversificar solução antes de salvar
                diversified_solution = self.diversify_solution(solution)
                self.best_solution = diversified_solution.copy()
                self.best_crossings = self.convert_to_crossing_details(diversified_solution)
                no_improvement_count = 0  # Reset contador
            else:
                no_improvement_count += 1
            
            # Track progress
            self.iteration_costs.append(self.best_cost)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(iteration + 1, self.max_iterations)
            
            # Convergência antecipada se não há melhoria por muitas iterações
            if no_improvement_count >= max_no_improvement:
                break
        
        return self.best_solution, self.best_cost, self.iteration_costs
    
    def convert_to_crossing_details(self, solution: List[Tuple[int, int]]) -> List[dict]:
        """
        Convert solution to detailed crossing information.
        
        Args:
            solution: List of crossing pairs (row, col)
            
        Returns:
            List of dictionaries with crossing details
        """
        crossings = []
        for i, j in solution:
            crossings.append({
                'pair1_idx': i,
                'pair2_idx': j,
                'pair1_name': self.pair_names[i],
                'pair2_name': self.pair_names[j],
                'coancestry': self.coancestry_matrix[i, j]
            })
        
        # Ordenar por coancestralidade (menor primeiro)
        crossings.sort(key=lambda x: x['coancestry'])
        return crossings
    
    def diversify_solution(self, solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Diversifica a solução para evitar usar animais em sequência.
        
        Args:
            solution: Solução atual
            
        Returns:
            Solução diversificada
        """
        if len(solution) <= 1:
            return solution
            
        diversified = []
        used_animals = set()
        
        # Primeira passagem: incluir cruzamentos sem animais repetidos
        for crossing in solution:
            i, j = crossing
            if i not in used_animals and j not in used_animals:
                diversified.append(crossing)
                used_animals.add(i)
                used_animals.add(j)
        
        # Segunda passagem: incluir cruzamentos restantes com menor sobreposição
        for crossing in solution:
            if crossing not in diversified:
                i, j = crossing
                overlap = (i in used_animals) + (j in used_animals)
                if overlap <= 1:  # Permitir sobreposição de até 1 animal
                    diversified.append(crossing)
                    used_animals.add(i)
                    used_animals.add(j)
        
        # Se ainda não temos cruzamentos suficientes, incluir os restantes
        if len(diversified) < len(solution) // 2:
            for crossing in solution:
                if crossing not in diversified:
                    diversified.append(crossing)
                    if len(diversified) >= len(solution):
                        break
        
        return diversified
    
    def get_best_crossings_matrix(self) -> np.ndarray:
        """
        Create a matrix highlighting the best crossings found.
        
        Returns:
            Binary matrix where 1 indicates selected crossings
        """
        matrix = np.zeros_like(self.coancestry_matrix)
        
        if self.best_solution:
            for i, j in self.best_solution:
                matrix[i, j] = 1
                matrix[j, i] = 1  # Simetria
        
        return matrix
    
    def get_all_crossings_ranked(self) -> List[dict]:
        """
        Get all possible crossings ranked by coancestry coefficient.
        
        Returns:
            List of all crossings sorted by coancestry (best first)
        """
        all_crossings = []
        
        for i in range(self.matrix_size):
            for j in range(i + 1, self.matrix_size):
                all_crossings.append({
                    'pair1_idx': i,
                    'pair2_idx': j,
                    'pair1_name': self.pair_names[i],
                    'pair2_name': self.pair_names[j],
                    'coancestry': self.coancestry_matrix[i, j],
                    'selected': (i, j) in (self.best_solution or [])
                })
        
        # Ordenar por coancestralidade (menor primeiro)
        all_crossings.sort(key=lambda x: x['coancestry'])
        return all_crossings
    
    def get_solution_statistics(self, solution: List[int]) -> dict:
        """
        Get statistics about a solution.
        
        Args:
            solution: Solution to analyze
            
        Returns:
            Dictionary with solution statistics
        """
        male_usage = {}
        for male_idx in solution:
            male_usage[male_idx] = male_usage.get(male_idx, 0) + 1
        
        coancestry_values = [self.coancestry_matrix[i, solution[i]] for i in range(self.num_females)]
        
        return {
            'total_cost': self.calculate_total_cost(solution),
            'male_usage': male_usage,
            'avg_coancestry': np.mean(coancestry_values),
            'max_coancestry': np.max(coancestry_values),
            'min_coancestry': np.min(coancestry_values),
            'num_unique_males': len(set(solution))
        }
