import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re

class DataProcessor:
    """
    Classe para processar dados de acasalamento animal e criar matrizes de coancestralidade.
    """
    
    def __init__(self, uploaded_file):
        """
        Inicializa o processador de dados com arquivo CSV carregado.
        
        Args:
            uploaded_file: Objeto de arquivo carregado do Streamlit
        """
        self.df = pd.read_csv(uploaded_file)
        self.validate_data()
        self.process_data()
    
    def validate_data(self):
        """
        Valida se o arquivo CSV possui as colunas necessárias.
        """
        required_columns = ['Animal_1', 'Animal_2', 'Coef']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Verificar valores nulos
        if self.df[required_columns].isnull().any().any():
            raise ValueError("Dataset contains null values in required columns")
        
        # Verificar valores de coeficiente
        if not pd.api.types.is_numeric_dtype(self.df['Coef']):
            raise ValueError("Coef column must contain numeric values")
    
    def extract_animals_from_pair(self, pair_string: str) -> Tuple[str, str]:
        """
        Extrai animais individuais de uma string de par como 'pai1_pai2'.
        
        Args:
            pair_string: String no formato 'animal1_animal2'
            
        Returns:
            Tupla de (animal1, animal2)
        """
        parts = pair_string.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair_string}")
        return parts[0], parts[1]
    
    def process_data(self):
        """
        Processa os dados para extrair animais e criar mapeamentos.
        """
        # Extrair todos os animais individuais dos pares
        all_animals = set()
        
        for _, row in self.df.iterrows():
            animal1_pair = row['Animal_1']
            animal2_pair = row['Animal_2']
            
            # Extrair animais do primeiro par
            a1, a2 = self.extract_animals_from_pair(animal1_pair)
            all_animals.update([a1, a2])
            
            # Extrair animais do segundo par
            a3, a4 = self.extract_animals_from_pair(animal2_pair)
            all_animals.update([a3, a4])
        
        # Ordenar animais para ordenação consistente
        all_animals = sorted(list(all_animals))
        
        # For this implementation, we'll assume the first animal in each pair is female
        # and the second is male. This is a simplification for the academic project.
        females = set()
        males = set()
        
        for _, row in self.df.iterrows():
            animal1_pair = row['Animal_1']
            animal2_pair = row['Animal_2']
            
            # Extract animals from pairs
            f1, m1 = self.extract_animals_from_pair(animal1_pair)
            f2, m2 = self.extract_animals_from_pair(animal2_pair)
            
            females.update([f1, f2])
            males.update([m1, m2])
        
        # Convert to sorted lists
        self.females = sorted(list(females))
        self.males = sorted(list(males))
        
        # Create mappings
        self.female_to_idx = {female: idx for idx, female in enumerate(self.females)}
        self.male_to_idx = {male: idx for idx, male in enumerate(self.males)}
        
        # Calculate statistics
        self.num_females = len(self.females)
        self.num_males = len(self.males)
        self.coef_mean = self.df['Coef'].mean()
        self.coef_max = self.df['Coef'].max()
        self.coef_min = self.df['Coef'].min()
        
        # Create coancestry matrix
        self.create_coancestry_matrix()
    
    def create_coancestry_matrix(self):
        """
        Create the coancestry matrix from the processed data.
        Matriz representa todos os cruzamentos possíveis entre pares do arquivo CSV.
        Versão otimizada para performance.
        """
        # Usar pandas para otimizar a criação da lista de pares únicos
        all_pairs = set(self.df['Animal_1'].unique()) | set(self.df['Animal_2'].unique())
        
        self.all_pairs = sorted(list(all_pairs))
        self.num_pairs = len(self.all_pairs)
        
        # Criar mapeamento de pares para índices
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.all_pairs)}
        
        # Inicializar matriz com zeros
        self.coancestry_matrix = np.zeros((self.num_pairs, self.num_pairs))
        
        # Preencher matriz usando indexação vetorizada do pandas
        df_indexed = self.df.copy()
        df_indexed['idx1'] = df_indexed['Animal_1'].map(self.pair_to_idx)
        df_indexed['idx2'] = df_indexed['Animal_2'].map(self.pair_to_idx)
        
        # Preencher valores na matriz usando indexação avançada
        idx1_values = df_indexed['idx1'].values
        idx2_values = df_indexed['idx2'].values
        coef_values = df_indexed['Coef'].values
        
        # Preencher simetricamente
        self.coancestry_matrix[idx1_values, idx2_values] = coef_values
        self.coancestry_matrix[idx2_values, idx1_values] = coef_values
        
        # Preencher diagonal principal com 1 (mesmo animal)
        np.fill_diagonal(self.coancestry_matrix, 1.0)
        
        # Para compatibilidade com o algoritmo GRASP, também criar matriz fêmeas x machos
        self.create_breeding_matrix()
    
    def create_breeding_matrix(self):
        """
        Cria uma matriz de breeding (fêmeas x machos) para o algoritmo GRASP.
        Usa a matriz de coancestralidade já criada para extrair os valores.
        """
        # Inicializar matriz breeding com zeros
        self.breeding_matrix = np.zeros((self.num_females, self.num_males))
        
        # Preencher matriz breeding usando a matriz de coancestralidade
        for f_idx, female in enumerate(self.females):
            for m_idx, male in enumerate(self.males):
                # Encontrar índices na matriz de coancestralidade completa
                female_pair_idx = self.pair_to_idx.get(female, -1)
                male_pair_idx = self.pair_to_idx.get(male, -1)
                
                if female_pair_idx != -1 and male_pair_idx != -1:
                    # Usar o valor da matriz de coancestralidade
                    self.breeding_matrix[f_idx, m_idx] = self.coancestry_matrix[female_pair_idx, male_pair_idx]
                else:
                    # Se não encontrar na matriz, usar 0 (sem parentesco conhecido)
                    self.breeding_matrix[f_idx, m_idx] = 0.0
    
    def get_animal_mapping(self) -> Dict[str, str]:
        """
        Get mapping from original animal IDs to new IDs (f1, f2, m1, m2, etc.).
        
        Returns:
            Dictionary mapping original IDs to new IDs
        """
        mapping = {}
        
        for i, female in enumerate(self.females):
            mapping[female] = f'f{i+1}'
        
        for i, male in enumerate(self.males):
            mapping[male] = f'm{i+1}'
        
        return mapping
    
    def get_breeding_pairs_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame with all possible breeding pairs and their coancestry values.
        
        Returns:
            DataFrame with columns: Female, Male, Female_Original, Male_Original, Coancestry
        """
        pairs = []
        
        for f_idx, female in enumerate(self.females):
            for m_idx, male in enumerate(self.males):
                pairs.append({
                    'Female': f'f{f_idx+1}',
                    'Male': f'm{m_idx+1}',
                    'Female_Original': female,
                    'Male_Original': male,
                    'Coancestry': self.coancestry_matrix[f_idx, m_idx]
                })
        
        return pd.DataFrame(pairs)
