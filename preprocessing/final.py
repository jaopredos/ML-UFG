import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def feature_engineering(df):
    df_feat = df.copy()
    
    # Tratamento básico de nulos para garantir que as funções funcionem
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for c in num_cols:
        if c in df_feat.columns: df_feat[c] = df_feat[c].fillna(0)
    
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin']
    for c in cat_cols:
        if c in df_feat.columns: df_feat[c] = df_feat[c].fillna(df_feat[c].mode()[0] if not df_feat[c].mode().empty else 'Unknown')

    # Engenharia de features
    if 'Cabin' in df_feat.columns:
        df_feat[['Deck', 'Num', 'Side']] = df_feat['Cabin'].str.split('/', expand=True)
        df_feat['Num'] = pd.to_numeric(df_feat['Num'], errors='coerce').fillna(0)
    
    # Recriar features derivadas
    cols_gastos = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df_feat['TotalSpend'] = df_feat[cols_gastos].sum(axis=1)
    
    # Log Transform
    cols_to_log = cols_gastos + ['TotalSpend', 'Num']
    for col in cols_to_log:
        if col in df_feat.columns:
            df_feat[col] = np.log1p(df_feat[col])
            
    return df_feat

# Preprocessor
print("1. Recarregando dados de Treino para ajustar as colunas...")
df_train = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\train_cleaned.csv') # Seu arquivo de treino original
df_train_proc = feature_engineering(df_train)

# Definir colunas para o modelo
cols_drop = ['PassengerId', 'Name', 'Cabin', 'Group', 'Surname', 'Transported']
features = [c for c in df_train_proc.columns if c not in cols_drop]

# Separar tipos
X_train_raw = df_train_proc[features]
num_cols = [c for c in X_train_raw.columns if X_train_raw[c].dtype in ['int64', 'float64', 'int32', 'float32']]
cat_cols = [c for c in X_train_raw.columns if X_train_raw[c].dtype == 'object' or X_train_raw[c].dtype == 'bool']

# Ajustar o preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    verbose_feature_names_out=False
)
preprocessor.fit(X_train_raw) # O Preprocessor aprende aqui (26 colunas)

# Processar dados de teste
print("2. Processando dados de Teste...")
df_test = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\test_cleaned.csv')
ids_kaggle = df_test['PassengerId'].copy()

# Engenharia no Teste
df_test_proc = feature_engineering(df_test)

# Garantir que o teste tenha apenas as colunas esperadas
X_test_raw = df_test_proc[features] # Seleciona as mesmas colunas do treino

X_test_final = preprocessor.transform(X_test_raw)

print(f"Shape esperado pelo modelo: (N, 26)")
print(f"Shape gerado agora: {X_test_final.shape}") 

# Predição
if X_test_final.shape[1] == 26:
    print("3. Carregando modelo e prevendo...")
    model_path = r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\model\melhor_modelo_spaceship.keras'
    model = tf.keras.models.load_model(model_path)
    
    previsoes = model.predict(X_test_final)
    previsoes_bool = (previsoes >= 0.4).astype(bool).flatten()
    
    submission = pd.DataFrame({'PassengerId': ids_kaggle, 'Transported': previsoes_bool})
    submission.to_csv('submission_corrected2.csv', index=False)
    print("Sucesso! Arquivo 'submission_corrected.csv' gerado.")
else:
    print("ERRO CRÍTICO: O número de colunas ainda está diferente. Verifique se o arquivo train.csv é o mesmo usado no treinamento.")