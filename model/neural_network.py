import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import os

# --- 1. CARREGAR DADOS (Separados) ---
# Ajuste os caminhos conforme necessário. Estou usando caminhos relativos para facilitar.
# Se estiver rodando localmente, pode manter seus caminhos absolutos (C:\Users...)
try:
    X_train = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\X_train.csv').values
    y_train = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\y_train.csv').values.ravel() # ravel transforma em vetor 1D
    X_val = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\X_val.csv').values
    y_val = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\y_val.csv').values.ravel()
except FileNotFoundError:
    # Fallback para os nomes antigos caso não tenha renomeado
    X_train = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\X_train.csv').values
    y_train = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\y_train.csv').values.ravel() # ravel transforma em vetor 1D
    X_val = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\X_val.csv').values
    y_val = pd.read_csv(r'C:\Users\João Pedro\Documents\UFG\AMS\AS2\data\y_val.csv').values.ravel()

print(f"Dados de Treino Carregados: {X_train.shape}")
print(f"Dados de Validação (Hold-out) Carregados: {X_val.shape}")

# --- 2. FUNÇÃO CONSTRUTORA DO MODELO ---
def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 3. CROSS-VALIDATION (Apenas nos dados de Treino - 80%) ---
print('\n' + '='*40)
print('FASE 1: Cross-Validation (5 Folds) nos dados de Treino')
print('='*40)

FOLDS = 5
kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
cv_scores = []
fold_no = 1

# Note que usamos apenas X_train e y_train aqui
for train_idx, test_idx in kfold.split(X_train, y_train):
    # Separar sub-treino e sub-teste (dentro dos 80%)
    X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
    y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]
    
    model = build_model(X_train.shape[1])
    
    # Early Stopping para o CV
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, monitor='val_loss'
    )
    
    model.fit(
        X_fold_train, y_fold_train,
        validation_data=(X_fold_test, y_fold_test),
        epochs=50, batch_size=32,
        callbacks=[early_stopping], verbose=0
    )
    
    scores = model.evaluate(X_fold_test, y_fold_test, verbose=0)
    print(f'Fold {fold_no}: {scores[1]*100:.2f}%')
    cv_scores.append(scores[1])
    fold_no += 1

print(f'\nMédia CV (Estabilidade do Modelo): {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)')

# --- 4. TREINAMENTO FINAL (Treino Completo vs Validação Final) ---
print('\n' + '='*40)
print('FASE 2: Treinamento Final (Treino Completo -> Validação Final)')
print('='*40)

# Criar modelo final
final_model = build_model(X_train.shape[1])

# Early Stopping monitorando a validação final (os 20% que guardamos)
final_es = keras.callbacks.EarlyStopping(
    patience=15, restore_best_weights=True, monitor='val_loss'
)

# Checkpoint para salvar o MELHOR modelo durante o treino
checkpoint = keras.callbacks.ModelCheckpoint(
    'melhor_modelo_spaceship.keras', # Nome do arquivo
    save_best_only=True,             # Salva apenas se melhorar
    monitor='val_accuracy',          # Monitora acurácia na validação
    mode='max',
    verbose=1
)

history = final_model.fit(
    X_train, y_train,                # Treina nos 80% totais
    validation_data=(X_val, y_val),  # Valida nos 20% guardados (Hold-out)
    epochs=100, 
    batch_size=32,
    callbacks=[final_es, checkpoint],
    verbose=1
)

# --- 5. AVALIAÇÃO FINAL ---
print('\n--- Resultados Finais no Dataset de Validação ---')
final_scores = final_model.evaluate(X_val, y_val, verbose=0)
print(f'Acurácia Final: {final_scores[1]*100:.2f}%')
print(f'Loss Final:     {final_scores[0]:.4f}')

# O modelo já foi salvo pelo Checkpoint como 'melhor_modelo_spaceship.keras'
print("\nModelo salvo com sucesso como 'melhor_modelo_spaceship.keras'")