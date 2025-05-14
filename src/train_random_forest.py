import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Carregar os dados padronizados
df_scaled = pd.read_csv('outputs/binance_data_scaled.csv', index_col='time', parse_dates=True)

# Criar variável alvo binária
df_scaled['price_change'] = df_scaled['close'].diff()
df_scaled['target_direction'] = np.where(df_scaled['price_change'].shift(-1) > 0, 1, 0)
df_scaled.dropna(subset=['target_direction'], inplace=True)
df_scaled.drop(columns=['price_change'], inplace=True)

# Separar features e target
X = df_scaled.drop(columns=['target_direction', 'target_close'], errors='ignore')
y = df_scaled['target_direction']

# Log
modeling_log = []
modeling_log.append('--- Modelagem: Random Forest com TimeSeriesSplit ---')
modeling_log.append(f"- Total de observações: {len(df_scaled)}")
modeling_log.append(f"- Features: {len(X.columns)} colunas")

# Configurar TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
results_by_fold = []

# Para armazenar previsões do último fold 
final_y_test = None
final_y_pred = None

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_by_fold.append({
        'fold': fold + 1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    })

    modeling_log.append(f"- Fold {fold+1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # Guardar o último fold para visualização
    final_y_test = y_test
    final_y_pred = y_pred

# Calcular médias
metrics_avg = {
    'accuracy': np.mean([r['accuracy'] for r in results_by_fold]),
    'precision': np.mean([r['precision'] for r in results_by_fold]),
    'recall': np.mean([r['recall'] for r in results_by_fold]),
    'f1_score': np.mean([r['f1_score'] for r in results_by_fold])
}

modeling_log.append("\n- Média das métricas nos folds:")
for k, v in metrics_avg.items():
    modeling_log.append(f"  - {k}: {v:.4f}")

# Salvar JSON com resultados
results = {
    'model_type': 'Random Forest Classifier (TimeSeriesSplit)',
    'target': 'target_direction',
    'metrics_average': metrics_avg,
    'folds': results_by_fold
}
with open('outputs/random_forest_timeseries_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Salvar log
with open('outputs/modeling_log.txt', 'a') as f:
    f.write('\n'.join(modeling_log) + '\n')

# === GRÁFICOS ===

# 1. Gráfico de métricas por fold
folds = [r['fold'] for r in results_by_fold]
plt.figure(figsize=(10,6))
plt.plot(folds, [r['accuracy'] for r in results_by_fold], marker='o', label='Accuracy')
plt.plot(folds, [r['precision'] for r in results_by_fold], marker='o', label='Precision')
plt.plot(folds, [r['recall'] for r in results_by_fold], marker='o', label='Recall')
plt.plot(folds, [r['f1_score'] for r in results_by_fold], marker='o', label='F1 Score')
plt.title("Métricas por Fold - Random Forest")
plt.xlabel("Fold")
plt.ylabel("Valor")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/random_forest_folds_metrics.png')
plt.show()

# 2. Matriz de confusão do último fold
cm = confusion_matrix(final_y_test, final_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Último Fold")
plt.tight_layout()
plt.savefig('outputs/random_forest_confusion_matrix.png')
plt.show()

print("Resultados e gráficos salvos com sucesso.")
