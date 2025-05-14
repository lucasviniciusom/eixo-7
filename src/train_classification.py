import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import json
import matplotlib.pyplot as plt

# Carregar os dados padronizados
input_file = 'outputs/binance_data_scaled.csv'
df_scaled = pd.read_csv(input_file, index_col='time', parse_dates=True)

log_file = 'outputs/modeling_log.txt'
try:
    with open(log_file, 'r') as f:
        modeling_log = f.read().splitlines()
except FileNotFoundError:
    modeling_log = []

modeling_log.append('\n--- Modelagem: Regressão Logística com TimeSeriesSplit (Previsão de Direção) ---')

# Criar variável alvo binária
df_scaled['price_change'] = df_scaled['close'].diff()
df_scaled['target_direction'] = np.where(df_scaled['price_change'].shift(-1) > 0, 1, 0)
df_scaled.dropna(subset=['target_direction'], inplace=True)
df_scaled.drop(columns=['price_change'], inplace=True)

# Separar features e alvo
if 'target_close' in df_scaled.columns:
    X = df_scaled.drop(['target_direction', 'target_close'], axis=1)
else:
    X = df_scaled.drop('target_direction', axis=1)
print(X.head())    

y = df_scaled['target_direction']
print(y.head()) 
exit()
modeling_log.append("- Variável Alvo (y): 'target_direction'")
modeling_log.append(f"- Features (X): {len(X.columns)} colunas")

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
accuracies, precisions, recalls, f1s = [], [], [], []
all_y_test, all_y_pred = [], []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    modeling_log.append(f"- Fold {fold+1}: Acurácia={acc:.4f}, Precisão={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

# Avaliação final
avg_metrics = {
    'accuracy': np.mean(accuracies),
    'precision': np.mean(precisions),
    'recall': np.mean(recalls),
    'f1_score': np.mean(f1s)
}
report = classification_report(all_y_test, all_y_pred)
modeling_log.append("\n--- Média dos Folds ---")
for k, v in avg_metrics.items():
    modeling_log.append(f"- {k.capitalize()}: {v:.4f}")
modeling_log.append("Relatório de Classificação Geral:")
modeling_log.extend([f"  {line}" for line in report.splitlines()])

# Matriz de confusão
cm = confusion_matrix(all_y_test, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - TimeSeriesSplit")
plt.tight_layout()
plt.savefig('outputs/confusion_matrix_logistic.png')
plt.show()
modeling_log.append("- Matriz de confusão salva em 'outputs/confusion_matrix_logistic.png'.")

# Salvar resultados
results = {
    'model_type': 'Logistic Regression (TimeSeriesSplit)',
    'target': 'target_direction',
    'metrics': avg_metrics,
    'classification_report': report
}
with open('outputs/classification_results.json', 'w') as f:
    json.dump(results, f, indent=4)

with open(log_file, 'w') as f:
    f.write('\n'.join(modeling_log))

print("Modelo treinado com TimeSeriesSplit. Resultados e log salvos com sucesso.")
