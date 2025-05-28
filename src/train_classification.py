import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

input_file = 'outputs/binance_data_scaled.csv'
log_file = 'outputs/modeling_log.txt'
results_file = 'outputs/classification_results.json'
confusion_matrix_img = 'outputs/confusion_matrix_logistic.png'

# === CARREGAR DADOS ===
df = pd.read_csv(input_file, index_col='time', parse_dates=True)

# === LOG INICIAL ===
try:
    with open(log_file, 'r') as f:
        modeling_log = f.read().splitlines()
except FileNotFoundError:
    modeling_log = []

modeling_log.append('\n--- Modelagem: Regressão Logística com TimeSeriesSplit (Previsão de Direção) ---')

# === CRIAR VARIÁVEL ALVO ===
df['price_change'] = df['close'].diff()
df['target_direction'] = np.where(df['price_change'].shift(-1) > 0, 1, 0)
df.dropna(subset=['target_direction'], inplace=True)
df.drop(columns=['price_change'], inplace=True)

# === DEFINIR FEATURES E TARGET ===
X = df.drop(columns=['target_direction', 'target_close'], errors='ignore')
y = df['target_direction']

modeling_log.append("- Variável Alvo: 'target_direction'")
modeling_log.append(f"- Total de features: {X.shape[1]}")

# === TIME SERIES SPLIT ===
tscv = TimeSeriesSplit(n_splits=5)
accuracies, precisions, recalls, f1s = [], [], [], []
all_y_test, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

# === MÉTRICAS FINAIS ===
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

# === MATRIZ DE CONFUSÃO ===
cm = confusion_matrix(all_y_test, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Regressão Logística")
plt.tight_layout()
plt.savefig(confusion_matrix_img)
plt.show()

modeling_log.append(f"- Matriz de confusão salva em '{confusion_matrix_img}'.")

# === SALVAR RESULTADOS ===
results = {
    'model_type': 'Logistic Regression (TimeSeriesSplit)',
    'target': 'target_direction',
    'metrics': avg_metrics,
    'classification_report': report
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

with open(log_file, 'w') as f:
    f.write('\n'.join(modeling_log))

print("✅ Modelo treinado e avaliado com sucesso. Resultados salvos.")
