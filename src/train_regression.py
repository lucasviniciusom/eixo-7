import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import matplotlib.pyplot as plt

# Carregar dados
df_scaled = pd.read_csv('outputs/binance_data_scaled.csv', index_col='time', parse_dates=True)

modeling_log = []
modeling_log.append('--- Modelagem: Regressão Linear com TimeSeriesSplit ---')

# Criar variável alvo (prever o close do próximo período)
df_scaled['target_close'] = df_scaled['close'].shift(-1)
df_scaled.dropna(subset=['target_close'], inplace=True)

X = df_scaled.drop('target_close', axis=1)
y = df_scaled['target_close']

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
rmse_list = []
r2_list = []

# Para gráficos e análise
all_real = []
all_pred = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    r2_list.append(r2)

    modeling_log.append(f"- Fold {fold+1}: RMSE={rmse:.4f}, R²={r2:.4f}")

    all_real.extend(y_test.values)
    all_pred.extend(y_pred)

# Resultados médios
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)

results = {
    'model_type': 'Linear Regression (TimeSeriesSplit)',
    'target': 'target_close',
    'metrics': {
        'average_rmse': avg_rmse,
        'average_r2': avg_r2,
        'folds': [{'rmse': r, 'r2': s} for r, s in zip(rmse_list, r2_list)]
    }
}

with open('outputs/regression_results.json', 'w') as f:
    json.dump(results, f, indent=4)

modeling_log.append(f"- Média dos Folds: RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")
modeling_log.append("- Resultados salvos em 'regression_results.json'.")

with open('outputs/modeling_log.txt', 'a') as f:
    f.write('\n'.join(modeling_log) + '\n')

# Gráfico: Real vs Previsto (todos os folds)
plt.figure(figsize=(12, 6))
plt.plot(all_real, label='Real', alpha=0.7)
plt.plot(all_pred, label='Previsto', alpha=0.7)
plt.title('Comparação de Todos os Folds: Real vs. Previsto')
plt.xlabel('Observações no tempo (concatenação dos folds)')
plt.ylabel('Preço padronizado')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/regression_all_folds_comparison.png')
plt.show()

# --- Gráfico de Acertos e Erros ---
erro_absoluto = np.abs(np.array(all_real) - np.array(all_pred))
limite = 0.03 
acertos = np.sum(erro_absoluto <= limite)
erros = np.sum(erro_absoluto > limite)

# Exibir gráfico de barras
plt.figure(figsize=(6, 5))
plt.bar(['Acertos', 'Erros'], [acertos, erros], color=['green', 'red'])
plt.title(f'Nº de Acertos vs Erros (Margem ±{limite})')
plt.ylabel('Quantidade')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('outputs/regression_accuracy_bar.png')
plt.show()
