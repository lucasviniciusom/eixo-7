import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carregar os dados pré-processados
input_file = '/home/ubuntu/binance_data_preprocessed.csv'
df = pd.read_csv(input_file, index_col='time', parse_dates=True)

# Carregar o log existente
log_file = '/home/ubuntu/preprocessing_log.txt'
with open(log_file, 'r') as f:
    preprocessing_log = f.read().splitlines()

preprocessing_log.append('\n--- Limpeza e Transformação Adicional ---')

# 1. Análise de Outliers (Método IQR)
# Selecionar apenas colunas numéricas
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

outlier_summary = []
outlier_summary.append('Análise de Outliers (Método IQR):')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Contar outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = outliers.shape[0]
    
    if outlier_count > 0:
        summary_line = f"- Coluna '{col}': {outlier_count} outliers detectados (abaixo de {lower_bound:.2f} ou acima de {upper_bound:.2f})"
        outlier_summary.append(summary_line)

print('\n'.join(outlier_summary))
preprocessing_log.append('\n'.join(outlier_summary))
preprocessing_log.append("- Nota: Outliers foram identificados, mas não removidos ou modificados nesta etapa, pois podem conter informações valiosas em dados financeiros.")

# 2. Padronização (Standardization)
# Aplicar StandardScaler às colunas numéricas
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

preprocessing_log.append("\n- Padronização (Z-score) aplicada a todas as colunas numéricas.")
print("\nPadronização (Z-score) aplicada.")

# Salvar o DataFrame padronizado
output_scaled_file = '/home/ubuntu/binance_data_scaled.csv'
df_scaled.to_csv(output_scaled_file)
preprocessing_log.append(f"- Dados padronizados salvos em '{output_scaled_file}'.")
print(f"DataFrame padronizado salvo em {output_scaled_file}")

# Salvar o log de pré-processamento atualizado
with open(log_file, 'w') as f:
    f.write('\n'.join(preprocessing_log))
print(f"Log de pré-processamento atualizado em {log_file}")

print("\nLimpeza e transformação adicionais concluídas.")

