import pandas as pd

# Carregar o arquivo Excel da pasta data/
df = pd.read_excel('../data/binance_data_6h.xlsx')

print(f'Formato original do DataFrame: {df.shape}')

# --- Documentação do Pré-processamento --- 
preprocessing_log = []
preprocessing_log.append('Etapas de Pré-processamento:')

# 1. Remover colunas inúteis
diff = (pd.to_datetime(df['close_time'], unit='ms') - df['time']).dt.total_seconds()
if diff.nunique() == 1 and diff.iloc[0] == 21599.999:
    print("'close_time' é redundante e será removida.")
    df = df.drop(columns=['close_time', 'ignore'])
    preprocessing_log.append("- Colunas 'close_time' e 'ignore' removidas por redundância/inutilidade.")
else:
    print("'close_time' não é diretamente redundante ou a diferença não é constante. Removendo apenas 'ignore'.")
    df = df.drop(columns=['ignore'])
    preprocessing_log.append("- Coluna 'ignore' removida por inutilidade.")

# 2. Lidar com valores ausentes (NaN)
print('\nContagem de NaNs antes do tratamento:')
print(df.isnull().sum())

if 'Parabolic_SAR' in df.columns:
    df = df.drop(columns=['Parabolic_SAR'])
    preprocessing_log.append("- Coluna 'Parabolic_SAR' removida devido à alta porcentagem de valores ausentes (~50%).")
    print("\nColuna 'Parabolic_SAR' removida.")

initial_rows = df.shape[0]
df.dropna(inplace=True)
final_rows = df.shape[0]
rows_dropped = initial_rows - final_rows
preprocessing_log.append(f"- Linhas com valores ausentes (NaN) removidas. Total de linhas removidas: {rows_dropped}.")
print(f'\n{rows_dropped} linhas removidas devido a valores NaN.')

print('\nContagem de NaNs após tratamento:')
print(df.isnull().sum().sum())

# 3. Definir 'time' como índice
df.set_index('time', inplace=True)
preprocessing_log.append("- Coluna 'time' definida como índice do DataFrame.")
print("\nColuna 'time' definida como índice.")

print(f'\nFormato final do DataFrame: {df.shape}')

# 4. Salvar o DataFrame pré-processado na pasta outputs/
output_file = '../outputs/binance_data_preprocessed.csv'
df.to_csv(output_file)
preprocessing_log.append(f"- Dados pré-processados salvos em '{output_file}'.")
print(f"\nDataFrame pré-processado salvo em {output_file}")

# 5. Salvar o log de pré-processamento na pasta outputs/
log_file = '../outputs/preprocessing_log.txt'
with open(log_file, 'w') as f:
    f.write('\n'.join(preprocessing_log))
print(f"Log de pré-processamento salvo em {log_file}")

# 6. Imprimir log no terminal
print('\n--- Log de Pré-processamento ---')
print('\n'.join(preprocessing_log))
