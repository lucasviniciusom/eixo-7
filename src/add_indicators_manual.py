import pandas as pd
import numpy as np

# Função para calcular ATR manualmente
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_prev_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

# Caminhos corrigidos
input_file = '../outputs/binance_data_preprocessed.csv'
log_file = '../outputs/preprocessing_log.txt'
output_file = '../outputs/binance_data_preprocessed.csv'

# Carregar os dados pré-processados
df = pd.read_csv(input_file, index_col='time', parse_dates=True)

# Carregar o log existente
with open(log_file, 'r') as f:
    preprocessing_log = f.read().splitlines()

# Evitar duplicação da seção
if '--- Adição de Indicadores de Trading (Cálculo Manual) ---' in preprocessing_log[-1]:
    try:
        start_index = preprocessing_log.index('--- Adição de Indicadores de Trading (Cálculo Manual) ---')
        preprocessing_log = preprocessing_log[:start_index]
    except ValueError:
        pass

preprocessing_log.append('--- Adição de Indicadores de Trading (Cálculo Manual) ---')

# Verificar indicadores já presentes
present_indicators = [col for col in df.columns if col in ['SMA_10', 'EMA_10', 'MACD', 'BB_High', 'BB_Middle', 'BB_Low', 'ADX']]
preprocessing_log.append(f"- Indicadores já presentes no dataset: {', '.join(present_indicators)}")
print(f"Indicadores já presentes: {', '.join(present_indicators)}")

# Calcular ATR manualmente
atr_period = 14
df['ATR_14'] = calculate_atr(df, period=atr_period)

# Verificar e remover NaNs
print(f"\nValores NaN na nova coluna ATR_14: {df['ATR_14'].isnull().sum()}")
initial_rows = df.shape[0]
df.dropna(subset=['ATR_14'], inplace=True)
final_rows = df.shape[0]
rows_dropped = initial_rows - final_rows
preprocessing_log.append(f"- Indicador ATR (Average True Range, período {atr_period}) calculado manualmente e adicionado como 'ATR_14'.")
preprocessing_log.append(f"- {rows_dropped} linhas removidas devido a NaNs gerados pelo cálculo do ATR.")
print(f"{rows_dropped} linhas removidas devido a NaNs do ATR.")

print(f'Formato do DataFrame após adicionar ATR: {df.shape}')

# Salvar o DataFrame atualizado
df.to_csv(output_file)
preprocessing_log.append(f"- DataFrame atualizado com ATR salvo em '{output_file}'.")
print(f"DataFrame atualizado com ATR salvo em {output_file}")

# Atualizar o log
with open(log_file, 'w') as f:
    f.write('\n'.join(preprocessing_log))
print(f"Log de pré-processamento atualizado em {log_file}")
