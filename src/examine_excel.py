import pandas as pd

# Carregar o arquivo Excel
file_path = '/home/ubuntu/upload/binance_data_6h.xlsx'
df = pd.read_excel(file_path)

# Exibir as primeiras linhas
print('Primeiras 5 linhas do DataFrame:')
print(df.head())

# Exibir informações sobre o DataFrame (tipos de dados, valores não nulos)
print('\nInformações do DataFrame:')
df.info()

# Exibir estatísticas descritivas
print('\nEstatísticas Descritivas:')
print(df.describe())

