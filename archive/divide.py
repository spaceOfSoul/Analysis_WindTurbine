import pandas as pd

file_path = 'T1.csv'
data = pd.read_csv('T1.csv')

total_rows = len(data)

train_end = int(total_rows * 0.6)
val_end = int(total_rows * 0.8)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
