import pandas as pd

idx = pd.read_csv('data/node_year_test.csv', index_col=0)
idx['0'] = 0
print(idx)
idx.to_csv('data/node_label_test.csv')
