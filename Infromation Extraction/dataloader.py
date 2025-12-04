import pandas as pd

file = "FINER.parquet"

data = pd.read_parquet(file)

for i in data['label']:
    print(i)