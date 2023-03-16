# load csv
import pandas as pd
df = pd.read_csv('/home/tmp00050/jiasheng/ProgressivePrompts/datasets/src/data/amazon/test.csv', header=None)
df = df.rename(columns={0: "label", 1: "title", 2: "content"})
df['content'] = 'aaa'
print(df)
