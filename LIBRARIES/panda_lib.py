import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Marks': [85, 92, 78]}
df = pd.DataFrame(data)
print(df.describe())
print("Average Marks:", df['Marks'].mean())
