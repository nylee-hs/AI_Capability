import pandas as pd
num = ['10', '20', '30', '40']
co=[-10, -20, -30, -50]
df = pd.DataFrame({'num': num, 'value': co})
df['delta'] = df['value'].diff() / df['value'][1:]
print(df)
print(df['value'].pct_change(1))
find = df['delta'] == df['delta'].max()
df_find = df[find]


# if co[0] > df_find['value'].tolist()[0]:
#     print("yes")
# else:
#     print("no")




# print(df['value'][0], df['delta'].max()[0])


