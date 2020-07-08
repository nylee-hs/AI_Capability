import pandas as pd

job_title = ['a', 'b', 'c']
temp = [1,2,3,4]
temp1 = [5,6,7,8]
temp3 = [9,10, 11, 12]
# df = pd.DataFrame({job_title[0] : temp})
df = pd.DataFrame()
df.loc[:, job_title[1]] = pd.Series(temp1, index=df.index)
df.loc[:, job_title[2]] = pd.Series(temp3, index=df.index)
print(df)




