import pandas as pd
#
# df = pd.read_csv('data/doc2vec_test_data/0831/data/0831_integeration.csv', encoding='utf-8')
# print(df)


texts = []
contents_list = []
with open('data/doc2vec_test_data/0831/data/intermediate.txt', encoding='utf-8', mode='r') as f:
    line = f.readline()
    while line !='':
        texts.append(line)
        line = f.readline()


texts = '. '.join(texts)
print(texts[:10], texts[-10:])

contents_list.append(texts)

texts1 = []
with open('data/doc2vec_test_data/0831/data/senior.txt', encoding='utf-8', mode='r') as f:
    line = f.readline()
    while line !='':
        texts1.append(line)
        line = f.readline()


texts1 = '. '.join(texts1)
contents_list.append(texts1)
print(texts1[:10], texts1[-10:])
texts2 = []
with open('data/doc2vec_test_data/0831/data/manager.txt', encoding='utf-8', mode='r') as f:
    line = f.readline()
    while line !='':
        texts2.append(line)
        line = f.readline()


texts2 = '. '.join(texts2)
contents_list.append(texts2)
print(texts2[:10], texts2[-10:])

texts3 = []
with open('data/doc2vec_test_data/0831/data/junior.txt', encoding='utf-8', mode='r') as f:
    line = f.readline()
    while line !='':
        texts3.append(line)
        line = f.readline()


texts3 = '. '.join(texts3)
contents_list.append(texts3)
print(texts3[:10], texts3[-10:])

title_list =['intermediate', 'senior', 'manager', 'junior']
ids = ['0', '1','2', '3']

df = pd.DataFrame({'id': ids, 'job_title': title_list, 'job_description': contents_list})
print(df)
df.to_csv('data/doc2vec_test_data/0831/data/0831_integration.csv', mode='w', encoding='utf-8')

