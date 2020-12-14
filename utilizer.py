import pandas as pd
#
# df = pd.read_csv('data/doc2vec_test_data/0831/data/0831_integeration.csv', encoding='utf-8')
# print(df)

class Integration:
    def getIntegratedData(self):

        texts = []
        contents_list = []
        with open('analysis/test/0907/data/manager.txt', encoding='utf-8', mode='r') as f:
            line = f.readline()
            # print(line)
            while line !='':
                line = line.rstrip('\n')
                line = line.replace('"', '')
                line = line.replace(',', ' ')
                texts.append(line)
                line = f.readline()
                # print(line)

        texts = '. '.join(texts)
        print(texts)


        contents_list.append(texts)

        texts1 = []
        with open('analysis/test/0907/data/senior.txt', encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line !='':
                line = line.rstrip('\n')
                line = line.replace('"', '')
                line = line.replace(',', ' ')
                texts1.append(line)
                line = f.readline()


        texts1 = '. '.join(texts1)
        contents_list.append(texts1)

        texts2 = []
        with open('analysis/test/0907/data/intermediate.txt', encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line !='':
                line = line.rstrip('\n')
                line = line.replace('"', '')
                line = line.replace(',', ' ')
                texts2.append(line)
                line = f.readline()


        texts2 = '. '.join(texts2)
        contents_list.append(texts2)


        texts3 = []
        with open('analysis/test/0907/data/junior.txt', encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line !='':
                line = line.rstrip('\n')
                line = line.replace('"', '')
                line = line.replace(',', ' ')
                line = line + '. '
                texts3.append(line)
                line = f.readline()


        texts3 = '. '.join(texts3)
        contents_list.append(texts3)

        title_list =['manager', 'senior', 'intermediate', 'junior']
        ids = ['0', '1','2', '3']

        df = pd.DataFrame({'id': ids, 'job_title': title_list, 'job_description': contents_list})

        return df

