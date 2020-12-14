from doc2vec_input2 import Doc2VecInput, Configuration
import pickle
import os.path
import pytagcloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from matplotlib import pyplot as plt

class Test:
    def __init__(self, config):
        self.data_path = config.data_path
        self.data_file_name = config.data_file_name

    def get_word_count(self, data_lemmatized):
        sline = [' '.join(line) for line in data_lemmatized]
        # print(len(sline))
        # 단어 빈도 계산
        word_list = []
        for line in sline:
            for word in line.split():
                word_list.append(word)
        word_count = {}
        for word in word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        word_count_list = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        key_list = []
        value_list = []
        for item in word_count_list:
            key_list.append(item[0])
            value_list.append(item[1])
        new_word_count_dict = dict(zip(key_list[:50], value_list[:50]))  ## 워드 클라우드는 50개까지 보여주기
        self.get_word_cloud(new_word_count_dict)
        self.get_word_graph((new_word_count_dict))
        df = pd.DataFrame({'Terms': key_list, 'Frequency': value_list})
        df.to_csv(self.data_path + self.data_file_name + '_frequency.csv', mode='w', encoding='utf-8')  ## 빈도 데이터 저장

        ## tfidf 계산
        cvectorizer = CountVectorizer(min_df=0)
        dtm = cvectorizer.fit_transform(sline)
        dtm_features = cvectorizer.get_feature_names()
        temp = dtm.toarray()
        # print(len(temp))
        # print(len(temp[0]))
        # print(len(dtm_features))
        cvec_df = pd.DataFrame(temp, columns=dtm_features)
        cvec_df.to_csv(self.data_path + self.data_file_name + '_dtm.csv', mode='w', encoding='utf-8')  ## dtm 데이터 저장

        tfidf = TfidfVectorizer(min_df=0, sublinear_tf=True, max_df=0.8).fit(sline)
        tfidf_sp = tfidf.transform(sline)
        tfidf_dict = tfidf.get_feature_names()
        data_array = tfidf_sp.toarray()
        df = pd.DataFrame(data_array, columns=tfidf_dict)
        # sum_array = df.sum(axis=0)
        df.loc['TF_IDF_SUM', :] = df.sum()
        df.to_csv(self.data_path + self.data_file_name + '_tf_idf.csv', mode='w', encoding='utf-8')  ## tf-idf 데이터 저장


    def get_word_cloud(self, word_count_dict):
        taglist = pytagcloud.make_tags(word_count_dict.items(), maxsize=100)
        pytagcloud.create_tag_image(taglist, self.data_path + self.data_file_name + '_word_cloud.jpg', size=(1200, 800), rectangular=False)


    def get_word_graph(self, word_count_dict):
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.grid(True)


        Sorted_Dict_Values = sorted(word_count_dict.values(), reverse=True)
        Sorted_Dict_Keys = sorted(word_count_dict, key=word_count_dict.get, reverse=True)

        plt.bar(range(len(word_count_dict)), Sorted_Dict_Values, align='center')
        plt.xticks(range(len(word_count_dict)), list(Sorted_Dict_Keys), rotation='90', fontsize=5)
        # plt.figure(num=None, figsize=(20, 10), dpi=80)
        plt.draw()
        plt.savefig(self.data_path+self.data_file_name+'_frequecy.png', dpi=300, bbox_inches='tight')
        plt.show()

config = Configuration()
test = Test(config)
with open(test.data_path+test.data_file_name+'.corpus', 'rb') as f:
    data = pickle.load(f)
test.get_word_count(data)
