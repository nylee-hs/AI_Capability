from gensim.models import Doc2Vec
from datamanager import DataManager
import random
import pandas as pd
import numpy as np
import pickle
from visualize_utils import visualize_between_words, visualize_words
from tqdm import tqdm
import pytagcloud
import matplotlib
import matplotlib.pyplot as plt

class Doc2VecEvaluator:

    def __init__(self, model_fname, use_notebook=False):
        self.model = Doc2Vec.load(model_fname)
        self.doc2idx = {el:idx for idx, el in enumerate(self.model.docvecs.doctags.keys())}
        self.use_notebook = use_notebook
        dm = DataManager()
        self.data = dm.load_csv(file='data/doc2vec_test_data/0702/merge_0629_adj.csv', encoding='utf-8')
        self.size = len(self.doc2idx.values())

    def get_words(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def most_similar_terms(self, topn=10):
        df = pd.DataFrame()
        data = pd.read_csv('data/doc2vec_test_data/0702/including_words_list.csv', encoding='utf-8')
        seed_term = data['Includingwords']

        for term in seed_term:
            similar_terms = self.model.wv.most_similar(term)
            temp = []
            for s_term, score in similar_terms:
                if score >= 0.8:
                    temp.append(s_term)
                else:
                    temp.append('none')
            df.loc[:, term] = pd.Series(temp)
        df.to_csv('data/doc2vec_test_data/0702/terms_results.csv', mode='w', encoding='utf-8')
        return df

    def most_similar(self, job_id, topn=10):
        similar_jobs = self.model.docvecs.most_similar('Job_ID_' + str(job_id), topn=topn)
        temp = 'Job_ID_' + str(job_id)
        print(f'(Query Job Title : {self.get_job_title(temp)})')
        job_ids = []
        job_titles = []
        scores = []
        for job_id, score in similar_jobs:
            job_titles.append(self.get_job_title(job_id)[0])
            job_ids.append(job_id)
            scores.append(score)
        df = pd.DataFrame({'ID':job_ids, 'Job_Title':job_titles, 'Score':scores})
        return df

    def most_similar_result(self, data_size, topn):
        df = pd.DataFrame()
        i = 0
        for job_id in range(data_size):
            job_id = 'Job_ID_' + str(job_id)

            title = self.get_job_title(job_id)[0]
            title = f'{title}({str(i)})'
            similar_jobs = self.model.docvecs.most_similar(job_id, topn=topn)
            sim_list = []
            for sim_job_id, score in similar_jobs:
                if score >= 0.8:
                    sim_job_titles = self.get_job_title(sim_job_id)[0]
                    sim_job_id = sim_job_id.split('_')[2]
                    input = f'{sim_job_titles}({sim_job_id})'
                    sim_list.append(input)
                else:
                    sim_list.append('None')
            i = i + 1
            df.loc[:, title] = pd.Series(sim_list)

        df.to_csv('data/doc2vec_test_data/0702/sim_title_result.csv', mode='w', encoding='utf-8')
        return df

    def get_titles_in_corpus(self, n_sample=5):
        job_ids = self.model.docvecs.doctags.keys()
        # job_ids = random.sample(self.model.docvecs.doctags.keys(), n_sample)
        return {job_id: self.get_job_title(job_id) for job_id in job_ids}

    def get_word_cloud(self, word_count_dict):
        taglist = pytagcloud.make_tags(word_count_dict.items(), maxsize=100)
        pytagcloud.create_tag_image(taglist, 'data/doc2vec_test_data/0702/word_cloud.jpg', size=(1200, 800), rectangular=False)

    def get_word_graph(self, word_count_dict):
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.grid(True)


        Sorted_Dict_Values = sorted(word_count_dict.values(), reverse=True)
        Sorted_Dict_Keys = sorted(word_count_dict, key=word_count_dict.get, reverse=True)

        plt.bar(range(len(word_count_dict)), Sorted_Dict_Values, align='center')
        plt.xticks(range(len(word_count_dict)), list(Sorted_Dict_Keys), rotation='90', fontsize=5)
        plt.figure(num=None, figsize=(20, 10), dpi=80)
        plt.show()

    def get_word_count(self, data):
        sline = [' '.join(line) for line in data]
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
        word_count_list = sorted(word_count.items(), key=lambda x:x[1], reverse=True)

        return word_count, word_count_list

    def get_job_title(self, job_id):
        data = self.data
        job_id = str(job_id)
        job_id = job_id.split('_')
        job_id = int(job_id[2])
        s = data[data['id'] == job_id]['job_title']
        s = s.tolist()
        # s = data[data['id']==int(job_id)]['job_title']
        # print(s)
        return s

    def get_simiarity(self, model):
        size = len(self.doc2idx.values())
        print(size)
        # total_result_dict = []
        # for i in range(size):
        #     result = self.model.docvecs.most_similar('Job_ID_'+str(i), topn=size)
        #     result_klist = [int(item[0].split('_')[2]) for item in result]
        #     result_slist = [item[1] for item in result]
        #     result_dict = {}
        #     for j in range(len(result_klist)):
        #         result_dict[result_klist[j]] = result_slist[j]
        #     total_result_dict.append(result_dict)
        sim_matrix=[]
        # print(total_result_dict[0])
        # for dic in total_result_dict:
        for i in range(size):
            _matrix = []
            for j in range(size):
                _matrix.append(self.model.docvecs.similarity('Job_ID_'+str(i), 'Job_ID_'+str(j)))
            sim_matrix.append(_matrix)
        np_matrix = np.array(sim_matrix)
        df = pd.DataFrame(np_matrix)
        col_name = ['Job_ID_'+str(i) for i in range(size)]
        row_name = ['Job_ID_'+str(i) for i in range(size)]
        df.columns = col_name
        df.index = row_name
        print(df.head())
        return df

    def similarity_matrix_to_csv(self):
        df = self.get_simiarity(self.model)
        dm = DataManager()
        dm.save_csv(df, file_name='data/doc2vec_test_data/0702/sim_matrix.csv', save_mode='w', encoding='utf-8')


    def visualize_jobs(self, palette='Viridis256', type='between'):
        print('Visualization Start')
        job_ids = self.get_titles_in_corpus(n_sample=len(self.model.docvecs.doctags.keys()))
        job_titles = [key for key in job_ids.keys()]
        job_vecs = [self.model.docvecs[self.doc2idx[job_id]] for job_id in tqdm(job_ids.keys())]

        if type == 'between':
            visualize_between_words(job_titles, job_vecs, palette, use_notebook=self.use_notebook)
        else:
            visualize_words(job_titles, job_vecs, palette, use_notebook=self.use_notebook)


        #     temp = []
        #     for i in range(len(dic)):
        #         temp.append(dic[i])
        #     print(temp)
    # def get_movie_title(self, movie_id):
    #     url = 'http://movie.naver.com/movie/point/af/list.nhn?st=mcode&target=after&sword=%s' % movie_id.split("_")[1]
    #     resp = requests.get(url)
    #     root = html.fromstring(resp.text)
    #     try:
    #         title = root.xpath('//div[@class="choice_movie_info"]//h5//a/text()')[0]
    #     except:
    #         title = ""
    #     return title

# dve = Doc2VecEvaluator('data/doc2vec_test_data/doc2vec.model')
# dve.visualize_jobs()
