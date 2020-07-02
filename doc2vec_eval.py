from gensim.models import Doc2Vec
from datamanager import DataManager
import random
import pandas as pd
import numpy as np
import pickle
from visualize_utils import visualize_between_words, visualize_words

class Doc2VecEvaluator:

    def __init__(self, model_fname, use_notebook=False):
        self.model = Doc2Vec.load(model_fname)
        self.doc2idx = {el:idx for idx, el in enumerate(self.model.docvecs.doctags.keys())}
        self.use_notebook = use_notebook
        dm = DataManager()
        self.data = dm.load_csv(file='data/doc2vec_test_data/0702/merge_0629.csv', encoding='utf-8')
        self.size = len(self.doc2idx.values())

    def get_words(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def most_similar(self, job_id, topn=10):
        similar_movies = self.model.docvecs.most_similar('Job_ID_' + str(job_id), topn=topn)
        temp = 'Job_ID_' + str(job_id)
        print(f'(Query Job Title : {self.get_job_title(temp)})')
        job_ids = []
        job_titles = []
        scores = []
        for job_id, score in similar_movies:
            job_titles.append(self.get_job_title(job_id)[0])
            job_ids.append(job_id)
            scores.append(score)
        df = pd.DataFrame({'ID':job_ids, 'Job_Title':job_titles, 'Score':scores})
        return df

    def get_titles_in_corpus(self, n_sample=5):
        job_ids = self.model.docvecs.doctags.keys()
        # job_ids = random.sample(self.model.docvecs.doctags.keys(), n_sample)
        return {job_id: self.get_job_title(job_id) for job_id in job_ids}

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
        job_vecs = [self.model.docvecs[self.doc2idx[job_id]] for job_id in job_ids.keys()]

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
