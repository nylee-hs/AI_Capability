from bokeh.models import ColumnDataSource, LabelSet, LinearColorMapper
from bokeh.plotting import figure, output_file, save
from bokeh.io import export_png, output_notebook, show
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
import os
import multiprocessing
from sklearn.manifold import TSNE

class Doc2VecModeler:
    def __init__(self, tagged_doc, config):
        self.model_path = config.model_path
        self.data_path = config.data_path
        self.data_name = config.data_file_name
        self.tagged_doc = tagged_doc
        self.model = self.run()

    def run(self):
        # cores = multiprocessing.cpu_count()
        model = Doc2Vec(self.tagged_doc, dm=0, dbow_words=1, window=10, alpha=0.025, vector_size=1024, min_count=10,
                min_alpha=0.025, workers=4, hs=1, negative=20, epochs=10)
        model.save(self.model_path + self.data_name + '_doc2vec.model')
        print('==== End Doc2Vec Process ====')
        return model

    # def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
    #     print('==== Start Doc2Vec Process ====')
    #     directory = 'data/doc2vec_test_data/' + input('data date : ') + '/model_doc2vec/'
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     return directory
    #
    # def get_file_name(self):
    #     file_name = input(' > file_name : ')
    #     return file_name

class Doc2VecEvaluator:

    def __init__(self, config, use_notebook=False):
        self.data_path = config.data_path
        self.model_path = config.model_path
        self.data_name = config.data_file_name

        self.model = Doc2Vec.load(self.model_path+self.data_name+'_doc2vec.model')
        self.doc2idx = {el:idx for idx, el in enumerate(self.model.docvecs.doctags.keys())}
        self.use_notebook = use_notebook
        dm = DataManager()

        self.data = dm.load_csv(file=self.data_path+self.data_name+'.csv', encoding='utf-8')
        self.size = len(self.doc2idx.values())


    # def get_data_name(self):
    #     data_name = input(' > data_name for doc2vec : ') ## 분석하고자 하는 csv 파일의 이름을 입력
    #     return data_name
    #
    # def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
    #     print('==== Analyzing Doc2Vec Process ====')
    #     analysis_date = input(' > date : ')
    #     data_directory = 'data/doc2vec_test_data/'+ analysis_date + '/data/'
    #     model_directory = 'data/doc2vec_test_data/' + analysis_date + '/model_doc2vec/'
    #     if not os.path.exists(data_directory):
    #         os.makedirs(data_directory)
    #     if not os.path.exists(model_directory):
    #         os.makedirs(model_directory)
    #     return data_directory, model_directory

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
        df.to_csv(self.model_path+self.data_name+'_terms_results.csv', mode='w', encoding='utf-8')
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
        df.to_csv(self.model_path+self.data_name+'_sim_matrix.csv', mode='w', encoding='utf-8')
        return df

    def get_titles_in_corpus(self, n_sample=5):
        job_ids = self.model.docvecs.doctags.keys()
        # job_ids = random.sample(self.model_doc2vec.docvecs.doctags.keys(), n_sample)
        return {job_id: self.get_job_title(job_id) for job_id in job_ids}

    def get_word_cloud(self, word_count_dict):
        taglist = pytagcloud.make_tags(word_count_dict.items(), maxsize=100)
        pytagcloud.create_tag_image(taglist, self.model_path+self.data_name+'_word_cloud.jpg', size=(1200, 800), rectangular=False)

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

    def most_similar_result(self, data_size, topn):
        print('   -> creating most similar job matrix')
        df = pd.DataFrame()
        i = 0
        keys_list = list(self.doc2idx.keys())
        for job_id in keys_list:
            job_id = 'Job_ID_' + str(job_id).split('_')[2]

            title = self.get_job_title(job_id)[0]
            title = f'{title}({str(i)})'
            similar_jobs = self.model.docvecs.most_similar(job_id, topn=len(keys_list))
            sim_list = []
            for sim_job_id, score in similar_jobs:
                if score >= 0.7:
                    sim_job_titles = self.get_job_title(sim_job_id)[0]
                    sim_job_id = sim_job_id.split('_')[2]
                    input = f'{sim_job_titles}({sim_job_id})'
                    sim_list.append(input)
                else:
                    sim_list.append('None')
            i = i + 1
            df.loc[:, title] = pd.Series(sim_list)

        df.to_csv(self.model_path+self.data_name+'_sim_title_result.csv', mode='w', encoding='utf-8')
        return df

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

    def get_similarity(self, model):
        print('   -> get similarity values')
        keys_list = list(self.doc2idx.keys())
        size = len(keys_list)
        keys_list = [key.split('_')[2] for key in keys_list]
        # total_result_dict = []
        # for i in range(size):
        #     result = self.model_doc2vec.docvecs.most_similar('Job_ID_'+str(i), topn=size)
        #     result_klist = [int(item[0].split('_')[2]) for item in result]
        #     result_slist = [item[1] for item in result]
        #     result_dict = {}
        #     for j in range(len(result_klist)):
        #         result_dict[result_klist[j]] = result_slist[j]
        #     total_result_dict.append(result_dict)
        sim_matrix=[]
        # print(total_result_dict[0])
        # for dic in total_result_dict:
        for i in keys_list:
            _matrix = []
            for j in keys_list:
                _matrix.append(self.model.docvecs.similarity('Job_ID_'+str(i), 'Job_ID_'+str(j)))
            sim_matrix.append(_matrix)
        np_matrix = np.array(sim_matrix)
        df = pd.DataFrame(np_matrix)
        col_name = ['Job_ID_'+str(i) for i in keys_list]
        row_name = ['Job_ID_'+str(i) for i in keys_list]
        df.columns = col_name
        df.index = row_name
        print(df.head())
        df.to_csv(self.model_path+self.data_name+'_sim_matrix.csv', mode='w', encoding='utf-8')
        return df


    def word_visulize(self, words, vecs, palette="Viridis256", filename="/notebooks/embedding/words.png",
                        use_notebook=False):
        circle_size = input('     >> circle size : ')
        text_size = input('     >> font size : ') + 'pt'

        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(vecs)

        df = pd.DataFrame(columns=['x', 'y', 'word'])
        df['x'], df['y'], df['word'] = tsne_results[:, 0], tsne_results[:, 1], list(words)
        # df['x'], df['y'] = tsne_results[:, 0], tsne_results[:, 1]
        df = df.fillna('')
        print(df.head())
        # print(ColumnDataSource.from_df(df))
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                          text_font_size=text_size, text_color="#555555",
                          source=source, text_align='center')

        color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))
        plot = figure(plot_width=1200, plot_height=1200)
        plot.scatter("x", "y", size=int(circle_size), source=source, color={'field': 'y', 'transform': color_mapper}, line_color=None,
                     fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot)
        output_file(self.model_path+self.data_name+'_tsne.html')
        save(plot)

    def visualize_jobs(self, palette='Viridis256', type='between'):
        print('   -> Visualization Start')
        job_ids = self.get_titles_in_corpus(n_sample=len(self.model.docvecs.doctags.keys()))
        #job_titles = [key for key in job_ids.keys()]
        keys_list = [key for key in job_ids.keys()]
        values_list = [value for value in job_ids.values()]
        job_titles = []
        for i in range(len(keys_list)):
            key = keys_list[i]
            key = key.split('_')[2]
            value = values_list[i][0]
            value = value.split('_')[-1]
            job_titles.append(key+'_'+value)


        #job_titles = self.get_job_title()
        job_vecs = [self.model.docvecs[self.doc2idx[job_id]] for job_id in tqdm(job_ids.keys())]

        if type == 'between':
            self.word_visulize(job_titles, job_vecs, palette, use_notebook=self.use_notebook)
        else:
            self.word_visulize(job_titles, job_vecs, palette, use_notebook=self.use_notebook)


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

# dve = Doc2VecEvaluator('data/doc2vec_test_data/doc2vec.model_doc2vec')
# dve.visualize_jobs()
