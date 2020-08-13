import os
import pickle
from collections import defaultdict

from gensim import corpora
from gensim.models import ldamulticore, CoherenceModel, LdaModel
import operator
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis.gensim as gensimvis

class LDABuilder:
    def __init__(self):
        self.data_path, self.model_path = self.make_save_path()
        self.data_name = self.get_data_name()
        self.corpus = self.get_corpus()
        self.documents = self.get_documents()
        self.num_topics = 0


    def get_data_name(self):
        data_name = input(' > data_name for lda : ') ## 분석하고자 하는 csv 파일의 이름을 입력
        return data_name

    def get_corpus(self):
        with open(self.data_path + self.data_name+'.corpus', 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    def get_documents(self):
        with open(self.data_path + self.data_name+'_tm.documents', 'rb') as f:
            documents = pickle.load(f)
        return documents

    def getOptimalTopicNum(self):
        dictionary = corpora.Dictionary(self.corpus)
        corpus = [dictionary.doc2bow(text) for text in self.corpus]

        com_nums = []
        for i in range(10, 110, 10):
            if i == 0:
                p = 1
            else:
                p = i
            com_nums.append(p)

        coherence_list = []

        for i in com_nums:
            # lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
            #                                       id2word=dictionary,
            #                                       num_topics=i,
            #                                       iterations=100,
            #                                       alpha='auto',
            #                                       random_state=100,
            #                                       update_every=1,
            #                                       chunksize=10,
            #                                       passes=20,
            #                                       per_word_topics=True)
            lda = ldamulticore.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            passes=20,
                                            num_topics=i,
                                            workers=4,
                                            iterations=100,
                                            alpha='symmetric',
                                            gamma_threshold=0.001)
            coh_model_lda = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_value = coh_model_lda.get_coherence()

            # coh = lda.log_perplexity(corpus)
            coherence_list.append(coherence_value)
            print('k = {}  coherence value = {}'.format(str(i), str(coherence_value)))

        coh_dict = dict(zip(com_nums, coherence_list))
        sorted_coh_dict = sorted(coh_dict.items(), key=operator.itemgetter(1), reverse=True)
        plt.plot(com_nums, coherence_list, marker='o')
        plt.xlabel('topic')
        plt.ylabel('coherence value')
        plt.draw()
        fig = plt.gcf()
        fig.savefig(self.model_path+self.data_name+'_coherence.png')
        t_ind = np.argmax(coherence_list)
        self.num_topics = sorted_coh_dict[0][0]
        print('optimal topic number = ', sorted_coh_dict[0][0])
        return sorted_coh_dict[0][0]

    def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== Modeling Building Process ====')
        analysis_date = input(' > date : ')
        data_directory = 'data/doc2vec_test_data/'+ analysis_date + '/data/'
        model_directory = 'data/doc2vec_test_data/'+ analysis_date + '/model/'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        return data_directory, model_directory

    def saveLDAModel(self):
        print(' ...start to build lda model...')
        dictionary = corpora.Dictionary(self.corpus)
        corpus = [dictionary.doc2bow(text) for text in self.corpus]

        lda_model = ldamulticore.LdaMulticore(corpus=corpus,
                                        id2word=dictionary,
                                        passes=20,
                                        num_topics=self.num_topics,
                                        workers=4,
                                        iterations=100,
                                        alpha='symmetric',
                                        gamma_threshold=0.001)
        print("00000")
        all_topics = lda_model.get_document_topics(corpus, minimum_probability=0.5, per_word_topics=False)

        documents = self.documents
        print(len(documents))
        with open(self.model_path + self.data_name + '_lda.results', 'w', -1, 'utf-8') as f:
            for doc_idx, topic in enumerate(all_topics):
                if len(topic) == 1:
                    print(doc_idx, ' || ', topic[0])
                    topic_id, prob = topic[0]
                    f.writelines(documents[doc_idx].strip() + "\u241E" + ' '.join(self.corpus[doc_idx]) + "\u241E" + str(topic_id) + "\u241E" + str(prob) + '\n')
        lda_model.save(self.model_path + self.data_name +'_lda.model')
        with open(self.model_path+self.data_name+'_model.dictionary', 'wb') as f:
            pickle.dump(dictionary, f)

        return lda_model

    def main(self):
        # self.model_path = self.make_save_path('models/0722')
        self.saveLDAModel()

class LDAModeler:
    def __init__(self):
        self.data_path, self.model_path = self.get_path()
        self.data_name = self.get_data_name()
        self.all_topics = self.load_results(result_fname=self.model_path + self.data_name + '_lda.results')
        self.model = LdaModel.load(self.model_path + self.data_name + '_lda.model')
        self.corpus = self.get_corpus(self.data_path + self.data_name + '.corpus')
        self.dictionary = self.get_dictionary(self.model_path + self.data_name + '_lda.model.id2word')

    def get_data_name(self):
        data_name = input(' > data_name for lda : ')
        return data_name

    def get_path(self):  ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== LDA Model Analyzer ====')
        date = input(' > date : ')
        model_directory = 'data/doc2vec_test_data/' + date + '/model/'
        data_directory = 'data/doc2vec_test_data/' + date + '/data/'
        return data_directory, model_directory

    def view_lda_model(self, model, corpus, dictionary):
        corpus = [dictionary.doc2bow(doc) for doc in corpus]
        prepared_data = gensimvis.prepare(model, corpus, dictionary)
        print(prepared_data)
        # pyLDAvis.save_html(prepared_data, self.model_path+'/vis_result.html')

    def get_corpus(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    def load_results(self, result_fname):
        topic_dict = defaultdict(list)
        with open(result_fname, 'r', encoding='utf-8') as f:
            for line in f:
                sentence, _, topic_id, prob = line.strip().split('\u241E')
                topic_dict[int(topic_id)].append((sentence, float(prob)))

        for key in topic_dict.keys():
            topic_dict[key] = sorted(topic_dict[key], key=lambda x: x[1], reverse=True)
        return topic_dict

    def show_topic_docs(self, topic_id, topn=10):
        return self.all_topics[topic_id][:topn]

    def show_topic_words(self, topic_id, topn=10):
        return self.model.show_topic(topic_id, topn)

    def show_topics(self, model):
        return self.model.show_topics(fommated=False)

    # def show_new_document_topic(self, documents, model):
    #     mecab = Mecab()
    #     tokenized_documents = [mecab.morphs(document) for document in documents]
    #     curr_corpus = [self.model.id2word.doc2bow(tokenized_documents) for tokenized_document in
    #                    tokenized_documents]
    #     topics = self.model.get_document_topics(curr_corpus, minimum_probability=0.5, per_word_topics=False)
    #     for doc_idx, topic in enumerate(topics):
    #         if len(topic) == 1:
    #             topic_id, prob = topic[0]
    #             print(documents[doc_idx], ', topic id: ', str(topic_id), ', prob:', str(prob))
    #         else:
    #             print(documents[doc_idx], ', there is no dominant topic.')

    def get_dictionary(self, dic_fname):
        with open(dic_fname, 'rb') as f:
            dictionary = pickle.load(f)
        return dictionary



