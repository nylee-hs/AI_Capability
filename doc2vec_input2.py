import pickle
import os.path
import pytagcloud
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
import spacy
from datamanager import DataManager
from gensim.models.doc2vec import TaggedDocument
# from konlpy.tag import Mecab
import gensim
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
import re
import nltk
from nltk.corpus import stopwords
from utilizer import Integration
from pprint import pprint
import pandas as pd
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Configuration:
    def __init__(self):
        self.date = self.make_save_path()
        self.data_path = self.date + '/data/'
        self.model_path = self.date + '/model_doc2vec/'
        self.data_file_name = self.get_file_name()
        self.factor = self.get_factor()

    def get_file_name(self):
        file_name = input(' > file_name : ')
        return file_name

    def get_factor(self): ## '0: responsiblities' 또는 '1: requirements' 입력
        factor = input(' > factor(description: 0, responsibilities:1, requirements:2) : ')
        if factor == '0':
            factor = 'job_description'
        elif factor == '1':
            factor = 'job_description_responsibilities'
        elif factor == '2':
            factor = 'job_description_requirements'
        return factor

    def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== Configuration ====')
        directory = 'data/doc2vec_test_data/' + input('data date : ')
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory


class Doc2VecInput:
    def __init__(self, config):
        self.data_path = config.data_path
        self.data_file_name = config.data_file_name
        self.factor = config.factor
        self.capa_terms = ['Strong', 'Experience in', 'experience in', 'experience', 'Experience', 'experience with', 'proven', 'Proven', 'Valid', 'Familiarity with', 'familiarity with',
                           'Excellent', 'Understading', 'Ability to', 'Ability', "degree in", "Knowledge of", "Must be", 'Skills', 'Degree in', "Bachelor's", "Master", "Doctoral", 'MS degree', 'BS degree', 'Must have',
                           'Evidence', 'Exposure', 'Graduate degree', 'Expert', 'Ph. D.', 'Proficiency in', 'proficiency', 'BS', 'MS', 'Ph.D', 'Must', 'Certifications', 'Skills in', 'skills in']
        self.capa_terms_cap = ['Strong', 'Experience', 'Proven', 'Familiarity', 'Ability', 'Excellent', 'Understading', 'Must have', 'Must be', 'Expert', 'Evidence', 'Exposure', 'Knowledge', 'Ability to', 'Proficiency', 'Demonstrate', 'Deep Understanding',
                               'Bachelors', 'Masters', 'Skills in']

        self.job_id, self.description = self.pre_prosseccing()
        self.tagged_doc_ = self.make_tag_document()


    def get_factor(self): ## '0: responsiblities' 또는 '1: requirements' 입력
        factor = input(' > factor(description: 0, responsibilities:1, requirements:2) : ')
        if factor == '0':
            factor = 'job_description'
        elif factor == '1':
            factor = 'job_description_responsibilities'
        elif factor == '2':
            factor = 'job_description_requirements'
        return factor

    def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== Preprocessing ====')
        directory = 'data/doc2vec_test_data/' + input('data date : ') + '/data/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def make_ngram(self, text, n):  ## n == 3 --> trigram, n==2 --> bigram
        # min_count : Ignore all words and bigrams with total collected count lower than this value.
        # threshold : Represent a score threshold for forming the phrases (higher means fewer phrases).
        #             A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
        #             Heavily depends on concrete scoring-function, see the scoring parameter.
        if n == 2:
            print(' ...make bigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            return [bigram_mod[doc] for doc in text]
        elif n == 3:
            print(' ...make trigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram = gensim.models.Phrases(bigram[text], threshold=20.0)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            return [trigram_mod[bigram_mod[doc]] for doc in text]


    def data_text_cleansing(self, text):
        print(' ...Run text cleanning...')
        # Convert to list
        data = text['job_description'].str.replace(pat=r'[^A-Za-z0-9]', repl= r' ', regex=True)
        data = data.dropna(axis=0).reset_index(drop=True)
        data = data.tolist()

        # # 영문자 이외의 문자는 공백으로 변환
        # data = [re.sub('[^a-zA-Z]', ' ', str(sent)) for sent in data]
        #
        # for sent in data:
        #     print(sent)

        # Remove emails
        data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]

        # Remove new line characters
        data = [re.sub('\s\s+', ' ', str(sent)) for sent in data]

        # Remove distracting single quotes
        data = [re.sub('\'', '', sent) for sent in data]

        data = [sent.replace('\n', ' ') for sent in data]

        return data

    def get_stop_words(self):
        print('   -> Getting stop word list...')
        file = 'stopwords_list.csv'
        stop_words_list = []
        if os.path.isfile(self.data_path+file):
            print('     -> Stop Words File is found')
            dm = DataManager()
            df = dm.load_csv(file=self.data_path + file, encoding='utf-8')
            stop_words_list = df['Stopwords'].tolist()
        else:
            print('     -> Stop Words File is not found')
        return stop_words_list

    def get_including_words(self):
        print('    -> Getting including word list...')
        file = 'including_words_list.csv'
        including_words_list = []
        if os.path.isfile(self.data_path+file):
            print('    -> Including Words File is found')
            dm = DataManager()
            df = dm.load_csv(file=self.data_path+file, encoding='utf-8')
            including_words_list = df['Includingwords'].tolist()
            print(f'    -> total : {len(including_words_list)} words')
        else:
            print('     -> Including Words File is not found')
        return including_words_list

    def remove_stopwords(self, texts):
        print(' ...Remove stopwords...')
        stop_words = stopwords.words('english')
        stopwords_list = self.get_stop_words()
        print('   -> Append stopwords list: ', len(stopwords_list), 'words')
        stop_words.extend(stopwords_list)  #추가할 stopwords list
        return [[word for word in simple_preprocess(str(doc), deacc=True, min_len=1) if word not in stop_words] for doc in texts]

    def word_filtering(self, texts):
        print(' ...Filtering words...')
        including_list = self.get_including_words()
        if len(including_list) == 0:
            return texts
        else:
            return [[word for word in doc if word in including_list] for doc in texts]

    def lematization(self, texts, allowed_postags=['NOUN', 'PROPN', 'NUM', 'VERB']): #['NOUN', 'ADJ', 'VERB', 'ADV']
        print(' ...Make lematization...')
        texts_out = []
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
        for sent in tqdm(texts):
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        # print(texts_out[0])
        return texts_out

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), min_len=1, deacc=True))

    def make_unique_words(self, data_lemmatized):
        print(data_lemmatized)
        uniquewords = [list(set(item)) for item in data_lemmatized]
        return uniquewords

    def get_file_name(self):
        file_name = input(' > file_name : ')
        return file_name

    def get_requirements_from_document(self, data):
        descpription = data['job_description']
        new_description = []
        c_terms = '|'.join(self.capa_terms)
        p = re.compile(c_terms)
        for texts in descpription:

            texts = texts.replace('\n', '. ')   ## 줄바꿈 --> 마침표 으로 처리
            texts = texts.replace("’", '')
            texts = texts.replace('“', '')
            texts = texts.replace('”', '')
            texts = texts.replace('\r', '')
            for term in self.capa_terms_cap:
                texts = texts.replace(term, '. '+term)
            str_list = texts.split('.') ## 마침표 기준 split()

            find_sentence = []
            for sentence in str_list:
                m = p.search(sentence)
                if m:
                    find_sentence.append(sentence)
            new_texts = '. '.join(find_sentence)
            new_description.append(new_texts)

        data['job_description_requirements'] = new_description
        data.to_csv(self.data_path+ self.data_file_name+'_new.csv', mode='w', encoding='utf-8')
        data['job_description'] = new_description
        data['job_description'].replace('', np.nan, inplace=True)
        data.dropna(subset=['job_description'], inplace=True)
        return data

    def get_word_count(self, data_lemmatized):
        sline = [' '.join(line) for line in data_lemmatized]
        # print(len(sline))
        #단어 빈도 계산
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
        key_list = []
        value_list = []
        for item in word_count_list:
            key_list.append(item[0])
            value_list.append(item[1])
        new_word_count_dict = dict(zip(key_list[:50], value_list[:50]))  ## 워드 클라우드는 50개까지 보여주기
        self.get_word_cloud(new_word_count_dict)
        self.get_word_graph((new_word_count_dict))
        df = pd.DataFrame({'Terms': key_list, 'Frequency': value_list})
        df.to_csv(self.data_path + self.data_file_name + '_frequency.csv', mode='w', encoding='utf-8') ## 빈도 데이터 저장

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


        tfidf = TfidfVectorizer(min_df=0,  sublinear_tf=True, max_df=0.8).fit(sline)
        tfidf_sp = tfidf.transform(sline)
        tfidf_dict = tfidf.get_feature_names()
        data_array = tfidf_sp.toarray()
        df = pd.DataFrame(data_array, columns=tfidf_dict)
        # sum_array = df.sum(axis=0)
        df.loc['TF_IDF_SUM', :] = df.sum()
        df.to_csv(self.data_path + self.data_file_name + '_tf_idf.csv', mode='w', encoding='utf-8')  ## tf-idf 데이터 저장

    def get_word_cloud(self, word_count_dict):
        taglist = pytagcloud.make_tags(word_count_dict.items(), maxsize=100)
        pytagcloud.create_tag_image(taglist, self.data_path+self.data_file_name+'_word_cloud.jpg', size=(1200, 800), rectangular=False)

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

    def pre_prosseccing(self):
        print('==== Preprocessing ====')
        dm = DataManager()
        integ = Integration()
        # data = dm.load_csv(file=self.data_path + self.data_file_name+'.csv', encoding='utf-8')
        data = integ.getIntegratedData()
        data = self.get_requirements_from_document(data)

        description = data[self.factor]
        description_reset = description.dropna(axis=0).reset_index(drop=True)

        description = [sent.replace('\n', ' ') for sent in description_reset]

        with open(self.data_path + self.data_file_name+'_d2v.documents', 'wb') as f:
            pickle.dump(description, f)

        # # 수정된 job_title에서 posting_id 가지고 오기
        # posting_ids = data['posting_id']
        # posting_list = posting_ids.to_list()
        #
        # # posting_id에 따라 description_data set 만들기
        # des_data = [data['job_description'][id] for id in posting_ids]
        # title_data = [data['job_title'][id] for id in posting_ids]
        # id_list = [i for i in range(len(posting_list))]
        # df = pd.DataFrame({'id': posting_list, 'job_title': title_data, 'job_description': des_data, 'posting_id':posting_list})
        # df.to_csv('data/doc2vec_test_data/0702/merge_0629_adj.csv', mode='w', encoding='utf-8')

        # 수정된 description set 불러와 데이터 전처리 수행
        # data = dm.load_csv(file='data/doc2vec_test_data/0702/merge_0629_adj.csv', encoding='utf-8')
        sentences = self.data_text_cleansing(data)
        data_words = list(self.sent_to_words(sentences))
        data_words_nostops = self.remove_stopwords(data_words)
        bigram = self.make_ngram(data_words_nostops, n=2)
        data_lemmatized = self.lematization(data_words_nostops)

        bigram = self.make_ngram(data_lemmatized, n=2)

        # bigram = self.make_bigram(data_words_nostops)
        # data_lemmatized = self.lematization(bigram)
        # for i in range(len(bigram)):
        #     print(f'[{i}] : {bigram[i]}')

        data_lemmatized_filter = self.word_filtering(bigram)

        for i in range(len(data_lemmatized_filter)):
            print(f'[{i}] : {data_lemmatized_filter[i]}')
        # # uniquewords = self.make_unique_words(data_lemmatized)
        with open(self.data_path + self.data_file_name+'.corpus', 'wb') as f:
            pickle.dump(data_lemmatized_filter, f)

        self.get_word_count(data_lemmatized_filter)
        final_terms = pd.DataFrame({'terms': data_lemmatized_filter})
        final_terms.to_csv(self.data_path + self.data_file_name + '_final_terms.csv', mode='w', encoding='utf-8')
        print('=== end preprocessing ===')

        return data['id'], data_lemmatized_filter

    def make_tag_document(self):
        tagged_doc = []
        i = 0
        for id in self.job_id:
            tokens = self.description[i]
            doc = TaggedDocument(words=tokens, tags=['Job_ID_%s' % id])
            tagged_doc.append(doc)
            i += 1
        return tagged_doc


    # def __iter__(self):
    #     for i in range(len(self.description)):
    #         try:
    #             tokens = self.description[i]
    #             # tokens = self.tokenizer.morphs(sentence)
    #             job_id = self.job_id[i]
    #             tagged_doc = TaggedDocument(words=tokens, tags=['Job_ID_%s' % job_id])
    #             print(tagged_doc.tags)
    #             yield tagged_doc
    #
    #     # with open(self.corpus_fname, encoding='utf-8') as f:
    #     #     for line in f:
    #     #         try:
    #     #             sentence, movie_id = line.strip().split("\u241E")
    #     #             tokens = self.tokenizer.morphs(sentence)
    #     #             tagged_doc = TaggedDocument(words=tokens, tags=['MOVIE_%s' % movie_id])
    #     #             yield tagged_doc
    #         except Exception as ex:
    #             print(ex)
    #             continue

config = Configuration()
dvi = Doc2VecInput(config)
