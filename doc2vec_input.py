import pickle
import os.path
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
from pprint import pprint
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Doc2VecInput:
    def __init__(self):
        # self.tokenizer = Mecab()
        # self.corpus_fname = 'data/doc2vec_test_data/processed_review_movieid.txt'
        self.job_id, self.description = self.pre_prosseccing()

    def make_bigram(self, text):
        # min_count : Ignore all words and bigrams with total collected count lower than this value.
        # threshold : Represent a score threshold for forming the phrases (higher means fewer phrases).
        #             A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
        #             Heavily depends on concrete scoring-function, see the scoring parameter.
        bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return [bigram_mod[doc] for doc in text]

    def data_text_cleansing(self, text):
        print('Run text cleanning...')
        # Convert to list
        data = text['job_description'].str.replace(pat=r'[^A-Za-z0-9]', repl= r' ', regex=True)
        data = text['job_description'].str.replace(pat=r'[\s\s+]', repl=r' ', regex=True)
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

        return data

    def get_stop_words(self, path):
        file = 'stopwords_list.csv'
        stop_words_list = []
        if os.path.isfile(path+'/'+file):
            print('Stop Words File is found')
            dm = DataManager()
            df = dm.load_csv(file='data/doc2vec_test_data/0702/stopwords_list.csv', encoding='utf-8')
            stop_words_list = df['Stopwords'].tolist()
        else:
            print('Stop Words File is not found')
        return stop_words_list

    def remove_stopwords(self, texts):
        print('Remove stopwords...')
        stop_words = stopwords.words('english')
        stopwords_list = self.get_stop_words('data/doc2vec_test_data/0702')
        print('Append stopwords list: ', len(stopwords_list), 'words')
        stop_words.extend(stopwords_list)  #추가할 stopwords list
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def lematization(self, texts, allowed_postags=['NOUN']): #['NOUN', 'ADJ', 'VERB', 'ADV']
        print('Make lematization...')
        texts_out = []
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for sent in tqdm(texts):
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        # print(texts_out[0])
        return texts_out

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def make_unique_words(self, data_lemmatized):
        print(data_lemmatized)
        uniquewords = [list(set(item)) for item in data_lemmatized]
        return uniquewords

    def pre_prosseccing(self):
        dm = DataManager()
        data = dm.load_csv(file='data/doc2vec_test_data/0702/merge_0629_adj.csv', encoding='utf-8')
        print(data.head())
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
        bigram = self.make_bigram(data_words_nostops)
        data_lemmatized = self.lematization(bigram)
        data_lemmatized_stop = self.remove_stopwords(data_lemmatized)
        print(data_lemmatized_stop[:20])
        # uniquewords = self.make_unique_words(data_lemmatized)
        with open('data/doc2vec_test_data/0702/model.corpus', 'wb') as f:
            pickle.dump(data_lemmatized_stop, f)
        return data['id'], data_lemmatized_stop

    def __iter__(self):
        for i in range(len(self.description)):
            try:
                tokens = self.description[i]
                # tokens = self.tokenizer.morphs(sentence)
                job_id = self.job_id[i]
                tagged_doc = TaggedDocument(words=tokens, tags=['Job_ID_%s' % job_id])
                yield tagged_doc
        # with open(self.corpus_fname, encoding='utf-8') as f:
        #     for line in f:
        #         try:
        #             sentence, movie_id = line.strip().split("\u241E")
        #             tokens = self.tokenizer.morphs(sentence)
        #             tagged_doc = TaggedDocument(words=tokens, tags=['MOVIE_%s' % movie_id])
        #             yield tagged_doc
            except Exception as ex:
                print(ex)
                continue

