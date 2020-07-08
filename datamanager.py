import pandas as pd
import csv
from tqdm import tqdm
import re
# from konlpy.tag import Mecab
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()
import MySQLdb

class DataManager:
    def __init__(self):
        self.host = '203.252.18.241'
        self.user = 'admin'
        self.password = '73127312Ab'
        self.port = 3306
        self.db = 'blog_data'

    def load_csv(self, file, encoding):
        csv_data = pd.read_csv(file, encoding=encoding)
        return csv_data

    def save_csv(dataFrame, file_name, save_mode, encoding):
        dataFrame.to_csv(file_name, mode=save_mode, encoding=encoding)
        print('파일 저장 완료')

    def text_cleanning(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U0001F1F2-\U0001F1F4"  # Macau flag
#         u"\U0001F1E6-\U0001F1FF"  # flags
#         u"\U0001F600-\U0001F64F"
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U0001f926-\U0001f937"
#         u"\U0001F1F2"
#         u"\U0001F1F4"
#         u"\U0001F620"
#         u"\u200d"
#         u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub('<.+?>', '', text, 0).strip()
        text = re.sub('[^0-9a-zA-Zㄱ-힗]', ' ', text)
        text = ' '.join(text.split())
        pattern_email = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9]+\.[a-zA-Z0-9.]+)'
        repl = ''
        text = re.sub(pattern=pattern_email, repl=repl, string=text)
        text = re.sub('[-=+,#/\?:^$.@*\"”※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', repl, text)
        text = text.replace('·', '')
        text = text.replace('“', '')
        text = text.replace('”', '')
        return text

    # def dataMorph(file_name):
    #     nouns_list = []
    #     csv_data = DataManager.load_csv(file_name, "text")
    #     mecab = Mecab()
    #     nouns_list = [mecab.nouns(text) for text in csv_data['text']]
    #     csv_data['nouns'] = nouns_list
    #     save_csv(csv_data, file_name, "w")
    #     return csv_data

    def getData_fromCSV(self, file_name):
        csv_data = DataManager.load_csv(file_name, 'utf-8')
        return csv_data

    def connectionDB(self):
        conn = pymysql.connect(host=self.host, user=self.user, password=self.password, port=self.port, db=self.db, charset='utf8')
        return conn

    def select_all_db(self, table):
        conn = self.connectionDB()
        sql = f'select * from {table}'
        result = pd.read_sql_query(sql, conn)
        print(result.head())
        conn.close()
        return result

    def insert_db(self, table, data):
        engine = create_engine(f'mysql+mysqldb://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}', encoding='utf-8')
        conn = engine.connect()
        data.to_sql(name=table, con=conn, if_exists='append', index=False)
        conn.close()

    def make_stop_words(self, word_count_list, number):
        stop_words_list = [word[0] for word in word_count_list[:number]]
        df = pd.DataFrame({'Stopwords':stop_words_list})
        df.to_csv('data/doc2vec_test_data/0702/stopwords_list.csv', mode='w', encoding='utf-8')


    # def get_nouns(self, data):


if __name__=='__main__':
    dm = DataManager()
    data = dm.load_csv('data/doc2vec_test_data/0702/stopwords_list.csv', 'utf-8')
    print(data)





