from doc2vec_eval import Doc2VecEvaluator, Doc2VecModeler
from doc2vec_input2 import Doc2VecInput, Configuration
from tm_input import TMInput
from gensim.models import Doc2Vec
import pandas as pd
from datamanager import DataManager
import visualize_utils
import multiprocessing
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    dm = DataManager()
    config = Configuration()

    print('============ MENU ============')
    print('==== 1. Processing All    ====')
    print('==== 2. Similarity        ==== ')
    print('==== 3. Visualization     ==== ')

    choice = input(' >> Select Number : ')

    if choice == '1':
        input_data = Doc2VecInput(config)

        #cores = multiprocessing.cpu_count()

        #model = Doc2Vec(tagged_doc, dm=0, dbow_words=1, window=8, seed=1234, alpha=0.025, vector_size=850, min_count=1, min_alpha=0.025, workers=cores, hs=1, negative=10, epochs=10)
        Doc2VecModeler(config=config, tagged_doc=input_data.tagged_doc_)

        #
        model = Doc2VecEvaluator(config=config)

        # 단어 빈도수 계산 및 시각화하기(시각화는 빈도수 기준 상위 100개)
        # data = model_doc2vec.get_words('data/doc2vec_test_data/0702/model_doc2vec.corpus')
        # for i in range(len(data)):
        #     print(f'[{i}] : {data[i]}')
        # word_count, word_count_list = model_doc2vec.get_word_count(data)
        # key_list = []
        # value_list = []
        # for item in word_count_list[:50]:
        #     key_list.append(item[0])
        #     value_list.append(item[1])
        # new_word_count_dict = dict(zip(key_list, value_list))
        # model_doc2vec.get_word_cloud(new_word_count_dict)
        # model_doc2vec.get_word_graph((new_word_count_dict))
        # df = pd.DataFrame({'Terms': key_list, 'Frequency': value_list})
        # df.to_csv('data/doc2vec_test_data/0702/frequency.csv', mode='w', encoding='utf-8') ## 빈도 데이터 저장

        # ## stopwords_list.csv 생성
        # dm.make_stop_words(word_count_list, 200)


        # ## 단어간 유사도 점수 계산
        # print(model.most_similar_terms())


        # similarity matrix 저장하기
        df = model.get_similarity(model)


        # ## JOB_ID - Job Title 매칭 파일 저장하기
        # df = pd.DataFrame(columns=['Job_ID', 'Job_Title'])
        # for i in range(model.size):
        #     # print(model_doc2vec.get_job_title('Job_ID_'+str(i)))
        #     df.loc[i] = ['Job_ID'+str(i), model.get_job_title('Job_ID_'+str(i))]
        # df.to_csv('data/doc2vec_test_data/0702/job_title_after.csv', index=False)

        # 시각화
        model.visualize_jobs(type='tsne')

        # id = ['Job_ID_1851', 'Job_ID_2414', 'Job_ID_1842']
        # for name in id:
        #     print(model_doc2vec.get_job_title(name))


        # ## 유사도 계산해보기
        # print(model_doc2vec.most_similar(job_id=1, topn=10))

        ## 전체 데이터 유사도 저장(topn=10)
        model.most_similar_result(len(model.doc2idx.values()), 10)

    elif choice == '2':
        model = Doc2VecEvaluator(config=config)
        model.get_similarity(model)
        model.most_similar_result(len(model.doc2idx.values()), 10)

    elif choice == '3':
        model = Doc2VecEvaluator(config=config)
        model.visualize_jobs(type='tsne')

    elif choice == '4':
        model = Doc2VecEvaluator(config=config)
        model.most_similar_result(len(model.doc2idx.values()), 10)

if __name__=='__main__':
    main()