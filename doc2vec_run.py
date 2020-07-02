from doc2vec_eval import Doc2VecEvaluator
from doc2vec_input import Doc2VecInput
from gensim.models import Doc2Vec
import pandas as pd
from datamanager import DataManager as dm
import visualize_utils
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    corpus = Doc2VecInput()
    # cores = multiprocessing.cpu_count()
    # model = Doc2Vec(corpus, dm=0, dbow_words=1, window=5, alpha=0.025, vector_size=300, min_count=20, min_alpha=0.025, workers=cores, hs=1, negative=10)
    # model.save('data/doc2vec_test_data/doc2vec_2.model')

    # model = Doc2VecEvaluator('data/doc2vec_test_data/doc2vec_2.model')
    #
    # data = model.get_words('data/doc2vec_test_data/model.corpus')
    # print(data[1])
    # df = model.get_simiarity(model)
    # dm.save_csv(df, file_name='data/doc2vec_test_data/sim_matrix_2.csv', save_mode='w', encoding='utf-8')
    # print(model.most_similar(job_id=0, topn=10))
    # print(corpus.description[0])
    # df = pd.DataFrame(columns=['Job_ID', 'Job_Title'])
    # for i in range(model.size):
    #     # print(model.get_job_title('Job_ID_'+str(i)))
    #     df.loc[i] = ['Job_ID'+str(i), model.get_job_title('Job_ID_'+str(i))]
    # df.to_csv('data/doc2vec_test_data/job_title.csv', index=False)
    # model.visualize_jobs(type='tsne')

    # id = ['Job_ID_1851', 'Job_ID_2414', 'Job_ID_1842']
    # for name in id:
    #     print(model.get_job_title(name))

if __name__=='__main__':
    main()