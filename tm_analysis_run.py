from tm_input import TMInput
from tm_analysis import LDABuilder, LDAModeler
import pandas as pd

def main():
    TMInput()
    builder = LDABuilder()
    topic_num = builder.getOptimalTopicNum()
    builder.num_topics = topic_num
    # builder.num_topics = 90
    builder.main()
    model = LDAModeler()
    for i in range(1):
        print(model.show_topic_words(i))
    # terms = [terms[0] for terms in model_doc2vec.show_topic_words(0)]
    # values = [terms[1] for terms in model_doc2vec.show_topic_words(0)]
    # print(values)
    #
    # df = pd.DataFrame({'terms':terms, 'values':values})
    # df.to_csv('data/doc2vec_test_data/0828/model_doc2vec/lda_value.csv', mode='w', encoding='utf-8')

    model.view_lda_model(model.model, model.corpus, model.dictionary)

    # model_doc2vec.show_document_topics()
if __name__ == '__main__':
    main()

