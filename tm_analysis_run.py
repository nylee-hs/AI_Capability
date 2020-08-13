from tm_input import TMInput
from tm_analysis import LDABuilder, LDAModeler

def main():
    TMInput()
    builder = LDABuilder()
    # topic_num = builder.getOptimalTopicNum()
    # if topic_num > 1 :
    #     builder.main()
    builder.num_topics = 10
    builder.main()
    model = LDAModeler()
    for i in range(10):
        print(model.show_topic_words(i))
    model.view_lda_model(model.model, model.corpus, model.dictionary)
if __name__ == '__main__':
    main()

