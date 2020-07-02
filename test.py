from visualize_utils import visualize_between_words, visualize_words


words = ['a', 'b', 'c', 'd']
a = [2.32269332e-01, -5.67735024e-02,  8.26407373e-02, -3.22979316e-03]

visualize_words(words=words, vecs=a, palette="Viridis256", filename="/notebooks/embedding/words.png",
                    use_notebook=False)
