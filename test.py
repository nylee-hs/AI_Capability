import spacy
import pandas as pd


texts = [['Minimum years of work experience: 12 years experience']]
df = pd.DataFrame({'desc':texts})
print(df)
data = df['desc']
data = data.str.replace(pat=r'[^A-Za-z0-9]', repl= r' ', regex=True)
nlp = spacy.load('en_core_web_sm')
print(data)
for sent in texts:
    # print(sent)
    doc = nlp(" ".join(sent))
    # print(doc)
    print([token.lemma_ for token in doc])
    print([token.pos_ for token in doc])
# import matplotlib.pyplot as plt
# plt.plot(['a', 'b'], [1.0, 1.1], marker='o')
# plt.xlabel('topic')
# plt.ylabel('coherence value')
# plt.draw()