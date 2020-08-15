import spacy

texts = [['progr', 'OpenAPI'], ['API']]
nlp = spacy.load('en_core_web_sm')
for sent in texts:
    print(sent)
    doc = nlp(" ".join(sent))
    print(doc)
    print([token.lemma_ for token in doc])
    print([token.pos_ for token in doc])