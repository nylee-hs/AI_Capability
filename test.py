import json
import pandas as pd


with open("data/doc2vec_test_data/0828/model/0828_manager_vis_result.json", "r") as st_json:
    st_python = json.load(st_json)
total = st_python['tinfo']['Total']
freq = st_python['tinfo']['Freq']
term = st_python['tinfo']['Term']
logprob = st_python['tinfo']['logprob']
loglift = st_python['tinfo']['loglift']

df = pd.DataFrame({'Term': term, 'Freq':freq, 'Total':total, 'logprob':logprob, 'loglift':loglift})
df.to_csv('data/doc2vec_test_data/0828/model_doc2vec/test_json2.csv', encoding='utf-8', mode='w')
