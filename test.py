import json

with open('glassdoor_json_sample.json', 'r', encoding='utf-8') as f:
    file = json.loads(f.read())

gaTrackerData = file['gaTrackerData']
if 'adOrderId' in gaTrackerData:
    print('yes')
else:
    print('no')
