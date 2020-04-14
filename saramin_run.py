import requests
import json
from datetime import datetime

access_key = 'i1GQvCfzUfAFkFMCqVhWnuzlbkLZsUvZEsizPo2rzEkdvP6IoZ2RS'
query = '인공지능'
url = 'https://oapi.saramin.co.kr/job-search?access-key='+access_key


'''
keyword : 쿼리단어
published_min : 검색시작일(2020-03-20)
published_max : 검색종료일(2020-03-20)
'''
def getJobData(keyword):
    global url
    header = {'accept': 'application/json'}


    search_url = url + '&keyword='+keyword
    print(search_url)
    req = requests.get(search_url, header)
    print(req)
    if req.status_code == requests.codes.ok:
        data = json.loads(req.text)
        job_list = data['jobs']['job']

        company_list = [job['company']['detail']['name'] for job in job_list]
        print(company_list)


def main():
    getJobData(query)

if __name__ == '__main__':
    main()

