from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os
import csv
import random

# QUERY = 'artificial-intelligence'
# URL = 'https://www.glassdoor.com/Job/' + QUERY + '-jobs-SRCH_KE0,23_IP'
# AGE = 1  ## 검색 기간 설정 시 값. 숫자는 일(day)을 의미
# directory = 'data\\glassdoor\\'  ## 수집 데이터 저장 폴더
json_header = {
    'referer': 'https://www.glassdoor.com/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
    'authority': 'www.glassdoor.com',
    'accept': 'application/json',
    'content-type': 'pplication/x-www-form-urlencoded; charset=UTF-8',
    'scheme': 'https',
    'accept-encoding': 'gzip, deflate, br',
    'method': 'GET',
    'sec-fetch-mode': 'cors',
    'cookie': 'JSESSIONID_JX_APP=2D1EC1C7395C086DB528258A2007FA07; gdId=c773f00c-71b5-41d6-a85a-bc75bed9f4b0; _ga=GA1.2.1635570328.1584540161; _gid=GA1.2.2112870736.1584540161; _gcl_au=1.1.828113750.1584540161; _fbp=fb.1.1584540161523.1150209086; G_ENABLED_IDPS=google; __qca=P0-1124285797-1584540161527; trs=INVALID:SEO:SEO:2020-03-18+02%3A01%3A29.617:undefined:undefined; _mibhv=anon-1584540328507-3654771112_6890; _micpn=esp:-1::1584540328507; ki_r=aHR0cDovL2xvY2FsaG9zdDo4ODg4L25vdGVib29rcy9Xb3JrbmV0L2dsYXNzZG9vci5pcHluYg%3D%3D; _delighted_fst=1584540912464:{}; __cf_bm=f97a938d3ffe5526acb7c157ef2a6c72029a8028-1584541063-1800-AbooNsesLBbMuCNHpNfvVC4N/JF9MqqsXvgAPS5EzYiPXvtn4iC9FmjcL5aLNXRGV/8yOXkzf6cxtA0r/1IEXqY=; uc=8F0D0CFA50133D96DAB3D34ABA1B8733072742F44A7116C7E100CAD6A034A8FA5600BB2954B3B4CC840833593E9B70BF7E52D4D059614985C624A54C44935743CEC2AF80698474928745941C9D6B1AB8E5F9F2BB9F466DE8723950C859955DE34917FDD37CD8254871B7CBBB889E29DE7DFCCEDD329AA6FB9D4FBD96E013055B525D3BC38C30AC0AEE2AEF3A06AFDA21454A0D3BDE9F15BC2A1F6ECE99240EC5; JSESSIONID=4B17708F9173882910F7329DC7761C54; GSESSIONID=4B17708F9173882910F7329DC7761C54; ht=%7B%22quantcast%22%3A%5B%22D%22%5D%7D; __gads=ID=4cfb113b7fb9bf1d:T=1584541137:S=ALNI_MYYWl0qWDFzhK75LMWsuDpZJ85b3g; G_AUTHUSER_H=0; ki_t=1584540560606%3B1584540560606%3B1584541800793%3B1%3B2; AWSALB=Qb4YLHbfXiC80972P9Wk5/3UebWD6F0qJGjyFTtyIUp8RLUVC7LhzaCjxoAMtf93MWqZaIvcD/Mui+tJiH7EZFQ6v73dZzPsvpj2XaPoyr//3LCt5p2yBGVzFq+pChXTdd1K9o0DP6BpG9TfyCMS5U8QKopJaIg6hmGMNKj26h8SVnYAw+zbDrtTLntyBA==; AWSALBCORS=Qb4YLHbfXiC80972P9Wk5/3UebWD6F0qJGjyFTtyIUp8RLUVC7LhzaCjxoAMtf93MWqZaIvcD/Mui+tJiH7EZFQ6v73dZzPsvpj2XaPoyr//3LCt5p2yBGVzFq+pChXTdd1K9o0DP6BpG9TfyCMS5U8QKopJaIg6hmGMNKj26h8SVnYAw+zbDrtTLntyBA==; cass=2',
    'sec-fetch-site': 'same-origin',
    'x-requested-with': 'XMLHttpRequest'
}

html_header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
               'accept-encoding': 'gzip, deflate, br',
               'cookie': 'gdId=8b0fdea5-3431-47c1-a465-0e5085c61c2f; trs=https%3A%2F%2Fwww.google.com%2F:SEO:SEO:2020-03-18+02%3A01%3A29.617:undefined:undefined; _gid=GA1.2.1132903540.1584522093; _ga=GA1.2.541850168.1584522093; G_ENABLED_IDPS=google; _gcl_au=1.1.1634337357.1584522095; _fbp=fb.1.1584522095130.1484441351; __gads=ID=4f045facc6dce7a8:T=1584522095:S=ALNI_MZrXw5W2Zw6QxTQXIj-UW-nC8e1dQ; ht=%7B%22quantcast%22%3A%5B%22D%22%5D%7D; __qca=P0-693164517-1584522095099; onboardingV2=false; _mibhv=anon-1584522271568-2402893593_6890; _micpn=esp:-1::1584522271568; _delighted_fst=1584522295483:{}; ignoreCount=0; _mkto_trk=id:899-LOT-464&token:_mch-glassdoor.com-1584522630411-85228; known_by_marketo_email=nylee@hs.ac.kr; _gd_visitor=3b631a70-f10d-4c29-8fe0-15e5fdb4d908; _cs_c=1; _cs_id=36341baf-8cb5-a322-aaa8-e53c84be3c4c.1584522630.1.1584522630.1584522630.1.1618686630899.Lax.0; _biz_uid=a955a7558c60491ef0437ec17cdfb831; _biz_nA=2; _gd_svisitor=767a4668fc3d000086e5715e810300000cc20100; driftt_aid=2f515ee5-c324-4203-97a2-969f7f07bd64; _biz_pendingA=%5B%5D; _hjid=434a59e7-d618-495a-958c-9eff79d11594; _biz_flagsA=%7B%22Version%22%3A1%2C%22Mkto%22%3A%221%22%2C%22XDomain%22%3A%221%22%2C%22ViewThrough%22%3A%221%22%7D; _hjIncludedInSample=1; DFTT_END_USER_PREV_BOOTSTRAPPED=true; G_AUTHUSER_H=0; GSESSIONID=AABFA1E83B3A519798A99E0B43A13536; mp_1faab7db770246ce2e76cc5a52005587_mixpanel=%7B%22distinct_id%22%3A%20%22170f0f59cab3d-0135919eb5a22b-4313f6a-384000-170f0f59cac785%22%2C%22%24device_id%22%3A%20%22170f0f59cab3d-0135919eb5a22b-4313f6a-384000-170f0f59cac785%22%2C%22%24initial_referrer%22%3A%20%22https%3A%2F%2Fwww.glassdoor.com%2F%22%2C%22%24initial_referring_domain%22%3A%20%22www.glassdoor.com%22%2C%22query_id%22%3A%20%22936d9267-6020-459a-bdce-6a0285b7b359%22%2C%22user_id%22%3A%20null%2C%22user_type%22%3A%20null%2C%22user_name%22%3A%20null%2C%22domain%22%3A%20%22help.glassdoor.com%22%2C%22client_id%22%3A%20342%2C%22session_id%22%3A%20%223e38858f-673d-4876-be82-1eaacbc0e88c%22%2C%22ab_treatment%22%3A%201%2C%22experiment%22%3A%20null%7D; JSESSIONID_JX_APP=51BBDDB9C448CF269FF83CFF637DCC7E; __cf_bm=f78bbd864435680775b6c94e776dc699ed873a48-1584598856-1800-Ad5nyri/Ya6Qwf30923bYlYddd1s30m9mDxozj9x0OzFQ/AhFZP/dUlLqxbrAxly7i5B2iVRvxc1jevW4Gosb3E=; JSESSIONID=E876F7D493C530BCFA8086C0A599FDFE; _uac=00000170f1750eb9a0a0d87867f85b97; uc=8F0D0CFA50133D96DAB3D34ABA1B8733072742F44A7116C7E100CAD6A034A8FA5600BB2954B3B4CC840833593E9B70BF7E52D4D059614985C624A54C44935743CEC2AF80698474928745941C9D6B1AB8E5F9F2BB9F466DE8723950C859955DE34917FDD37CD8254871B7CBBB889E29DE7DFCCEDD329AA6FB9D4FBD96E013055B525D3BC38C30AC0AEE2AEF3A06AFDA21454A0D3BDE9F15BC617E3DC42232BFA3E4B1CA9754A71BBE9C35B04AF9E9EAF43DD0475C21A1DA9659E92D08ABB5402C7D676CFFCBD2DE02; ki_t=1584526808605%3B1584599154892%3B1584599154892%3B2%3B3; ki_r=; cass=2; at=7j8SuWu7clmr5Fz39fcCgpHP4dTQqn5-HKYhwEKPngfMAExD1syxi2s0m4yMrhLkHq2SiSfqGwfWuVj2bneR6vxDl4EMfyR5xkTct9s2GvyRmNWpeioowSJ24H3SngRv7R73A9-B-wj6V_id-2kcOySnPv32VJan0XoGqGfLKHJfB3ZRhWRbZM7YAlCMD8hszT9GtkMzjdsQWQAP4w-SiC_7IcvCYWfh7LmvjesytAY2BFTsmNZ_BssClt8_WylC0G-zxpAltHnWRtzpJxP1bS1uSyPMqZy1a30MvV7SqA8iAuUvEXc4vTxS7EcSpc5smWZIxX2zujKztdz_awvXQDL2bv0zzDWdIbWYH6mWNpWD8kVP8ZD289dSo5fZZyD9It9cWLHpRGXSiwMP2cvpvgruBwvh8nEOLLBXGH3S3OjTnKGu9Nq-pJ4A1UPfuVq3NRjqG0T6XeYrlbSC9VVZDf6thZTutxkfxM7KIgOrbQOkk6kEPvpTL3v3xWLoqcbal63cyfM8C64ILnbvMoeVsVFfSX3PISUTv8_79UBKtSJJOI4LesCJZoxo92dDIv2BUKvzkW_lThxuVvlLjAuMaGKbvaMoZ2kVnuskXABXUQplg1GmtqmqUPhNjA4yWlXN6AWCRxTduLataq19CSQ50XcdPd52Wggu87iM8NZCIM_0yQQlkkYLI4aI21gm7l7_zUW9meWpjpy8_zEcsgv8I1Ai7bAA6uy2JAj9ZO638jNnIaKrED5qwPgNexPnREnFc4QUBI6F1QqgxYiU0yHoCPl5OZKbxWuPpLaBT_8ObR_BTrOFSGyVXbYaBANqMR7LQOw4XHAAkyHFWUl7copL9x4jcVmzSVuZ7pNFSmqUUuw8R_FiWbPmug; _dc_gtm_UA-2595786-1=1; AWSALB=NuUEhLPE2EqbLi4bQX9cHY2W+6raSNtdFvKAJZQI+0Oq9EdJTi33GXvCYNSN+kRGoUuDJ7ng7Cn2I0J/gR794ayy3oBp7Y/i8k7Gpemch/YmmbSGdQaEE1TWLotJ2h2ZeFWyhb5N7CHR+YnqGlu2QGT8dF0B7Wni0qKvaqBLQ1/kLLtSPLyDpvye/SCkPaQCpkRCzcP+wyVESYouSuXe0hAJIs5G1lgXnCF0kPvJ+dzWQ00FWEX4tUb0La/6pn8Fo7GTnuUP5viGl+TQ1JxVYNwJKx2jhLD3qoFGU7jJyMMKAKpamawpMARiv7r1J8kA; AWSALBCORS=NuUEhLPE2EqbLi4bQX9cHY2W+6raSNtdFvKAJZQI+0Oq9EdJTi33GXvCYNSN+kRGoUuDJ7ng7Cn2I0J/gR794ayy3oBp7Y/i8k7Gpemch/YmmbSGdQaEE1TWLotJ2h2ZeFWyhb5N7CHR+YnqGlu2QGT8dF0B7Wni0qKvaqBLQ1/kLLtSPLyDpvye/SCkPaQCpkRCzcP+wyVESYouSuXe0hAJIs5G1lgXnCF0kPvJ+dzWQ00FWEX4tUb0La/6pn8Fo7GTnuUP5viGl+TQ1JxVYNwJKx2jhLD3qoFGU7jJyMMKAKpamawpMARiv7r1J8kA; _gat_UA-2595786-1=1', }


def get_conf():
    with open('glassdoor_config.json', 'r', encoding='utf-8') as f:
        con_file = json.loads(f.read())

    return con_file

def get_position_link(url, returnType):
    global html_header
    links = []
    json_links = []
    returnTypes = []
    header = html_header
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')

    a = soup.select('div.jobContainer > div.jobHeader > a.jobLink')

    for i in a:
        #         print('https://www.glassdoor.com'+ i.get('href'))
        raw_link = i.get('href')
        links.append('https://www.glassdoor.com' + raw_link)

        json_str = raw_link.replace('/partner/jobListing.htm?', '')
        #         print('https://www.glassdoor.com/Job/json/details.htm?'+json_str)
        json_links.append('https://www.glassdoor.com/Job/json/details.htm?' + json_str)
        returnTypes.append(returnType)

    returnData = []
    if returnType == 'html':
        returnData = links
    elif returnType == 'json':
        returnData = json_links

    return returnData, returnTypes


def pagination():
    global html_header
    guery = get_conf()['CONFIGURE']['QUERY']
    url = 'https://www.glassdoor.com/Job/' + guery + '-jobs-SRCH_KE0,23_IP'
    age = get_conf()['CONFIGURE']['AGE']

    header = html_header
    response = requests.get(url + '.htm?fromAge=' + str(age), headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    print('Response Code : ' + str(response.status_code))
    page_tag = soup.find_all('div', {'cell middle hideMob padVertSm'})
    page = page_tag[0].text.split()
    total_pages = page[len(page) - 1]

    if int(total_pages) > 30:
        total_pages = 30
        print('Total pages : {}. (more 30+)'.format(total_pages))
    #     print(URL+'.htm?fromAge='+str(AGE))
    else:
        print('Total pages : {}.'.format(total_pages))
    return total_pages


def get_all_links(totalpages, returnType):
    # global AGE
    age = get_conf()['CONFIGURE']['AGE']
    directory = get_conf()['CONFIGURE']['DIRECTORY']
    guery = get_conf()['CONFIGURE']['QUERY']
    url = 'https://www.glassdoor.com/Job/' + guery + '-jobs-SRCH_KE0,23_IP'
    # global directory
    links = []
    returnTypes = []

    print('Collecting links....')
    totalpages = int(totalpages)

    for pagenum in tqdm(range(totalpages), desc=' LINK GATHERING'):
        try:
            # print('PAGE NUM: ' + str(pagenum))
            url_main = url + str(pagenum+1) + '.htm?fromAge=' + str(age) + '&sortBy=date_desc'  ## 검색 기간 설정 시 htm?fromAge= 값을 변경할 것. 뒤에 숫자는 몇일 전을 의미
            # print(' - URL: '+ url_main)
            link, returnt = get_position_link(url_main, returnType)
            # print(link)
            #             print(returnt)
            links.extend(link)
            returnTypes.extend(returnt)
            time.sleep(round(random.uniform(0.5, 1.0), 1))
        except:
            print('No more pages found.')

    print(' - Current Links : ' + str(len(links)) + ' | Types : ' + str(returnTypes[0]))
    df = pd.DataFrame({'link': links, 'type': returnTypes})

    try:
        if not os.path.exists(os.getcwd() + directory):
            os.makedirs(directory)
            print(' - Creating directory is completed.')
    except OSError:
        print(' - Directory is already existed. | ' + directory)

    save_csv(df, os.path.join(os.getcwd(), directory) + datetime.today().strftime("%Y%m%d") + '_glassdoor_links.csv')


#     return links, returnt


def scrap_job_page(link, returnType):
    return_data = []
    if returnType == 'html':
        return_data = scrap_job_page_html(link)
    elif returnType == 'json':
        return_data = scrap_job_page_json(link)
        time.sleep(round(random.uniform(0.5, 1.0), 1))
    return return_data


def scrap_job_page_json(link):
    return_list = []
    req = requests.get(link, headers=json_header)
    if req.status_code == requests.codes.ok:
        data = json.loads(req.text)
        gaTrackerData = data['gaTrackerData']
        header = data['header']

        if 'jobTitle' in gaTrackerData:
            jobTitle = gaTrackerData['jobTitle']
            return_list.append(jobTitle)
        else:
            return_list.append('NaN')

        if 'employerName' in header:
            company = header['employerName']
            return_list.append(company)
        else:
            return_list.append('NaN')

        url = 'JSON: ' + link
        return_list.append(url)

        job = data['job']
        description = job['description']
        return_list.append(description)

        pub_date = header['posted']
        return_list.append(pub_date)

        today = datetime.today()
        return_list.append(today)

        if 'empSize' in gaTrackerData:
            csize = gaTrackerData['empSize']
            return_list.append(csize)
        else:
            return_list.append('NaN')

        if 'industry' in gaTrackerData:
            cindustry = gaTrackerData['industry']
            return_list.append(cindustry)
        else:
            return_list.append('NaN')

        if 'salaryHigh' in data['header']:
            salary_high = data['header']['salaryHigh']
            return_list.append(salary_high)
        else:
            return_list.append('NaN')

        if 'salaryLow' in data['header']:
            salary_low = data['header']['salaryLow']
            return_list.append(salary_low)
        else:
            return_list.append('NaN')

        if 'revenue' in data['overview']:
            crevenue = data['overview']['revenue']
            return_list.append(crevenue)
        else:
            return_list.append('NaN')

        if 'type' in data['overview']:
            ctype = data['overview']['type']
            return_list.append(ctype)
        else:
            return_list.append('NaN')

        time.sleep(round(random.uniform(0.5, 1.0), 1))

    else:
        print(" - JSON link has some problems.")
        print(" - link = " + link)

    return return_list


def scrap_job_page_html(url):
    dic = {}
    header = html_header
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    body = soup.find('body')
    #     print(title)

    try:
        #         job title
        print(body.find('h2', class_='mt-0 margBotXs strong').text.strip())
        dic['job_title'] = body.find('h2', class_='mt-0 margBotXs strong').text.strip()
    except:
        print('NaN')
        dic['job_title'] = np.nan

    try:
        # company name
        dic['company_name'] = body.find('span', class_='strong ib').text.strip()
    except:
        dic['company_name'] = np.nan


def save_csv(dataFrame, file_name):
    if not os.path.exists(file_name):
        dataFrame.to_csv(file_name, index=False, mode='w', encoding='utf-8-sig', header=True)
    else:
        dataFrame.to_csv(file_name, index=False, mode='a', encoding='utf-8-sig', header=False)


    if len(file_name.split('_')) == 6 :
        print(' ** LINK 파일 저장 완료'+'\n')
    else:
        print(' ** DATA 파일 저장 완료.')



def load_csv(file_name, col_names):
    # data파일일 경우 :    col_names=['job_title', 'company', 'url', 'job_description', 'published_date', 'scrap_date']
    # link 파일일 경우 :   col_names=['link', 'returnType']
    df = pd.read_csv(file_name, delimiter=',', names=col_names, encoding='utf-8-sig')

    return df

def main():
    directory = get_conf()['CONFIGURE']['DIRECTORY']

    total_page = pagination()  # 전체 페이지수 받아오기

    get_all_links(total_page, 'json') #개별 링크 데이터 수집 후 links.csv로 저장
    df = load_csv(os.path.join(os.getcwd(), directory)+datetime.today().strftime("%Y%m%d")+'_glassdoor_links.csv', col_names=['link', 'returnType'])
    returnType = df.iloc[1]['returnType']
    links = list(df['link'])[1:]
    print('Link Data Return Type : ' + returnType + '\n')

    title_list = []
    company_list = []
    job_des_url_list = []
    description_list = []
    date_list = []
    today_list = []
    csize_list = []
    cindustry_list = []
    salary_high_list = []
    salary_low_list = []
    crevenue_list = []
    ctype_list = []


    time.sleep(0.5)
    print('Collecting Job Data.... ')
    time.sleep(0.5)
    for link in tqdm(links, total=len(links), desc='  GET DATA'):
        # print(link)
        try:
            return_data = scrap_job_page(link, returnType)
            title_list.append(return_data[0])
            company_list.append(return_data[1])
            job_des_url_list.append(return_data[2])
            description_list.append(return_data[3])
            date_list.append(return_data[4])
            today_list.append(return_data[5])
            csize_list.append(return_data[6])
            cindustry_list.append(return_data[7])
            salary_high_list.append(return_data[8])
            salary_low_list.append(return_data[9])
            crevenue_list.append(return_data[10])
            ctype_list.append(return_data[11])
        except:
            title_list.append('NaN')
            company_list.append('NaN')
            job_des_url_list.append('NaN')
            description_list.append('NaN')
            date_list.append('NaN')
            today_list.append('NaN')
            csize_list.append('NaN')
            cindustry_list.append('NaN')
            salary_high_list.append('NaN')
            salary_low_list.append('NaN')
            crevenue_list.append('NaN')
            ctype_list.append('NaN')
            pass

    df = pd.DataFrame({'job_title': title_list, 'company': company_list, 'url':job_des_url_list, 'job_description':description_list, 'published_date':date_list, 'scrap_date':today_list, 'company_size':csize_list,
                       'salary_high':salary_high_list, 'salary_low':salary_low_list, 'revenue':crevenue_list, 'company_type':ctype_list})

    save_csv(df, directory + datetime.today().strftime("%Y%m%d") + '_glassdoor.csv')


if __name__ == '__main__':
    main()

