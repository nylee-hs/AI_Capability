import urllib.request
import urllib
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame,Series
from tqdm import tqdm
import requests
from datetime import datetime, timedelta
import os
import json

# LIMIT = 15
# directory = 'data\\indeed\\'  ## 수집 데이터 저장 폴더
# AGE = 1 ## 데이터 수집 기간(7일전까지)
# QUERY = 'artificial+intelligence'
# # URL = f"https://www.indeed.com/jobs?q=" + QUERY + "&limit={LIMIT}&sort=date&start="
# URL = f"https://www.indeed.com/jobs?as_and=artificial+intelligence&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=all&st=&as_src=&salary=&fromage={AGE}&limit={LIMIT}&sort=date&psf=advsrch&from=advancedsearch"

html_header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
               'accept-encoding': 'gzip, deflate, br',
               'cookie': 'CTK=1e3goal6b1fq3000; loctip=1; _gcl_au=1.1.695759262.1584331576; _ga=GA1.2.1671762697.1584331576; pjps=1; JCLK=1; hpnoproxy=1; RF="J0HMqjBXMIS7-jp7Y_exixEX9uvoa-6wBrjIcvshJb-kHka7arvzqwM8vijeii-fHKGT8nDzuyli2rJqZJsaAfX8byBw_Tems65ng9Pa8JX-CY_8q4F_XD9cRNRaLEijNBXerOn-ESAYklDAumKEIA=="; PREF="TM=1584429116502:L="; LC="co=US&hl=en"; RSJC=d2ed92024d429eb1; _gid=GA1.2.1148651147.1584693266; INDEED_CSRF_TOKEN=YYKX4H3Mk4r2dMpLhiwHICplLuZvnE2T; LV="LA=1584693643:LV=1584525858:CV=1584693386:TS=1584331576"; gonetap=2; SURF=t0yz1tjhPGhSvJYN6Z6DPU7dUbjOzRLf; JSESSIONID=83E2AF7E5537EEC1559A2A1F9AF39081.jasxB_sjc-job20; jasx_pool_id=5a51c9; jobAlertPopoverShown=1; ROJC=9c83e18de3b9d4f8:10c75165529d4156:d3c473040c07df53:7eb8f88edc8db970:29df5c8bc063734b:a12804a373c15b32:4fbbe2bd93bcdc12:b3bced9faac432cd:64534b8877b04ace:4e60c84279b7a054; _gali=resultsCol; _gat=1; JCLK=1; UD="LA=1584705483:CV=1584704496:TS=1584704496:SG=645475b75b78bb7c767d8604629ee7f0"; RQ="q=artificial+intelligence&l=&ts=1584705483771&pts=1584596647618:q=Artificial+Intelligence+Engineer&l=&ts=1584693643509:q=Artificial+Intelligence+Engineer+Specialist&l=&ts=1584693386467&pts=1584417158333:q=Director%2C+Software+Engineering+Search+Advertisement&l=&ts=1584430252655:q=artificial+intelligence&l=50&sort=date&ts=1584429048431:q=Software+Development+Engineer%2C+Alexa+Artificial+Intelligence&l=&ts=1584428996742:q=title%3A%28artificial+intelligence%29&l=&ts=1584332163423"; jaSerpCount=16; PPN=2',
          }

# print(URL)

def get_conf():
    with open('indeed_config.json', 'r', encoding='utf-8') as f:
        con_file = json.loads(f.read())

    return con_file

def resultCount():
    global html_header
    limit = get_conf()['CONFIGURE']['LIMIT']
    age = get_conf()['CONFIGURE']['AGE']
    url = f"https://www.indeed.com/jobs?as_and=artificial+intelligence&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=all&st=&as_src=&salary=&fromage={age}&limit={limit}&sort=date&psf=advsrch&from=advancedsearch"

    header = html_header
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    #     print(soup)
    #     pagination = soup.findAll('div.resultsTop > div.showing')
    pagination = soup.select('div#searchCountPages')[0].text.strip().split()[3].replace(',', '')
    pages = (int(pagination) // limit)
    lefts = (int(pagination) % limit)
    if lefts > 0:
        pages = pages + 1
    print("Total count = " + pagination)
    print("Total pages = " + str(pages))
    return pages


#     print(results)

def extract_job(start, last):
    directory = get_conf()['CONFIGURE']['DIRECTORY']
    age = get_conf()['CONFIGURE']['AGE']
    limit = get_conf()['CONFIGURE']['LIMIT']
    global html_header

    today = datetime.today()

    # for page in range(start, last):
    for page in range(start, last):
        print('PAGE NUM: ' + str(page))
        search_url = f'https://www.indeed.com/jobs?q=artificial+intelligence&limit={limit}&sort=date&filter=0&fromage='+str(age)+'&start=' + str(page * limit)
        print(' - URL: ' + search_url)
        html = requests.get(search_url, headers=html_header)
        # html = urllib.request.urlopen(search_url)
        try:
            if not os.path.exists(os.getcwd() + directory):
                os.makedirs(directory)
                print(' - New directory is created.')
        except OSError:

            print(' - Directory is existed.')

        with open(os.path.join(os.getcwd(), directory)+'doc.html', 'w', encoding='utf-8-sig') as f:
            f.write(html.text)

        html_file = open(os.path.join(os.getcwd(), directory)+'doc.html', 'r', encoding='utf-8-sig')
        html = html_file.read()
        html_file.close()

        soup = BeautifulSoup(html, 'html.parser')


        job_info = soup.findAll('h2', 'title')

        dates = soup.findAll('span', 'date')
        companies = soup.select('div.sjcl > div > span.company')

        info = zip(job_info, dates, companies)

        title_list = []
        date_list = []
        company_list = []
        company_employees_list = []
        company_industry_list = []
        company_revenues_list = []
        description_list = []
        today_list = []
        job_des_url_list = []

        # # print(job_info)
        for job, date, company in tqdm(zip(job_info, dates, companies), total=len(job_info), desc=' -- GET DATA'):
            try:
                job_des_url = job.a['href']
                return_text = get_details(job_des_url=job_des_url)
                title = return_text[0]
                title_list.append(title)
                c_info = []
                try:
                    company_url = company.a['href']
                    c_info = get_company_info(company_url)
                except:
                    c_info = ['NA', 'NA', 'NA']
                company = company.text.strip()
                company_list.append(company)

                company_employees_list.append(c_info[0])
                company_industry_list.append(c_info[1])
                company_revenues_list.append(c_info[2])
                job_des_url_list.append(job_des_url)
                description = return_text[1]
                description_list.append(description)
                date = date.text
                date_list.append(date)
                today_list.append(today)
                # print(title, " || ", "https://www.indeed.com" + job_des_url)

            except Exception as ex:
                print('Error!', title, ex)
                title_list.append('NA')
                company_list.append('NA')
                job_des_url_list.append('NA')
                description_list.append('NA')
                date_list.append('NA')
                today_list.append(today)
                company_employees_list.append('NA')
                company_industry_list.append('NA')
                company_revenues_list.append('NA')

                pass

        #             if(title == 'Artificial Intelligence Engineer (Specialist)' and company == 'Fulcrum'):
        #                 print(title, company, description)

        df = pd.DataFrame({'job_title': title_list, 'company': company_list, 'url': job_des_url_list, 'job_description': description_list, 'published_date': date_list, 'scrap_date': today_list, 'company_employees': company_employees_list,
                           'company_industry': company_industry_list, 'company_revenue': company_revenues_list})
        save_csv(df, os.path.join(os.getcwd(), directory) + datetime.today().strftime("%Y%m%d")+"_indeed_data.csv")


    return df


def get_company_info(company_url): ## 기업 정보 받아오기
    html = urllib.request.urlopen('https://indeed.com'+company_url)
    soup = BeautifulSoup(html, 'html.parser')

    ## 기업 정보 페이지에 정보 디스플레이 화면이 2가지가 있어 종류에 따라 csize_a와 csize_b로 수집한 후 최종적으로 csize 변수에 저장
    c_info = []
    com_info_b1 = soup.select('div.cmp-AboutMetadata-itemTitle')
    com_info_b2 = soup.select('div.cmp-AboutMetadata-itemCotent')
    com_info_a = soup.select('div.cmp-AboutSection-grid')

    if(len(com_info_a) >= 1):

        csize = soup.select('div.cmp-FormattedEmployeeRange')[0].text
        cindustry = soup.select('div.cmp-AboutSection-gridText--small')[0].text
        crevenue = soup.select('div.cmp-FormattedRevenueRange')[0].text
        c_info.append(csize)
        c_info.append(cindustry)
        c_info.append(crevenue)
    elif(len(com_info_a) < 1 and len(com_info_b1) >= 1):
        temp1 = [item.text for item in com_info_b1]
        temp2 = [item.text for item in com_info_b2]
        dic_com = dict(zip(temp1, temp2))


        # if dic_com.get('Headquarters'):
        #     c_info.append(dic_com['Headquarters'])
        # else:
        #     c_info.append('NA')
        if dic_com.get('Employees'):
            c_info.append(dic_com['Employees'])
        else:
            c_info.append('NA')
        if dic_com.get('Industry'):
            c_info.append(dic_com['Industry'])
        else:
            c_info.append('NA')
        if dic_com.get('Revenue'):
            c_info.append(dic_com['Revenue'])
        else:
            c_info.append('NA')

    else:
        csize = 'NA'
        cindustry = 'NA'
        crevenue = 'NA'
        c_info.append(csize)
        c_info.append(cindustry)
        c_info.append(crevenue)

    return c_info




def get_details(job_des_url):
    html = urllib.request.urlopen('https://www.indeed.com' + job_des_url)
    soup = BeautifulSoup(html, 'html.parser')

    # description 받아오기
    description = soup.select('div.jobsearch-jobDescriptionText')
    return_text = ''
    if len(description) is 0:  # 일부 페이지의 경우 상세페이지의 구조가 다르게 나타남.
        description = soup.select('div.jobDetailDescription')
        # print(description)
        for t in description:
            return_text = return_text + t.text
    else:
        return_text = description[0].text

    # job title 받아오기
    title = soup.select('h3.jobsearch-JobInfoHeader-title')

    return_list = [title[0].text, return_text]

    return return_list


def load_csv(file):
    csv_data = pd.read_csv(file, encoding='utf-8-sig')
    return csv_data


def save_csv(dataFrame, file_name):
    if not os.path.exists(file_name):
        dataFrame.to_csv(file_name, index=False, mode='w', encoding='utf-8-sig')
    else:
        dataFrame.to_csv(file_name, index=False, mode='a', encoding='utf-8-sig', header=False)
    print(' ** File saved.\n')


# def html_file_check(filename='indeed_data_check.csv'):
#     today = datetime.today()
#     html_file = open('doc.html', 'rb')
#     html = html_file.read()
#     html_file.close()
#
#     soup = BeautifulSoup(html, 'html.parser')
#
#     job_info = soup.findAll('div', 'title')
#     dates = soup.findAll('span', 'date')
#     companies = soup.select('div.sjcl > div > span.company')
#
#     info = zip(job_info, dates, companies)
#
#     title_list = []
#     date_list = []
#     company_list = []
#     description_list = []
#     today_list = []
#     job_des_url_list = []
#
#     for job, date, company in tqdm(info, desc="job(" + str("0") + ") : "):
#         try:
#             title = job.a['title']
#             title_list.append(title)
#             company = company.text.strip()
#             company_list.append(company)
#             job_des_url = job.a['href']
#             job_des_url_list.append(job_des_url)
#             description = get_details(job_des_url=job_des_url)
#             description_list.append(description)
#             date = date.text
#             date_list.append(date)
#             today_list.append(today)
#             print(title, " || ", 'https://www.indeed.com' + job_des_url)
#         except:
#             print('Error occurred!')
#             title_list.append('NaN')
#             company_list.append('NaN')
#             job_des_url_list.append('NaN')
#             description_list.append('NaN')
#             date_list.append('NaN')
#             today_list.append(today)
#             pass

    #             if(title == 'Artificial Intelligence Engineer (Specialist)' and company == 'Fulcrum'):
    #                 print(title, company, description)

    # df = pd.DataFrame({'job_title': title_list, 'company': company_list, 'url': job_des_url_list, 'job_description': description_list, 'published_date': date_list, 'scrap_date': today_list})
#     # save_csv(df, "indeed_data_check.csv")
#     # return df


def load_csv(file_name):
    col_names = ['job_title', 'company', 'url', 'job_description', 'published_date', 'scrap_date', 'company_employees', 'company_industry', 'company_revenue']
    df = pd.read_csv(file_name, delimiter=',', names=col_names, encoding='utf-8-sig')

    return df


def main():
    pages = resultCount()
    extract_job(0, pages)

    # print(get_conf()['CONFIGURE']['AGE'])
    # get_details('/rc/clk?jk=f4837c61ead9b1ea&fccid=2525cc4a9a704809&vjs=3')
    # return df
    # print(get_company_info('https://www.indeed.com/cmp/UBS?from=SERP&fromjk=a92d0d0ba1e8396a&jcid=1c76c3a36f6c7557&attributionid=serp-linkcompanyname'))
    # print(get_company_info('https://www.indeed.com/cmp/Woods-Hole-Oceanographic-Institution'))

if __name__ == '__main__':
    main()