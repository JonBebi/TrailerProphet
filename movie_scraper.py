from bs4 import BeautifulSoup
import requests

## Scrapes firstshowing.net for titles and release dates
def scraper(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content,'html.parser')
    body = soup.body
    div_wrapper = body.find('div',id='wrapper')
    sched = div_wrapper.find('div',class_='schedcontent')
    unclean_list = list(sched.find_all(['h4','p']))
    clean_list = [x.get_text().split('\n') for x in unclean_list][81:135]
    for dates in clean_list:
        for string in dates:
            if 'June' in string:
                string = string.replace('June ','2019-06-')
            if 'July' in string:
                string = string.replace('July ','2019-07-')
            if 'August' in string:
                string = string.replace('August ','2019-08-')
            if 'Semptember' in string:
                string = string.replace('September ','2019-09-')
            if '(Limited)' in string:
                string = string.replace('(Limited)','')
            if '(Friday)' in string:
                string = string.replace('(Friday)','')
            if '(Wednesday)' in string:
                string = string.replace('(Wednesday)','')
        print(clean_list)

scraper('https://www.firstshowing.net/schedule2019/')
