from bs4 import BeautifulSoup
import requests

## Scrapes firstshowing.net for titles and release dates
def scraper(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content,'html.parser') ## Make soup
    body = soup.body ## Start with the body of the HTML
    div_wrapper = body.find('div',id='wrapper') ## Find the child div tag with id=wrapper
    sched = div_wrapper.find('div',class_='schedcontent') ## Find the child div in schedcontent class
    unclean_list = sched.find_all(['h4','p']) ## List of the dates (h4 tags) and movies (p tage)
    clean_list = [x.get_text().split('\n') for x in unclean_list][81:135] ## Slice above list from June to Sept

    ## Loop through the list and change the format of the dates
    restrictions = ['2019-06-2019','2019-07-2019','2019-08-2019''2019-09-2019']
    for l in clean_list:
        for i,string in enumerate(l):
            if 'June' in string:
                string = string.replace('June ','2019-06-')
                l[i] = string
                next
            elif 'July' in string:
                string = string.replace('July ','2019-07-')
                l[i] = string
                next
            elif 'August' in string:
                string = string.replace('August ','2019-08-')
                l[i] = string
                next
            elif 'September' in string:
                string = string.replace('September ','2019-09-')
                l[i] = string
                next
        else:
            if '(Limited)' in string:
                string = string.replace(' (Limited)','')
                l[i] = string
                next
            if '(Friday)' in string:
                string = string.replace(' (Friday)','')
                l[i] = string
                next
            if '(Wednesday)' in string:
                string = string.replace(' (Wednesday)','')
                l[i] = string
                next
            if string in restrictions:
                l.remove(string)
    return clean_list

test = scraper('https://www.firstshowing.net/schedule2019/')
test

def clean_list(ls):
    clean_l = []
    for i in range(len(ls)):
        try:
            if len(ls[i+1]) < 2:
                continue
        except:
            if len(ls[i]) < 2 and '2019' in ls[i]:
                ls[i].extend(ls[i+1])
        else:
            if len(ls[i+1]) < 2 and '2019' in ls[i+1]:
                clean_l.extend(ls[i])
        if
    return clean_l

clean_list(test)

ls1 = []
ls2 = []
for i in range(len(test)):
    if test[i] < 2 and '2019' in test[i]:
        ls1.append(test[i])
    if test[i+1] = True and '2019' in test[i+i]:
        ls1.append(test[i])
