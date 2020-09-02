import requests
from bs4 import BeautifulSoup
from fp.fp import FreeProxy
import re
import time
from selenium import webdriver
import os
import pandas as pd
import random
proxy = FreeProxy(country_id=['US', 'BR'],rand=True).get()
#print(proxy)
proxies = {}

# store exceptions
wiley_Exceptions = 0

# Increments count for unknown pdf
PDF_icre = 0
Abstracts = []
Pdf_Names = []
proxies['http'] = proxy
proxies['https'] = proxy
#print(proxies)
inputstr = 'body-centric wireless communications'
#Search_Res = requests.get("https://www.google.com/search?q=" + inputstr)
Folder_Name = inputstr.replace(' ', '_')
print(inputstr)
print(Folder_Name)
if os.path.isdir('./' + Folder_Name):
    pass
else:
    os.mkdir(Folder_Name)
Path = os.getcwd()
#driver = webdriver.Chrome(Path+"/chromedriver")

for page in range(50):
    time.sleep(random.randint(65,100))
    Strt_Page = str(page * 10)
    site = "https://scholar.google.com/scholar?start="+Strt_Page+"&q=" + inputstr
    # print(site)
    Search_Res = requests.get("https://scholar.google.com/scholar?start="+Strt_Page+"&q=" + inputstr)
    soup = BeautifulSoup(Search_Res.text,'html.parser')

    # - Same code using Selinium - #
    # driver = webdriver.Chrome(Path+"/chromedriver")
    #
    # driver.get("https://scholar.google.com/scholar?start="+Strt_Page+"&q=" + inputstr)
    # time.sleep(15)
    # Search_Res = driver.page_source
    # soup = BeautifulSoup(Search_Res, 'html.parser')

    #print(soup.prettify())

    results =  soup.select('a')
    # driver.close()
    #print(results)
    Links = []
    IEEE_pdf_links = []
    PDf_links = []
    for link in results:

        #print(link.get('href'))
        s = link.get('href')
        if s.startswith('https://') == True and s.startswith('https://accounts.google.com/')== False:
            if s.endswith('.pdf') == True and (s.startswith('http://ieeexplore.ieee.org') or s.startswith('https://ieeexplore.ieee.org')) :
                IEEE_pdf_links.append(s)
            elif s.endswith('.pdf') or (s.find('pdf') > 1 and s.startswith('https://onlinelibrary.wiley.com') == False):

                PDf_links.append(s)
            elif s.startswith('https://scholar.googleusercontent.com') == False and s.startswith('https://ieeexplore.ieee.org/abstract/') == False:

                Links.append(s)


    print(IEEE_pdf_links)
    print(PDf_links)
    print(Links)




    import urllib
    import urllib.request

    #- IEEE pdfs download - #
    for links in IEEE_pdf_links:
        links = links.replace('/',' ')
        links = links.replace('.pdf', '')
        words = links.split()
        #print(str(int(words[-1])))
        ID = str(int(words[-1]))
        command = 'wget "http://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&isnumber=&arnumber=variant" -O mypaper.pdf'
        command = command.replace('variant',ID)
        command = command.replace('mypaper', Folder_Name+'/Paper_'+ID)
        #print(command)

        files = os.system(command)
        time.sleep(random.randint(1,6))
    # - Other pdf download - #
    for links in PDf_links:
        Name = links.replace('/', ' ')
        Name = Name.replace('.pdf', '')
        Name = Name.replace('pdf', '')

        words = Name.split()
        #print(str(words[-1]))
        ID = str(words[-1])
        if ID =='' or ID == ' ':
            PDF_icre += 1
            ID = str(PDF_icre)

        command = 'wget' +' '+'"'+ links +'"'+ ' '+'-O mypaper.pdf'
        command = command.replace('mypaper', Folder_Name + '/Paper_' + ID)
        #print(command)

        files = os.system(command)
        time.sleep(random.randint(1, 6))
    #driver = webdriver.Chrome("/Users/abulhasan/Documents/Chuha_Proj/chromedriver")

    for links in Links:
        Name = links.replace('/', ' ')
        Name = Name.replace('.pdf', '')
        words = Name.split()
        #print(str(words[-1]))
        ID = str(words[-1])

        # - wiley Files - #
        if links.startswith('https://onlinelibrary.wiley.com'):
            if links.find('abs') > 1 :
                ###########
                driver = webdriver.Chrome(Path+"/chromedriver")
                #print(links)
                driver.get(links)
                time.sleep(18)
                content = driver.page_source
                soup = BeautifulSoup(content)
                # print(soup.prettify())
                try:
                    results =  soup.find(class_='article-section__content en main').text
                except:
                    wiley_Exceptions +=1
                Abstracts.append(results)
                Pdf_Names.append(ID)
                driver.close()

        # - Sciencedirect - #
        if links.startswith('https://www.sciencedirect.com'):
            driver = webdriver.Chrome(Path+"/chromedriver")
            #print(links)
            driver.get(links)
            time.sleep(random.randint(1, 6))
            content = driver.page_source
            soup = BeautifulSoup(content)
            # print(soup.prettify())
            try:
                results =  soup.find(class_='abstract author').text
                results = results.replace('Abstract','')

                if results not in Abstracts:
                    Abstracts.append(results)
                    Pdf_Names.append(ID)
            except:
                pass
            driver.close()

        # - Pubsrsc - #
        if links.startswith('https://pubs.rsc.org'):
            driver = webdriver.Chrome(Path+"/chromedriver")
            driver.get(links)
            content = driver.page_source
            time.sleep(random.randint(1, 6))
            soup = BeautifulSoup(content)
            # print(soup.prettify())
            try:
                results = soup.find(class_='abstract').text
                results = results.replace('Abstract', '')
                if results not in Abstracts:
                    Abstracts.append(results)
                    Pdf_Names.append(ID)
            except:
                pass
            driver.close()

print(len(Abstracts))
print(Pdf_Names)
print('wiley_Exceptions: ', wiley_Exceptions)

data = {'Abstract':Abstracts,'ID':Pdf_Names}
df = pd.DataFrame(data)
df.to_csv(Folder_Name+'/'+Folder_Name +'_Abstracts.csv')