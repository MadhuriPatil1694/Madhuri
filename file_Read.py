import os
import pandas as pd

# - find all files in directory - #
Topic = "body-centric_wireless_communications"
#os.chdir(Topic)

files = os.listdir(Topic)
print(files)

# -Find path of current directory

Path = os.getcwd()
print(Path)
Abstracts = []
ID =[]
broken= []
for file in files:

    if file.endswith('.pdf'):

        Txt_name = file.replace('.pdf','.txt')
        command = 'pdftotext ' + Path +'/' +Topic+ '/'+ file + ' ' + Path + '/'+Topic+'/'+Txt_name
        cmd = os.system(command)
        ID.append(file.replace('.pdf',''))


        try:
            f = open(Topic+'/'+Txt_name,'r')
            not_broken =True
        except:
            #print('exception')
            broken.append(file)

        reading = 0
        Abstract = ''

        if not_broken:
            for x in f:
                # if x == 'INTRODUCTION\n' or x == 'Introduction\n' or x =='I NTRODUCTION\n':
                #     break

                if x.lower().replace(' ', '').find('introduction') != -1:
                    break

                if reading == 1:
                    Abstract += x

                # if x == 'ABSTRACT\n':
                if x.lower().replace(' ','').find('abstract') != -1:
                    #print('Found')
                    reading = 1
                    x = x.replace('Abstract -', '')
                    x = x.replace('Abstract-', '')
                    x = x.replace('Abstract', '')

                    Abstract += x

                # elif x.startswith('Abstract') or x.startswith('Abstract-') or x.startswith('Abstract -'):
                #     x = x.replace('Abstract -','')
                #     x = x.replace('Abstract-','')
                #     x = x.replace('Abstract', '')

                    # reading = 1
                    # Abstract += x

            Abstract = Abstract.replace('-\n','-')
            Abstract = Abstract.replace('\n',' ')
            #print(Abstract)
            Abstracts.append(Abstract)


print(len(Abstracts))
data = {'Abstract':Abstracts,'ID':ID}
df = pd.DataFrame(data)
print(df)
print(broken)
if os.path.isfile(Topic+'/'+Topic +'_Abstracts.csv'):
    print('True')
    temp = pd.read_csv(Topic+'/'+Topic +'_Abstracts.csv')
    temp = temp.append(df,ignore_index=True)
    temp.to_csv(Topic+'/'+Topic +'_Abstracts.csv')
else:
    df.to_csv(Topic+'/'+Topic +'_Abstracts.csv')