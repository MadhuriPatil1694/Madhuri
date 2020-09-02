# -*- coding: utf-8 -*-



#read a csv.file for the topic which need to be summarize which contains abstracts 
from summarizer import Summarizer
import pandas as pd
Topic = 'flexible_electronic_skin_sensors'# topics are (body-centric_wireless_communication,flexible_electronic_skin_sensors,wireless_medical_implants)
path=Topic + '/' + Topic+ '_Abstracts.csv')
Test_data=pd.read_csv("flexible_electronic_skin_sensors_Abstracts2.csv", encoding='latin1')   
summary=[]

# using pretrained BERT to summarize each abstract of the topic selected
cleaned_text = []
model = Summarizer()
joint_summ=''
print(Test_data['Abstract'][114])
for i in range(len(Test_data['Abstract'])):
  if (str(Test_data['Abstract'][i])!='nan'):
    temp = model(Test_data['Abstract'][i],ratio=0.15)
    cleaned_text.append(temp)
    #print(Test_data['Abstract'][i])
    if i < len(Test_data['Abstract'] ) - 1:
      joint_summ= joint_summ + temp + ' '
    else:
      joint_summ= joint_summ + temp
  else:
    cleaned_text.append('')# merged all the summaries obtained
print(cleaned_text)

# summarise the merged data using BERT
model = Summarizer()
Final_sum= model(joint_summ,ratio=0.1,algorithm='kmeans')

#Evaluation
import numpy as np
from rouge_score import rouge_scorer
from operator import add, floordiv
import textstat
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(joint_summ,Final_sum)
print(scores)
test_data=Final_sum
print(textstat.automated_readability_index(test_data))
print(textstat.flesch_reading_ease(test_data))