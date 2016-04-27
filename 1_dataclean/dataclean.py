# -*- coding: utf-8 -*-  
import csv
import uniout
import pandas as pd
import string
import re
import jieba
import os

stop_words = set(['与','是','的','它','用于','基于','及','对','和','为','之','也','在','我们','有',
                        '有了','通过','意义','一定的','存在','一种','以','本文','而','而且','对于','以来',
                        '总的','近年来','已经','他','她','认为','可以','或者','从','试图','尝试','目前','在于',
                        '没有','并','即使','既是','之一','主要','主题','不但','等','方面','就','被','不是','研究过程',
                        '首先','其次','最后','简要','论述','之后','先','后','一个','一名','简要','说明','至于','这样',
                        '既','又','之所以','所以','结论','因为','是因为','好'])
def remove_stopwords(str):
    for item in stop_words:
        str.replace(item, '')
    return str

identify = string.maketrans('', '')    
file_path = os.path.dirname(os.getcwd())+'/data/'
file_list = ['A','B','C','D','E','F','G','H','I','J','K','N','O','P','Q','R','S','T','U','V','X']
for i in range(len(file_list)):
    file_name = file_list[i]+'.csv'
    file_after = 'new_'+file_list[i]+'.csv'
    reader = csv.reader(file(file_path+file_name,'rb'))
    writer = csv.writer(file(file_path+file_after,'wb'))
    j = 1
    for line in reader:
        if len(line[5])>2:
#            line[0] = file_list[i]
            line[0] = i+1
            line[1] = line[1]+line[2]+line[4]
            delset = string.punctuation+string.digits
            line[1] = line[1].translate(None,delset)
            p=re.compile(r'\w*',re.L)  
            line[1] = p.sub("", line[1])
            j += 1 
#            line[1] = remove_stopwords(line[1])
            writer.writerow([line[0],line[1]])
        if j > 10000:
            break
print 'Done!'

def create_freshdataset():
    file_fresh_train = os.path.dirname(os.getcwd())+'/data/21class/21train.csv'
    file_fresh_test = os.path.dirname(os.getcwd())+'/data/21class/21test.csv'
    writer = csv.writer(file(file_fresh_train,'wb'))
    writer_test = csv.writer(file(file_fresh_test,'wb'))
    for i in range(len(file_list)):
        file_name =  'new_'+file_list[i]+'.csv'
        reader = csv.reader(file(file_path+file_name,'rb'))
        k = 0
        for line in reader:
            k += 1
            line_seg = jieba.cut(line[1],cut_all=False)
            line[1] = ' '.join(line_seg).encode('utf-8')
            line[1] = " ".join(word for word in line[1].split() if word not in stop_words)
#            line[1] = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode('utf-8'), '',line[1])
            if k<4502:
                writer.writerow(line)
            else:
                writer_test.writerow(line)
                
create_freshdataset()
print 'Done!'
print 1