# -*- coding: utf-8 -*-
#合并两个csv(train和test)的同类
#每一类shuffle掉
#分开出前4500作为train
import csv
import uniout
import os
import numpy as np
import random
import string
import pandas
import sys
import re 
reload(sys)
sys.setdefaultencoding('utf-8') 

def data_shuffle(class_index):
    data = pandas.read_csv(file_path+'21merge'+str(class_index)+'.csv', header=None)
    df = pandas.DataFrame(data)
    rindex =  np.array(random.sample(xrange(len(df)), len(df)))
    dfr = df.ix[rindex]
    dfr.to_csv(file_path+'21shuffle'+str(class_index)+'.csv',index=False)
    
file_path = os.path.dirname(os.getcwd())+'/data/21class/' 
# reader1 = csv.reader(file(file_path+'21test.csv','rb'))
# reader2 = csv.reader(file(file_path+'21train.csv','rb'))

for class_index in range(1,22):
    writer1 = csv.writer(file(file_path+'21merge'+str(class_index)+'.csv','wb'))
    reader1 = csv.reader(file(file_path+'21test.csv','rb'))
    reader2 = csv.reader(file(file_path+'21train.csv','rb'))
    i = class_index
    for line in reader1:
        if line[0] == str(i):
            line[1] = line[1].strip().decode('utf-8', 'ignore') 
            p2 = re.compile(ur'[^\u4e00-\u9fa5]')
            zh = " ".join(p2.split(line[1])).strip()
            zh = " ".join(zh.split()) 
            line[1] = zh.strip().encode('utf-8')
            writer1.writerow(line)
            print i
    for line in reader2:
        if line[0] == str(i):
            line[1] = line[1].strip().decode('utf-8', 'ignore') 
            p2 = re.compile(ur'[^\u4e00-\u9fa5]')
            zh = " ".join(p2.split(line[1])).strip()
            zh = " ".join(zh.split()) 
            line[1] = zh.strip().encode('utf-8')
            writer1.writerow(line)
    data_shuffle(class_index)