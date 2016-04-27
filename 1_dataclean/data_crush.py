# -*- coding: utf-8 -*-  
import csv
import uniout
import pandas as pd
import string
import re
import os

file_path = os.path.dirname(os.getcwd())+'/data/'
def create_freshdataset():
    file_fresh_train = os.path.dirname(os.getcwd())+'/data/21class/21train_.csv'
    file_fresh_test = os.path.dirname(os.getcwd())+'/data/21class/21test_.csv'
    writer = csv.writer(file(file_fresh_train,'wb'))
    writer_test = csv.writer(file(file_fresh_test,'wb'))
    for i in range(1,22):
        file_name =  '21class/21shuffle'+str(i)+'.csv'
        reader = csv.reader(file(file_path+file_name,'rb'))
        k = 0
        for line in reader:
            if len(line[1])>30:
                if k<4501:
                    writer.writerow(line)
                else:
                    writer_test.writerow(line)
                k += 1    
create_freshdataset()
print 'Done!'
print 1