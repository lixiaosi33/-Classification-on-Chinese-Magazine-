# -*- coding: utf-8 -*-
import csv
reader1 = csv.reader(file('19test_.csv','rb'))
reader2 = csv.reader(file('19train_.csv','rb'))
reader3 = open('reduce_wiki','r')

writer = open('cnki_wiki','w')
for line in reader1:
    writer.write(line[1])
for line in reader2:
    writer.write(line[1])
for line in reader3.readlines():
    writer.write(line)
#切记在训练词向量之前将文本非utf8字符消去
#iconv -c -t UTF-8 < wiki.zh.text.jian.seg > wiki.zh.text.jian.seg.utf-8