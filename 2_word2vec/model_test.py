# -*- coding: utf-8 -*-
import uniout
import gensim

model = gensim.models.Word2Vec.load('cnki_wiki.model')
try:
    print model.most_similar(u"天上")
    print model[u'天上']
except KeyError:
    print 0
    
# print model.similarity(u'中国', u'韩国')
# from gensim import matutils
# from numpy import dot
# a = model[u'中国']
# b = model[u'韩国']
# print dot(matutils.unitvec(a), matutils.unitvec(b))