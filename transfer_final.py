#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 22:35:22 2016

@author: ghulamahmedansari
"""

import cPickle as pickle
import random


dics = [{},{},{}]
for j in range(1,4):
    with open("symbolMapping"+str(j)+".txt", 'r') as fp:
        data = fp.read().split('\n')
        for i in range(len(data) - 1):
            splitdata = data[i].split(' ')[::-1]
            dics[j-1][(splitdata[1])] = int(splitdata[0])
    dics[j-1]["NULL"] = 0

dic_trans = {}

with open("symbolMapping1236.txt", 'r') as fp:
    data = fp.read().split('\n')
    for i in range(len(data) - 1):
        splitdata = data[i].split(' ')[::-1]
        dic_trans[int(splitdata[0])] = splitdata[1]
dic_trans[0] = "NULL"


d = {}
for i in [1,2,3]:
    with open("embedTeacher"+str(i)+".p","rb") as fp:
        temp = pickle.load(fp)
        for key in temp.keys():
            if key not in d.keys():
                if dic_trans[key] in dics[i-1].keys():
#                    print dic_trans[key]
#                    print dic_trans[key] in dics[i-1].keys()
                    d[key] = temp[key]
            else:
                if random.random() > 0.5:
                    if dic_trans[key] in dics[i-1].keys():
#                        print dic_trans[key]
#                        print dic_trans[key] in dics[i-1].keys()
                        d[key] = temp[key]
#with open("embedcombTeacher.p","wb") as fp:
#    pickle.dump(d,fp)