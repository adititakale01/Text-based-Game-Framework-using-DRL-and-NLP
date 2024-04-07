#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:25:34 2016

@author: ghulamahmedansari
"""
import sys
import numpy as np;
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt;
import cPickle as pickle

def heatmap(X,key, show=True, save=False):
    plt.figure();
    plt.imshow(X, interpolation='nearest', cmap=plt.cm.hot);
    plt.colorbar();
    plt.tick_params(axis='both', labelsize=5)
    if key==2:
        plt.title("Game 4")
    else:
        plt.title("Game "+str(key+1))
    if show:
        plt.show();
    if save:
        plt.savefig(save,dpi=440,bbox_inches='tight');

with open("maps.p","rb") as fp:
    maps = pickle.load(fp)

_max = -sys.maxint
for key in maps.keys():
    maps[key] = np.abs(np.array(maps[key]))
    maps[key] = np.mean(maps[key],axis=0)
    _max = max(_max,np.max(maps[key]))


for key in maps.keys():
    h = np.array(maps[key]/_max*255,dtype=np.uint8)
    maps[key] = h
    heatmap(h,key,False,"CG"+str(key+1)+".eps")



with open("maps_a.p","rb") as fp:
    maps = pickle.load(fp)

_max = -sys.maxint
for key in maps.keys():
    if key!=1:
        maps[key] = np.abs(np.array(maps[key]))
        maps[key] = np.mean(maps[key],axis=0)
        _max = max(_max,np.max(maps[key]))



for key in maps.keys():
    if key!=1:
        h = np.array(maps[key]/_max*255,dtype=np.uint8)
        maps[key] = h
        heatmap(h,key,False,"G"+str(key+1)+".eps")


with open("maps_o.p","rb") as fp:
    maps = pickle.load(fp)

_max = -sys.maxint
for key in maps.keys():
    if key!=1:
        maps[key] = np.abs(np.array(maps[key]))
        maps[key] = np.mean(maps[key],axis=0)
        _max = max(_max,np.max(maps[key]))


for key in maps.keys():
    if key!=1:
        h = np.array(maps[key]/_max*255,dtype=np.uint8)
        maps[key] = h
        heatmap(h,key,False,"oG"+str(key+1)+".eps")

#a = interp2d(range(maps[0].shape[0]),range(maps[0].shape[1]),maps[0],kind='linear')
#b = interp2d(range(maps[1].shape[0]),range(maps[1].shape[1]),maps[1],kind='linear')
#x = a-b
#print np.mean(np.abs(x))