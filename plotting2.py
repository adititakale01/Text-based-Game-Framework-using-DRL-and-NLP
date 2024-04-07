#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:32:55 2016

@author: ghulamahmedansari
"""

import numpy as np
from tensorflow.python.summary.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())
    keys = event_acc.Tags()['scalars']
    match_str = 'average_reward'
    avgr = {}
    qcomp = {}
    for key in keys:
        if match_str in key:
            if 'cnt' not in key:
                avgr[key] = event_acc.Scalars(key)
            else:
                qcomp[key] = event_acc.Scalars(key)

    color = ['r','b','g']
    plt.figure()
    for key in avgr.keys():
        if 'reward0' in key:
            i = 0
        elif 'reward1' in key:
            i = 1
        elif 'reward2' in key:
            i = 2
        step = []
        y = []
        for _ in avgr[key]:
            step.append(_[1]/1000)
            y.append(_[2])
        if i !=2:
            plt.plot(step,y,label = "Game "+str(i+1),c = color[i],linewidth=2)
        else:
            plt.plot(step,y,label = "Game 3",c = color[i],linewidth=2)
    plt.plot(step,[0.89 for s in step],label = "Student Best",c = "m",linewidth=2,linestyle='--')
    plt.xlabel("Epoch")
    plt.ylim(-2,1.1)
    plt.xlim(0,80)
    plt.ylabel("Average Reward per Episode")
    plt.legend(loc=1, frameon=True, borderaxespad = 2)
    plt.savefig('mlr.jpg',dpi=440)
    plt.show()

    plt.figure()
    for key in qcomp.keys():
        if 'cnt0' in key:
            i = 0
        elif 'cnt1' in key:
            i = 1
        elif 'cnt2' in key:
            i = 2
        step = []
        y = []
        for _ in qcomp[key]:
            step.append(_[1]/1000)
            y.append(_[2]*100)
        if i !=2:
            plt.plot(step,y,label = "Game "+str(i+1),c = color[i],linewidth=2)
        else:
            plt.plot(step,y,label = "Game 3",c = color[i],linewidth=2)
    plt.plot(step,[100 for s in step],label = "Student Best",c = "m",linewidth=2,linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Quest Completion Percentage(%)")
    plt.ylim(0,110)
    plt.xlim(0,80)
    plt.legend(loc=1, frameon=True, borderaxespad = 2)
    plt.savefig('mlq.jpg',dpi=440)
    plt.show()


if __name__ == '__main__':
    log_file = '/Users/ghulamahmedansari/Downloads/plots/slave3/text-based-game-rl-tensorflow/logs/train/1/events.out.tfevents.1478037991.nslabslave3'
    plot_tensorflow_log(log_file)