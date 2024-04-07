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
#    for i in range(3):
#        step = []
#        y = []
#        for _ in avgr.values()[i]:
#            step.append(_[1])
#            y.append(_[2])
#        plt.figure()
#        plt.plot(step,y,c = color[i])
#        plt.xlabel("Epoch")
#        plt.ylabel("Average Reward per Episode")
#        if i!=2:
#            plt.title('Game '+str(i+1))
#        else:
#            plt.title('Game 4')
#    #    plt.legend(loc=0, frameon=True)
#
#        plt.savefig('Average Reward per Episode'+str(i+1)+'.jpg',dpi=440)
#        plt.show()
    plt.figure()
    for key in avgr.keys():
        if 'reward1' in key:
            i = 0
        elif 'reward2' in key:
            i = 1
        elif 'reward3' in key:
            i = 2
        step = []
        y = []
        for _ in avgr[key]:
            step.append(_[1])
            y.append(_[2])
        if i !=2:
            plt.plot(step,y,label = "Game "+str(i+1),c = color[i],linewidth=2)
        else:
            plt.plot(step,y,label = "Game 4",c = color[i],linewidth=2)
        plt.xlabel("Number of  Training Steps")
        plt.ylabel("Average Reward per Episode")
        plt.xlim(0,1500)
        plt.legend(loc=0, frameon=True)
    plt.savefig('Average_Reward_per_Episode.jpg',dpi=440)
    plt.show()

    plt.figure()
    for key in qcomp.keys():
        if 'cnt1' in key:
            i = 0
        elif 'cnt2' in key:
            i = 1
        elif 'cnt3' in key:
            i = 2
        step = []
        y = []
        for _ in qcomp[key]:
            step.append(_[1])
            y.append(_[2]*100)
        if i !=2:
            plt.plot(step,y,label = "Game "+str(i+1),c = color[i],linewidth=2)
        else:
            plt.plot(step,y,label = "Game 4",c = color[i],linewidth=2)
        plt.xlabel("Number of  Training Steps")
        plt.ylabel("Quest Completion Percentage(%)")
        plt.ylim(0,110)
        plt.xlim(0,1500)
        plt.legend(loc=0, frameon=True)
    plt.savefig('Quest_Completion_Percentage.jpg',dpi=440)
    plt.show()


if __name__ == '__main__':
    log_file = "./logs/events.out.tfevents.1456909092.DTA16004"
    plot_tensorflow_log(log_file)