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

def plot_tensorflow_log(path,inp,flag=0,plot=0):

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
    color = ['b','g','c','r','m','k']
    if plot==0:
        for key in avgr.keys():
            step = []
            y = []
            label = "$A_"+inp +"$"
            for _ in avgr[key]:
                step.append(_[1]/1000)
                y.append(_[2])
            if inp == "4":
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1.5)
            elif inp == "5":
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1.5,linestyle='--')
            elif inp == "6":
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1.5,linestyle='--')
            else:
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1)
            plt.xlabel("Epoch")
            plt.ylabel("Average Reward per Episode")
            plt.ylim(-2,1.1)
            plt.xlim(0,40)
            plt.legend(loc=0, frameon=True)
        if flag==1:
            plt.savefig('transferr.jpg',dpi=440)
            plt.show()

    else:
        for key in qcomp.keys():
            step = []
            y = []
            label = "$A_"+inp +"$"
#            label = label[0:-1]
            for _ in qcomp[key]:
                step.append(_[1]/1000)
                y.append(_[2]*100)
            if inp == "4":
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1.5)
            elif inp == "5":
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1.5,linestyle='--')
            elif inp == "6":
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1.5,linestyle='--')
            else:
                plt.plot(step,y,label = label, c = color[int(inp)-1],linewidth=1)
            plt.xlabel("Epoch")
            plt.ylabel("Quest Completion Percentage(%)")
            plt.ylim(0,110)
            plt.xlim(0,40)
            plt.legend(loc=0, frameon=True)
        if flag==1:
            plt.savefig('transferq.jpg',dpi=440)
            plt.show()


if __name__ == '__main__':
    plt.figure()
    log_file = '/Users/ghulamahmedansari/Downloads/plots/206/T1/logs/train/1/events.out.tfevents.1478024170.cooltoad-ThinkStation-XXXX'
    plot_tensorflow_log(log_file,"1")
    log_file = '/Users/ghulamahmedansari/Downloads/plots/206/T2/logs/train/2/events.out.tfevents.1477758918.karthik-biotech2'
    plot_tensorflow_log(log_file,"2")
    log_file  =  '/Users/ghulamahmedansari/Downloads/plots/211/T3/logs/train/3/events.out.tfevents.1477734904.iitm'
    plot_tensorflow_log(log_file,"3")
    log_file  =  '/Users/ghulamahmedansari/Downloads/plots/211/T4/logs/train/1/events.out.tfevents.1477735025.iitm'
    plot_tensorflow_log(log_file,"4")
    log_file  =  '/Users/ghulamahmedansari/Downloads/plots/211/fixrandom/logs/train/1/events.out.tfevents.1478264218.iitm'
    plot_tensorflow_log(log_file,"5",1,0)

    plt.figure()
    log_file = '/Users/ghulamahmedansari/Downloads/plots/206/T1/logs/train/1/events.out.tfevents.1478024170.cooltoad-ThinkStation-XXXX'
    plot_tensorflow_log(log_file,"1",0,1)
    log_file = '/Users/ghulamahmedansari/Downloads/plots/206/T2/logs/train/2/events.out.tfevents.1477758918.karthik-biotech2'
    plot_tensorflow_log(log_file,"2",0,1)
    log_file  =  '/Users/ghulamahmedansari/Downloads/plots/211/T3/logs/train/3/events.out.tfevents.1477734904.iitm'
    plot_tensorflow_log(log_file,"3",0,1)
    log_file  =  '/Users/ghulamahmedansari/Downloads/plots/211/T4/logs/train/1/events.out.tfevents.1477735025.iitm'
    plot_tensorflow_log(log_file,"4",0,1)
    log_file  =  '/Users/ghulamahmedansari/Downloads/plots/211/fixrandom/logs/train/1/events.out.tfevents.1478264218.iitm'
    plot_tensorflow_log(log_file,"5",1,1)

