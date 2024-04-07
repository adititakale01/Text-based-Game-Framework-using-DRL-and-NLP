# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
from models.MultiDQN import MDQN
import numpy as np
import cPickle as cpickle
from models.config import Config
from tqdm import tqdm
import random
import sys
from environment import Environment


def savegame(config):
    fp = open('symbolMapping1.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')[::-1] for data_ in data]
    dic1 = dict(spd[0:-1])
    dic1['0'] = 'NULL'
    fp.close()

    fp = open('symbolMapping2.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')[::-1] for data_ in data]
    dic2 = dict(spd[0:-1])
    dic2['0'] = 'NULL'
    fp.close()

    fp = open('symbolMapping3.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')[::-1] for data_ in data]
    dic3 = dict(spd[0:-1])
    dic3['0'] = 'NULL'
    fp.close()

    fp = open('symbolMapping5.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')for data_ in data]
    dic_global = dict(spd[0:-1])
    dic_global['NULL']='0'
    fp.close()

    # Step 2: init DQN
    actions = 5 #manually setting to avid creating env everytime
    objects = 8 #manually setting to avid creating env everytime
    config.setnumactions(actions)
    config.setnumobjects(objects)
    # config.setvocabsize(env.vocab_size())
    brain = MDQN(config)


    fp = open("MDQN_combined_embeddings.txt","w")
    for word in dic_global.keys():
        state = np.zeros([64,config.seq_length])
        state[:,0]=int(dic_global[word])
        embedding = brain.output_embed.eval(feed_dict={brain.stateInput : state},session=brain.session)[0,0,:]
        print >> fp, word
        for element in embedding:
            print >> fp, element,
        print >> fp
    fp.close()

    fp = open("MDQN_game1_embeddings.txt","w")
    for index in dic1.keys():
        state = np.zeros([64,config.seq_length])
        state[:,0]=int(dic_global[dic1[index]])
        embedding = brain.output_embed.eval(feed_dict={brain.stateInput : state},session=brain.session)[0,0,:]
        print >> fp, dic1[index]
        for element in embedding:
            print >> fp, element,
        print >> fp
    fp.close()        

    fp = open("MDQN_game2_embeddings.txt","w")
    for index in dic2.keys():
        state = np.zeros([64,config.seq_length])
        state[:,0]=int(dic_global[dic2[index]])
        embedding = brain.output_embed.eval(feed_dict={brain.stateInput : state},session=brain.session)[0,0,:]
        print >> fp, dic2[index]
        for element in embedding:
            print >> fp, element,
        print >> fp
    fp.close()        

    fp = open("MDQN_game3_embeddings.txt","w")
    for index in dic3.keys():
        state = np.zeros([64,config.seq_length])
        state[:,0]=int(dic_global[dic3[index]])
        embedding = brain.output_embed.eval(feed_dict={brain.stateInput : state},session=brain.session)[0,0,:]
        print >> fp, dic3[index]
        for element in embedding:
            print >> fp, element,
        print >> fp
    fp.close()            

    brain.session.close()

def main():
    config = Config()
 #   config.test()
    # config.game_num = sys.argv[1]
    savegame(config)

if __name__ == '__main__':
    main()
