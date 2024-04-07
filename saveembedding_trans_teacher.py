# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
from models.DQN import DQN
# from models.bow_DQN import DQN
# from models.lstdq import DQN
import numpy as np
import cPickle as cpickle
from models.config import Config
from tqdm import tqdm
import random
import sys
from environment import Environment
import tensorflow as tf


def savegame(config):
    # Step 1: init Game
    env = Environment(config.game_num) #1 is for main game 2 is for evaluation
    ###################
    # Step 2: init DQN
    actions = env.action_size()
    objects = env.object_size()
    config.setnumactions(actions)
    config.setnumobjects(objects)
    config.setvocabsize(env.vocab_size())

    brain = DQN(config)

    # checkStates = None
    #adding progress bar for training
    dic = {}
    with open("symbolMapping"+str(sys.argv[1])+".txt", 'r') as fp:
        data = fp.read().split('\n')
        for i in range(len(data) - 1):
            splitdata = data[i].split(' ')
            dic[int(splitdata[1])] = splitdata[0]
    dic[0] = "NULL"

    dic_trans = {}
    with open("symbolMapping1236.txt", 'r') as fp:
        data = fp.read().split('\n')
        for i in range(len(data) - 1):
            splitdata = data[i].split(' ')
            dic_trans[splitdata[0]] = int(splitdata[1])
    dic_trans["NULL"] = 0   

    dic_embedding = {}
    #1st let us initialize it randomly
    sess = tf.InteractiveSession()
    stateInput = tf.placeholder(tf.int32, [len(dic_trans.keys())])
    embed = tf.Variable(tf.random_uniform([len(dic_trans.keys()), 20], -1, 1),name="embed")
    word_embeds = tf.nn.embedding_lookup(embed, stateInput)
    tf.initialize_all_variables().run()
    state = sorted(dic_trans.values())
    state_map = word_embeds.eval(feed_dict={stateInput : state})

    for i in range(len(state)):
        dic_embedding[state[i]] = state_map[i]
    sess.close()

    for i in range(config.vocab_size-1):
        state = np.zeros([config.batch_size,config.seq_length])
        state[:,0]=i
        embedding = brain.word_embeds.eval(feed_dict={brain.stateInput : state},session=brain.session)[0,0]
        dic_embedding[dic_trans[dic[i]]] = embedding
    brain.session.close()

    cpickle.dump( dic_embedding, open( "embedTeacher"+str(sys.argv[1])+".p", "wb" ))

def main():
    config = Config()
 #   config.test()
    config.game_num = sys.argv[1]
    savegame(config)

if __name__ == '__main__':
    main()
