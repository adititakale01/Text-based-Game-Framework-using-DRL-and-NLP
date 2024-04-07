# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
from models.student import student
# from models.bow_DQN import DQN
# from models.lstdq import DQN
import numpy as np
import cPickle as cpickle
from models.config import Config
from tqdm import tqdm
import random
import sys
from environment import Environment


def savegame(config):
    # Step 2: init DQN
    actions = 5 #manually setting to avid creating env everytime
    objects = 8 #manually setting to avid creating env everytime
    config.setnumactions(actions)
    config.setnumobjects(objects)
    brain = student(config)    
    fp = open('symbolMapping5.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')for data_ in data]
    dic_global = dict(spd[0:-1])
    dic_global['NULL']='0'
    fp.close()

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




    fp = open("student_combined_embeddings.txt","w")
    for word in dic_global.keys():
        state = np.zeros([256,config.seq_length])
        state[:,0]=int(dic_global[word])
        embedding = brain.word_embeds.eval(feed_dict={brain.stateInput : state},session=brain.session)[0,0]
        dic_embedding[dic_trans[word]] = embedding
    fp.close()
    brain.session.close()
    cpickle.dump( dic_embedding, open( "embedStudent.p", "wb" ))
def main():
    config = Config()
 #   config.test()
    # config.game_num = sys.argv[1]
    savegame(config)

if __name__ == '__main__':
    main()
