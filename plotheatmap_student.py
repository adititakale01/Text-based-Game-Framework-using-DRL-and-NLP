import tensorflow as tf
from models.student_1_ev import student
from utils import load_data
import numpy as np
from models.config import Config
from tqdm import tqdm
import sys
from environment import Environment
import os
import cPickle  as cpickle
#global Dictionaries for state space conversion
dic = [0,0,0]
for _ in range(3):
    fp = open('symbolMapping'+str(_+1)+'.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')[::-1] for data_ in data]
    dic[_] = dict(spd[0:-1])
    dic[_]['0'] = 'NULL'
    fp.close()

fp = open('symbolMapping5.txt','r')
data = fp.read().split('\n')
spd = [data_.split(' ')for data_ in data]
dic_global = dict(spd[0:-1])
dic_global['NULL']='0'
fp.close()

maps = dict(zip([0,1,2],[[],[],[]]))
a_map = dict(zip([0,1,2],[[],[],[]]))
o_map = dict(zip([0,1,2],[[],[],[]]))

def convert_state(state,dic1,dic2):
    out = map(lambda x: int(dic2[dic1[str(x)]]),state)
    return out

def evaluate(brain,env,config,game_id,H,H1a,H3a, H1o, H3o):
    state, reward, terminal, available_objects = env.newGame()
    state = convert_state(state,dic[game_id-1],dic_global)
    brain.history[game_id-1].add(state)

    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    episode_length = 0

    quest3_reward_cnt = 0
    quest2_reward_cnt = 0
    quest1_reward_cnt = 0    
    pbar =  tqdm(total = config.NUM_EVAL_STEPS, desc = 'TESTING')
    for estep in range(config.NUM_EVAL_STEPS/10):
        #@TODO:add progress bar here

        action_indicator = np.zeros(env.action_size())
        object_indicator = np.zeros(env.object_size())
        #predict
        action_index,object_index = brain.getAction(available_objects,game_id)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1

        #heatmap ops
        state_batch = np.zeros([brain.BATCH_SIZE, brain.config.seq_length])
        state_batch[0] = brain.history[game_id-1].get()
        # print brain.mean_pool.get_shape()
        # print brain.linear_output.get_shape()
        maps[game_id-1].append(brain.session.run(H,feed_dict={brain.stateInput:state_batch})[0])

        if game_id==1:
            a_map[game_id-1].append(brain.session.run(H1a,feed_dict={brain.stateInput:state_batch})[0])
            o_map[game_id-1].append(brain.session.run(H1o,feed_dict={brain.stateInput:state_batch})[0])
        # elif game_id==2: 
        elif game_id==3:
            a_map[game_id-1].append(brain.session.run(H3a,feed_dict={brain.stateInput:state_batch})[0])
            o_map[game_id-1].append(brain.session.run(H3o,feed_dict={brain.stateInput:state_batch})[0])

        ##-- Play game in test mode (episodes don't end when losing a life)
        nextstate,reward,terminal, available_objects = env.step(action_index,object_index)
        episode_length += 1

        #observe
        nextstate = convert_state(nextstate,dic[game_id-1],dic_global)
        brain.history[game_id-1].add(state)
        state = nextstate


        if config.TUTORIAL_WORLD:
            if(reward > 9):
                quest1_reward_cnt =quest1_reward_cnt+1

            elif reward > 0.9:
                quest2_reward_cnt = quest2_reward_cnt + 1
            elif reward > 0:
                quest3_reward_cnt = quest3_reward_cnt + 1 #--defeat guardian
        else:
            if(reward > 0.9):
                quest1_reward_cnt =quest1_reward_cnt+1

        #-- record every reward
        episode_reward = episode_reward + reward

        if reward != 0:
           nrewards = nrewards + 1

        if (terminal  or ((episode_length % config.max_episode_length) == 0)):
            total_reward = total_reward + episode_reward
            episode_reward = 0
            episode_length = 0
            nepisodes = nepisodes + 1
            state, reward, terminal, available_objects = env.newGame()
            state = convert_state(state,dic[game_id-1],dic_global)
            brain.history[game_id-1].add(state)

        pbar.update(1)

    total_reward /= (nepisodes*1.0)
    quest1_reward_cnt /= (nepisodes*1.0)
    nrewards /= (nepisodes*1.0)
    env.START_NEW_GAME = True
    if config.TUTORIAL_WORLD:
        quest2_reward_cnt /= (nepisodes*1.0)
        quest3_reward_cnt /= (nepisodes*1.0)
        return total_reward, nrewards, nepisodes, quest1_reward_cnt, quest2_reward_cnt, quest3_reward_cnt
    else:
        return total_reward, nrewards, nepisodes, quest1_reward_cnt


def reader(fileName):
    data = load_data(fileName)
    return zip (data[0],data[1],data[2])

def learnstudent(config):
    # Step 1: init Game
    env = [Environment(1),Environment(2),Environment(3)]
    actions = env[0].action_size()  #here all 3 game #actions and #objects
    objects = env[0].object_size()  #here all 3 game #actions and #objects
    config.setnumactions(actions)
    config.setnumobjects(objects)
    config.setvocabsize(env[0].vocab_size())


    brain = student(config)
    # brain.data[1] = reader('1_mem.txt')
    # brain.data[2] = reader('2_mem.txt')
    # brain.data[3] = reader('3_mem.txt')

    
    print "--"*100
    print int(brain.linear_output.get_shape()[-1])
    print "--"*100    
    
    jacob  = [tf.gradients(var,[brain.mean_pool])[0] for var in tf.split(1, int(brain.linear_output.get_shape()[-1]), brain.linear_output)]
    H = tf.pack(jacob,axis = 2 )

    jacob_1a  = [tf.gradients(var,[brain.linear_output])[0] for var in tf.split(1, 5, brain.action_value_1)]
    H1a = tf.pack(jacob_1a,axis = 2 )

    jacob_1o  = [tf.gradients(var,[brain.linear_output])[0] for var in tf.split(1, 8, brain.object_value_1)]
    H1o = tf.pack(jacob_1o,axis = 2 )    

    jacob_3a  = [tf.gradients(var,[brain.linear_output])[0] for var in tf.split(1, 5, brain.action_value_3)]
    H3a = tf.pack(jacob_3a,axis = 2 )    

    jacob_3o  = [tf.gradients(var,[brain.linear_output])[0] for var in tf.split(1, 8, brain.object_value_3)]
    H3o = tf.pack(jacob_3o,axis = 2 )        
    # print "check"
    # print H3o.get_shape()
    # print "--"*100
    for i in range(1,4):
        env_eval = env[i-1]
        total_reward, nrewards, nepisodes, quest1_reward_cnt = evaluate(brain, env_eval, config, i,H, H1a, H3a, H1o, H3o)


    brain.session.close()

    with open("maps.p","wb") as fp:
        cpickle.dump(maps,fp)
    with open("maps_a.p","wb") as fp:
        cpickle.dump(a_map,fp)
    with open("maps_o.p","wb") as fp:
        cpickle.dump(o_map,fp)        
        
def main():
    config = Config()
    learnstudent(config)

if __name__ == '__main__':
    main()
