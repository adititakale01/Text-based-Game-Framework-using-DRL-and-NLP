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

def convert_state(state,dic1,dic2):
    out = map(lambda x: int(dic2[dic1[str(x)]]),state)
    return out

def evaluate(brain,env,config,game_id):
    state, reward, terminal, available_objects = env.newGame()
    state = convert_state(state,dic[game_id],dic_global)
    brain.history[game_id].add(state)

    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    episode_length = 0

    quest3_reward_cnt = 0
    quest2_reward_cnt = 0
    quest1_reward_cnt = 0    
    pbar =  tqdm(total = config.NUM_EVAL_STEPS, desc = 'TESTING')
    for estep in range(config.NUM_EVAL_STEPS):
        #@TODO:add progress bar here

        action_indicator = np.zeros(env.action_size())
        object_indicator = np.zeros(env.object_size())
        #predict
        action_index,object_index = brain.getAction(available_objects,game_id, True)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1

        
        ##-- Play game in test mode (episodes don't end when losing a life)
        nextstate,reward,terminal, available_objects = env.step(action_index,object_index)
        episode_length += 1

        #observe
        nextstate = convert_state(nextstate,dic[game_id],dic_global)
        brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, game_id, True)
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
            state = convert_state(state,dic[game_id],dic_global)
            brain.history[game_id].add(state)

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

def playgame(config):
    env = [Environment(1),Environment(2),Environment(3)]
    actions = env[0].action_size()  #here all 3 game #actions and #objects
    objects = env[0].object_size()  #here all 3 game #actions and #objects
    config.setnumactions(actions)
    config.setnumobjects(objects)
    config.setvocabsize(env[0].vocab_size())

    brain = MDQN(config)

    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')
    
    episode_length = [0,0,0]
    num_episodes = [0,0,0]
    total_reward = [0,0,0]
    
    # state_, reward_, terminal_, availableObjects_ = [0,0,0],[0,0,0],[0,0,0],[0,0,0]
    while True:
        # for env,episode_length,num_episodes,total_reward,ind in zip(env_,episode_length_,num_episodes_,total_reward_,[0,1,2]):
        for i in range(3):
            if env[i].START_NEW_GAME:
                episode_length[i] = 0
                env[i].START_NEW_GAME = False

                state, reward, terminal, availableObjects = env[i].newGame()
                state = convert_state(state,dic[i],dic_global)
                brain.history[i].add(state)

            action_indicator = np.zeros(actions)
            object_indicator = np.zeros(objects)
            #predict
            action_index,object_index = brain.getAction(availableObjects,i,False)
            action_indicator[action_index] = 1
            object_indicator[object_index] = 1
            #act
            nextstate,reward,terminal, availableObjects = env[i].step(action_index,object_index)
            total_reward[i] += reward
            episode_length[i] += 1
            #observe
            nextstate = convert_state(nextstate,dic[i],dic_global)
            brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, i, False)
            
            if ((terminal)):
                num_episodes[i] += 1
                with open("train_reward_"+str(i)+".txt", "a") as fp:
                    print >> fp, (total_reward[i] / (num_episodes[i] * 1.0))    
                env[i].START_NEW_GAME = True
    #####################################################################
            #for evaluating qvalues
            if i==2:
                if ((brain.timeStep/3) % config.EVAL == 0) and (brain.timeStep/3 != 0):
                    for i in range(3):
                        if (brain.timeStep / (3*config.EVAL) == 1):
                            if not ((os.path.exists("checkStates"+str(i)+".txt")) and (os.path.getsize("checkStates"+str(i)+".txt") > 0)):                    
                                assert config.SAMPLE_STATES % config.BATCH_SIZE == 0 
                                assert config.SAMPLE_STATES < brain.memory[i].count
                                checkStates, _1, _2, _3, _4, _5 = brain.memory[i].sample()
                                with open("checkStates"+str(i)+".txt", "w") as fp:
                                    cpickle.dump(checkStates,fp)
                            else:
                                with open("checkStates"+str(i)+".txt", 'r') as fp:
                                    checkStates = cpickle.load(fp)

                        evalQValues_a = brain.action_valueT.eval(feed_dict={brain.stateInputT:checkStates,brain.controller_id : [i]},session = brain.session)

                        maxEvalQValues_a = np.max(evalQValues_a, axis = 1)
                        avgEvalQValues_a = np.mean(maxEvalQValues_a)

                        with open("evalQValue_a"+str(i)+".txt", "a") as fp:
                            print >>fp,avgEvalQValues_a

                        evalQValues_o = brain.object_valueT.eval(feed_dict={brain.stateInputT:checkStates,brain.controller_id : [i]},session = brain.session)
                        maxEvalQValues_o = np.max(evalQValues_o, axis = 1)
                        avgEvalQValues_o = np.mean(maxEvalQValues_o)

                        with open("evalQValue_o"+str(i)+".txt", "a") as fp:
                            print >>fp,avgEvalQValues_o
            #####################################################################

                        env_eval = env[i]
                        print i
                        total_reward_, nrewards, nepisodes, quest1_reward_cnt = evaluate(brain, env_eval, config,i)
                        
                        with open("test_reward"+str(i)+".txt", "a") as fp:
                            print >> fp, total_reward_

                        if i==1:
                            #setting the best network        
                            if len(env_eval.reward_history)==0 or total_reward_ > max(env_eval.reward_history):
                                # save best network
                                if not os.path.exists(os.getcwd()+'/MDQNSavednetworks'):
                                    os.makedirs(os.getcwd()+'/MDQNSavednetworks')
                                brain.saver.save(brain.session, os.getcwd()+'/MDQNSavednetworks/'+'network' + '-dqn', global_step = brain.timeStep)                

                        env_eval.reward_history.append(total_reward_) #doing this for keeping track of best network    
                        
            #####################################################################
                        values = [avgEvalQValues_a,avgEvalQValues_o,total_reward_,nrewards,nepisodes,quest1_reward_cnt]
                        keys_ = ['average.q_a','average.q_o','average_reward','average_numrewards','number_of_episodes','quest1_average_reward_cnt']
                        keys = [key + str(i) for key in keys_]
                        brain.inject_summary(dict(zip(keys,values)), brain.timeStep/3)
            #####################################################################
        pbar.update(1)
            
            
        if (brain.timeStep/3) > config.MAX_FRAMES:
            brain.train_writer.close()
            break

    brain.session.close()

def main():
    config = Config()
    playgame(config)

if __name__ == '__main__':
    main()
