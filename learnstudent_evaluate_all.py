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

def convert_state(state,dic1,dic2):
    out = map(lambda x: int(dic2[dic1[str(x)]]),state)
    return out

def evaluate(brain,env,config,game_id):
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
    for estep in range(config.NUM_EVAL_STEPS):
        #@TODO:add progress bar here

        action_indicator = np.zeros(env.action_size())
        object_indicator = np.zeros(env.object_size())
        #predict
        action_index,object_index = brain.getAction(available_objects,game_id)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1

        
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
    brain.data[1] = reader('1_mem.txt')
    brain.data[2] = reader('2_mem.txt')
    brain.data[3] = reader('3_mem.txt')


    #adding progress bar for training
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')
    while True:
        for _ in range(1,4):
            brain.train(_)
        brain.timeStep += 1
#####################################################################
        #for evaluating qvalues
        if (brain.timeStep % 100)==0 and (brain.timeStep != 0):
            # save last network
            if not os.path.exists(os.getcwd()+'/StudentSavednetworks'):
                os.makedirs(os.getcwd()+'/StudentSavednetworks')
            brain.saver.save(brain.session, os.getcwd()+'/StudentSavednetworks/'+'network' + '-student', global_step = brain.timeStep)

            for i in range(1,4):
                env_eval = env[i-1]
                if config.TUTORIAL_WORLD:
                    total_reward, nrewards, nepisodes, quest1_reward_cnt, quest2_reward_cnt, quest3_reward_cnt = evaluate(brain, env_eval, config, i)
                else:
                    total_reward, nrewards, nepisodes, quest1_reward_cnt = evaluate(brain, env_eval, config, i)

    #####################################################################
                values = [total_reward,nrewards,nepisodes,quest1_reward_cnt]
                keys_ = ['average_reward','average_numrewards','number_of_episodes','quest1_average_reward_cnt']
                keys = [key + str(i) for key in keys_]
                brain.inject_summary(dict(zip(keys,values)), brain.timeStep)
#####################################################################
        pbar.update(1)


        if (brain.timeStep) > config.MAX_FRAMES:
            brain.train_writer.close()
            break
    brain.session.close()

def main():
    config = Config()
    learnstudent(config)

if __name__ == '__main__':
    main()
