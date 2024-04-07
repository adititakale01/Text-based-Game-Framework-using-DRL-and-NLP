# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import sys
from models.student import student
import numpy as np
from models.config import Config
from tqdm import tqdm
from environment import Environment


def convert_state(state,dic1,dic2):
    out = map(lambda x: int(dic2[dic1[str(x)]]),state)
    return out
def savegame(config):
    fp = open('symbolMapping'+str(sys.argv[1])+'.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')[::-1] for data_ in data]
    dic_local = dict(spd[0:-1])
    dic_local['0'] = 'NULL'
    fp.close()

    fp = open('symbolMapping5.txt','r')
    data = fp.read().split('\n')
    spd = [data_.split(' ')for data_ in data]
    dic_global = dict(spd[0:-1])

    dic_global['NULL']='0'
    fp.close()    

    # Step 1: init Game
    env = Environment(config.game_num) #1 is for main game 2 is for evaluation
    ###################
    # Step 2: init DQN
    actions = env.action_size()
    objects = env.object_size()
    config.setnumactions(actions)
    config.setnumobjects(objects)
    config.setvocabsize(env.vocab_size())
    brain = student(config)

    # checkStates = None
    #adding progress bar for training
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')
    episode_length = 0
    num_episodes = 0
    total_reward = 0
    MAX_STEPS = 105000
    totalSteps = 0
    
    
    while totalSteps < MAX_STEPS:
        totalSteps += 1
        if env.START_NEW_GAME:
            episode_length = 0
            env.START_NEW_GAME = False
            state, reward, terminal, availableObjects = env.newGame()
            state = convert_state(state,dic_local,dic_global)
            brain.history.add(state)
        action_indicator = np.zeros(actions)
        object_indicator = np.zeros(objects)
        #predict
        action_index,object_index = brain.getAction(availableObjects, int(sys.argv[1]))
        
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1
     
     
        #act
        nextstate,reward,terminal, availableObjects = env.step(action_index,object_index)
        nextstate =  convert_state(nextstate,dic_local,dic_global)
        total_reward += reward
        episode_length += 1
        #observe
        brain.history.add(nextstate)
        state = nextstate


        if ((terminal) or ((episode_length % config.max_episode_length) == 0)):
            num_episodes += 1
            with open("check_student_reward.txt", "a") as fp:
                print >> fp, (total_reward / (num_episodes * 1.0))
            env.START_NEW_GAME = True

        pbar.update(1)


        if (brain.timeStep) > config.MAX_FRAMES:
            brain.train_writer.close()
            break

    brain.session.close()

def main():
    config = Config()
    config.game_num = sys.argv[1]
    config.testepsilon = 0
    savegame(config)

if __name__ == '__main__':
    main()
