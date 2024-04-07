# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
from models.DQN import DQN
# from models.bow_DQN import DQN
#from models.lstdq import DQN
import numpy as np
import cPickle as cpickle
from models.config import Config
from tqdm import tqdm
import random
import sys
from environment import Environment

def evaluate(brain,env,config):
    state, reward, terminal, available_objects = env.newGame()
    brain.history.add(state)

    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0

    quest3_reward_cnt = 0
    quest2_reward_cnt = 0
    quest1_reward_cnt = 0    
    pbar =  tqdm(total = config.NUM_EVAL_STEPS, desc = 'TESTING')
    for estep in range(config.NUM_EVAL_STEPS):
        #@TODO:add progress bar here

        action_indicator = np.zeros(env.action_size())
        object_indicator = np.zeros(env.object_size())
        #predict
        action_index,object_index = brain.getAction(available_objects, True)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1

        
        ##-- Play game in test mode (episodes don't end when losing a life)
        nextstate,reward,terminal, available_objects = env.step(action_index,object_index)

        #observe
        brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, True)
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

        if terminal:
            total_reward = total_reward + episode_reward
            episode_reward = 0
            nepisodes = nepisodes + 1
            state, reward, terminal, available_objects = env.newGame()
            brain.history.add(state)

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
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')
    episode_length = 0
    num_episodes = 0
    total_reward = 0
    while True:
        if env.START_NEW_GAME:
            episode_length = 0
            env.START_NEW_GAME = False
            state, reward, terminal, availableObjects = env.newGame()
            brain.history.add(state)
        action_indicator = np.zeros(actions)
        object_indicator = np.zeros(objects)
        #predict
        action_index,object_index = brain.getAction(availableObjects)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1
        #act
        nextstate,reward,terminal, availableObjects = env.step(action_index,object_index)
        total_reward += reward
        episode_length += 1
        #observe
        brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, False)
        state = nextstate

        if ((terminal) or ((episode_length % config.max_episode_length) == 0)):
            num_episodes += 1
            with open("train_reward.txt", "a") as fp:
                print >> fp, (total_reward / (num_episodes * 1.0))    
            env.START_NEW_GAME = True
#####################################################################
        #for evaluating qvalues
        if (brain.timeStep % config.EVAL == 0) and (brain.timeStep != 0):
            if (brain.timeStep / config.EVAL == 1):
                if not ((os.path.exists("checkStates.txt")) and (os.path.getsize("checkStates.txt") > 0)):                    
                    assert config.SAMPLE_STATES % config.BATCH_SIZE == 0 
                    assert config.SAMPLE_STATES < brain.memory.count
                    checkStates, _1, _2, _3, _4, _5 = brain.memory.sample()
                    with open("checkStates.txt", "w") as fp:
                        cpickle.dump(checkStates,fp)
                else:
                    with open("checkStates.txt", 'r') as fp:
                        checkStates = cpickle.load(fp)

            evalQValues_a = brain.action_valueT.eval(feed_dict={brain.stateInputT:checkStates},session = brain.session)
            maxEvalQValues_a = np.max(evalQValues_a, axis = 1)
            avgEvalQValues_a = np.mean(maxEvalQValues_a)

            with open("evalQValue_a.txt", "a") as fp:
                print >>fp,avgEvalQValues_a

            evalQValues_o = brain.object_valueT.eval(feed_dict={brain.stateInputT:checkStates},session = brain.session)
            maxEvalQValues_o = np.max(evalQValues_o, axis = 1)
            avgEvalQValues_o = np.mean(maxEvalQValues_o)

            with open("evalQValue_o.txt", "a") as fp:
                print >>fp,avgEvalQValues_o
#####################################################################
            #save current history before starting evaluation
            # temp_history_data = brain.history.copy()
            #now let us evaluate avg reward                        
            #create alternate environment for EVALUATION
            # env_eval = Environment(2)
            env_eval = env
            if config.TUTORIAL_WORLD:
                total_reward, nrewards, nepisodes, quest1_reward_cnt, quest2_reward_cnt, quest3_reward_cnt = evaluate(brain, env_eval, config)
            else:
                total_reward, nrewards, nepisodes, quest1_reward_cnt = evaluate(brain, env_eval, config)
            
            with open("test_reward.txt", "a") as fp:
                print >> fp, total_reward

            #setting the best network        
            if len(env_eval.reward_history)==0 or total_reward > max(env_eval.reward_history):
                # save best network
                if not os.path.exists(os.getcwd()+'/Savednetworks'):
                    os.makedirs(os.getcwd()+'/Savednetworks')
                brain.saver.save(brain.session, os.getcwd()+'/Savednetworks/'+'network' + '-dqn', global_step = brain.timeStep)                

            env_eval.reward_history.append(total_reward) #doing this for keeping track of best network    
            
            #go back to saved frame after evaluation completed
            # brain.history.add(temp_history_data)
#####################################################################
            if config.TUTORIAL_WORLD:
                brain.inject_summary({
                    'average.q_a': avgEvalQValues_a,
                    'average.q_o': avgEvalQValues_o,
                    'average.q': (0.5*avgEvalQValues_o+0.5*avgEvalQValues_a),
                    'average_reward':total_reward,
                    'average_num_pos_reward':nrewards,
                    'number_of_episodes':nepisodes,
                    'quest1_average_reward_cnt':quest1_reward_cnt,
                    'quest2_average_reward_cnt':quest2_reward_cnt,
                    'quest3_average_reward_cnt':quest3_reward_cnt
                  }, brain.timeStep)
            else:
                brain.inject_summary({
                    'average.q_a': avgEvalQValues_a,
                    'average.q_o': avgEvalQValues_o,
                    'average.q': (0.5*avgEvalQValues_o+0.5*avgEvalQValues_a),
                    'average_reward':total_reward,
                    'average_numrewards':nrewards,
                    'number_of_episodes':nepisodes,
                    'quest1_average_reward_cnt':quest1_reward_cnt
                  }, brain.timeStep)
#####################################################################
        pbar.update(1)
            
            
        if (brain.timeStep) > config.MAX_FRAMES:
            brain.train_writer.close()
            break

    brain.session.close()

def main():
    config = Config()
 #   config.test()
    config.game_num = sys.argv[1]
    playgame(config)

if __name__ == '__main__':
    main()
