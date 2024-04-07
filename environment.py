# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:05:34 2016

@author: ghulamahmedansari
"""

import zmq
import numpy as np
from collections import deque
import zmq
import sys

class Environment(object):
    def __init__(self,counter):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.availableObjects = []
        self.socket.connect ("tcp://localhost:1234" + str(counter))
        self.START_NEW_GAME = True
        self.CURR_REWARD = 0 
        #for best network saving
        self.reward_history = deque()
        # ZeroMQ Context
        context = zmq.Context()
        # Define the socket using the "Context"
        sock = context.socket(zmq.REQ)
        self.vocabsize = None

    
    def getScrRewTer(self, msg):
        msgsplit = msg.split("#")
        # print msgsplit[0]
        # print msgsplit[1]
        # print msgsplit[2]
        self._screen_ = [int(i) for i in msgsplit[0].split(" ")]
        self.reward = float(msgsplit[1])
        # print "DEBUG::" + str(msgsplit[2])
        if msgsplit[2] == "true":
            self.terminal = True
        else:
            self.terminal = False
        self.availableObjects = [int(i) for i in msgsplit[3].split(" ")]
        return self._screen_, self.reward, self.terminal, self.availableObjects

    def interact(self, str):
        self.socket.send(str)
        msg = self.socket.recv()
        return msg


    def newGame(self):
        str = 'newGame'
        msg = self.interact(str)
        self._screen_, self.reward, self.terminal, self.availableObjects = self.getScrRewTer(msg)
        return self._screen_, self.reward, self.terminal, self.availableObjects

    def step(self, action, object):
        str_ = 'step_game' + '#' + str(action+1) + '#' +str(object+1)
        msg = self.interact(str_)
        self._screen_, self.reward, self.terminal, self.availableObjects = self.getScrRewTer(msg)
        return self._screen_, self.reward, self.terminal, self.availableObjects
        # self._screen_, self.reward, self.terminal, _ = self._env.step(action)         
               
    def action_size(self):
        str = 'getActions'
        msg = self.interact(str)
        return int(msg)
            
    def object_size(self):
        str = 'getObjects'
        msg = self.interact(str)
        return int(msg)

    def vocab_size(self):
        if self.vocabsize == None:
            str = 'vocab_size'    
            msg = self.interact(str)
            self.vocabsize = int(msg)+1+1 #1 to accomodate Null Index
        return self.vocabsize
