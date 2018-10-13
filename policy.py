#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:34:56 2018

@author: Wellington
"""

import os
import numpy as np

DIR_PATH = os.path.abspath(os.path.dirname('__file__'))
SAVE_DIR = '{}/policies'.format(DIR_PATH)
FILE_NAME_CVS = '{}/outfile.csv'.format(SAVE_DIR)

# ============================== #
#          Building the AI       #
# ============================== #  

class Policy():
    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is the standard deviation of the rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step
        
    def save(self):
        print("Saving best policy: {} to {}".format(self.theta, FILE_NAME_CVS))
        np.savetxt(FILE_NAME_CVS, np.asarray(self.theta), delimiter=",")
    
    def read(self):
        self.theta = np.genfromtxt(FILE_NAME_CVS,delimiter=',')
        