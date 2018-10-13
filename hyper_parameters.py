#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:46:33 2018

@author: Wellington
"""

# ============================== #
#         Hyperparameters        #
# ============================== # 

class Hp():
    def __init__(self,
                 nb_steps=1000,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 seed=1,
                 env_name='BipedalWalker-v2',
                 record_every=50):

        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.record_every = record_every