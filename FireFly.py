# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:03:47 2019

@author: Guinex
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:25:07 2019

@author: Guinex
"""

from kivy.app import App
from pong import PongPaddle, PongBall, PongGame
from kivy.clock import Clock
from functools import partial
from model import model
import numpy as np
#from time import sleep
import time
from random import *
import threading
import operator
from numpy import copy
import pickle
import statistics
STRENGTH = 3.0
W1_shape = (4, 4)
W2_shape = (4, 5)
W3_shape = (1, 5)
DISCARD_TOO_GOOD=True
INIT_POPULATION_SIZE = 5
PICK_FIT_MAX = 3
POPULATION_MAX = INIT_POPULATION_SIZE - 2
MAX_GENERATION = 100
TRAIN = False
final_best_individual = None
previous_score = 0 # will be used for early stopping
SPEED = 1.0/60.0
MOVE_BY_PACE = 0.3
def initialize_population():
    population = []
    #convention: each chromosome will have dimensions 1x33
    for i in range(INIT_POPULATION_SIZE):
        W1 = np.random.randn(W1_shape[0], W1_shape[1])
        W2 = np.random.randn(W2_shape[0], W2_shape[1])
        W3 = np.random.randn(W3_shape[0], W3_shape[1])
        population.append(np.concatenate((W1.flatten().reshape(1,W1_shape[0]*W1_shape[1] ),
                                W2.flatten().reshape(1,W2_shape[0]*W2_shape[1] ),
                          W3.flatten().reshape(1,W3_shape[0]*W3_shape[1] )),axis = 1)
                          )
    ##// can use generate random chromosome function
    return population

def get_weights_from_encoded(individual):
    W1 = individual[:, 0:W1_shape[0]*W1_shape[1]]
    W2 = individual[:, W1_shape[0]*W1_shape[1]:W1_shape[0]*W1_shape[1]+W2_shape[0]*W2_shape[1]]
    W3 = individual[:, W1_shape[0]*W1_shape[1]+W2_shape[0]*W2_shape[1]:]
    return (W1.reshape(W1_shape[0], W1_shape[1]), W2.reshape(W2_shape[0], W2_shape[1]), W3.reshape(W3_shape[0], W3_shape[1]))

def generate_random_chromosome():
    W1 = np.random.randn(W1_shape[0], W1_shape[1])
    W2 = np.random.randn(W2_shape[0], W2_shape[1])
    W3 = np.random.randn(W3_shape[0], W3_shape[1])
    return np.concatenate((W1.flatten().reshape(1,W1_shape[0]*W1_shape[1] ),
                                W2.flatten().reshape(1,W2_shape[0]*W2_shape[1] ),
                          W3.flatten().reshape(1,W3_shape[0]*W3_shape[1] )),axis = 1)
    


def moveTowardsParent(chromosome,parent, prob):
    global STRENGTH
    chromosome0 = copy(chromosome)
    operators = ['add', 'subtract']

    operators = ['add', 'subtract']
    for i in range(len(chromosome0)):
        if random() < prob:
            if choice(operators) == 'add':
                chromosome0[i] += parent[i]/STRENGTH
                mutated = True
            else:
                chromosome0[i] -= parent[i]/STRENGTH
                mutated = True
    return chromosome0, True
    


def selectindex(from_new):
    if from_new:
        return randint(0, PICK_FIT_MAX)
    else:
        return randint(0, POPULATION_MAX) 

def tournament(population, pong): #rank function
    fitnesses = []
    for individual in population:
        #generate a model from an individual
        new_model = model(get_weights_from_encoded(individual))
        pong.game_over = False
        pong.player1.score = 0
        pong.player2.score = 0
        pong.player2.pseudo_score = 0
        pong.result = ""
        pong.player1.center_y = pong.height/2
        pong.player2.center_y = pong.height/2
        t1 = time.time()
        while not pong.game_over:
            pong.update("", new_model)
        t2 = time.time()
#        print("time elapsed:"+ str(t2 - t1))
        fitnesses.append(pong.player2.pseudo_score)
    zip1 = zip(fitnesses,population)
    sorted_results = sorted(zip1, key=operator.itemgetter(0), reverse = True)
    sorted_pop = [x for _,x in sorted_results]
    sorted_fitnesses = [_ for _,x in sorted_results]
    print(sorted_fitnesses)
    return sorted_pop, sorted_fitnesses
    

def update_strength(generation):
    global STRENGTH
    STRENGTH = STRENGTH/generation
        
def main_function(dt, pong):
    global final_best_individual, previous_score
    moving_prob = MOVE_BY_PACE 
    population = initialize_population() # the very first population
    for i in range(MAX_GENERATION):
        print("Generation ", i)
        sorted_pop, sorted_fitnesses = tournament(population, pong)
        if sorted_fitnesses[0] > previous_score:
            final_best_individual = sorted_pop[0] # save the model if score is better than previous
            previous_score = sorted_fitnesses[0] # the best score from previous generation
        print(sorted_fitnesses[0])
        #The first two always make it
        current_generation= sorted_pop
        new_population = []
        if DISCARD_TOO_GOOD:
            if sorted_fitnesses[0] > sorted_fitnesses[1]*10:
                sorted_pop.pop(0)
                sorted_fitnesses.pop(0)
        update_strength(i+1)
        new_population.append(sorted_pop[0])
        new_population.append(sorted_pop[1])
        new_population.append(sorted_pop[2])
        while len(new_population) < INIT_POPULATION_SIZE:
                idx1 = selectindex(True)
                idx2 = selectindex(False)

                child, moved = moveTowardsParent(current_generation[idx2],current_generation[idx1], prob = moving_prob)
                if moved and len(new_population) < INIT_POPULATION_SIZE:
                    new_population.append(child)

                if random() < 0.4 and len(new_population) < INIT_POPULATION_SIZE:
                    new_population.append(generate_random_chromosome())
        population = list(np.copy(new_population))
    sorted_pop, sorted_losses = tournament(population, pong)
    if sorted_fitnesses[0] > previous_score:
        final_best_individual = sorted_pop[0]
    return sorted_losses[0], sorted_pop[0]
        
def pickle_weights(weights):
    W1, W2, W3 = get_weights_from_encoded(weights)
    weights0 = {'W1':copy(W1), "W2":copy(W2), "W3":copy(W3)}
    with open("weightsfirefly.pkl", 'wb') as file:
        pickle.dump(weights0, file)


def load_pickled_weights():
    with open("weights.pkl", 'rb') as file:
        weights = pickle.load(file)
    return weights["W1"], weights['W2'], weights["W3"]


class PongApp(App):
    event = None
    def build(self):
        game = PongGame()
        game.serve_ball()
        threading.Thread(target = partial(main_function, "", game)).start()
        return game


if __name__ == "__main__" and TRAIN:
  app = PongApp()
  app.run()

  pickle_weights(final_best_individual) #save the best evolved weights


W1, W2, W3 = load_pickled_weights()
trained_individual = np.concatenate((W1.flatten().reshape(1,W1_shape[0]*W1_shape[1] ),
                                W2.flatten().reshape(1,W2_shape[0]*W2_shape[1] ),
                          W3.flatten().reshape(1,W3_shape[0]*W3_shape[1] )),axis = 1)

class PongApp(App):
    event = None
    def build(self):
        game = PongGame()
        game.serve_ball()
        print("now playing")
        #target = partial(game.update,model = model(get_weights_from_encoded(final_best_individual)))).start()
        self.event = Clock.schedule_interval(partial(game.update, model = model(get_weights_from_encoded(trained_individual))), SPEED)
        return game

app = PongApp()
app.run()
