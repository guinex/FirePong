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

W1_shape = (4, 4)
W2_shape = (4, 5)
W3_shape = (1, 5)
DISCARD_TOO_GOOD=True
initial_population_size = 10
PICK_FIT_MAX = initial_population_size //3
POPULATION_MAX = initial_population_size - 2
MAX_GENERATION = 50
TRAIN = False
final_best_individual = None
previous_score = 0 # will be used for early stopping
SPEED = 1.0/60.0
CRX2 = 0.3
CRX = 0.6
MUTATE = 0.1
def initialize_population():
    population = []
    #convention: each chromosome will have dimensions 1x33
    for i in range(initial_population_size):
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
    


def mutate(chromosome, prob):
    if random() >= prob:
        return chromosome, False # No mutation done
    else:
        #mutate each element with a probability of 'prob'
        mutated = False
        chromosome0 = copy(chromosome)
        operators = ['add', 'subtract']
        for i in range(len(chromosome0)):
            if random() < prob:
                if choice(operators) == 'add':
                    chromosome0[i] += random()
                    mutated = True
                else:
                    chromosome0[i] -= random()
                    mutated = True
        return chromosome0, mutated # mutated
    

def crossover(chromosomes, prob ):
    if random() >= prob:
        return chromosomes, False # No crossover done
    else:
        indx = randint(1, chromosomes[0].shape[1]-1)
        p0 = copy(chromosomes[0]); p1 = copy(chromosomes[1])
        temp = copy(p0)
        p0[:, 0:indx] = p1[:, 0:indx]
        p1[:, 0:indx] = temp[:, 0:indx]
        return [p0, p1], True
    

def crossover2(chromosomes, prob):
    p0 = copy(chromosomes[0]); p1 = copy(chromosomes[1])
    crossovered = False
    for i in range(chromosomes[0].shape[1]):
        if random() < prob:
            #swap the numbers at index i
            p0[0, i] = chromosomes[1][0][i]
            p1[0, i] = chromosomes[0][0][i]
            crossovered = True
    return [p0, p1], crossovered


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
        

def main_function(dt, pong):
    global final_best_individual, previous_score
    crossover2_prob = CRX2 
    crossover_prob = CRX 
    mutation_prob = MUTATE 
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
        new_population.append(sorted_pop[0])
        new_population.append(sorted_pop[1])
        while len(new_population) < initial_population_size:
                # select any from the top 6 of the population and randomly breed and mutate them
                # First crossover:
                idx1 = selectindex(True)
                idx2 = selectindex(False) # two parents
                if idx1 != idx2:
                    children, crossovered = crossover([current_generation[idx1],population[idx2]], prob = crossover_prob)
                    if crossovered and len(new_population) < initial_population_size-1:
                        new_population.extend(children)
                # Mutation:
                idx1 = selectindex(True)
                child, mutated = mutate(current_generation[idx1], prob = mutation_prob)
                if mutated and len(new_population) < initial_population_size:
                    new_population.append(child)
#                # Crossover 2:
                idx1 = selectindex(True);
                idx2 = selectindex(False)
                if idx1 != idx2:
                    children, crossovered = crossover2([current_generation[idx1],population[idx2]], prob = crossover2_prob)
                    if crossovered and len(new_population) < initial_population_size-1:
                        new_population.extend(children)
                if random() < 0.35 and len(new_population) < initial_population_size:
                    new_population.append(generate_random_chromosome())
        population = list(np.copy(new_population))
    sorted_pop, sorted_losses = tournament(population, pong)
    if sorted_fitnesses[0] > previous_score:
        final_best_individual = sorted_pop[0]
    return sorted_losses[0], sorted_pop[0]
        
def pickle_weights(weights):
    W1, W2, W3 = get_weights_from_encoded(weights)
    weights0 = {'W1':copy(W1), "W2":copy(W2), "W3":copy(W3)}
    with open("weights4.pkl", 'wb') as file:
        pickle.dump(weights0, file)


def load_pickled_weights():
    with open("weights3.pkl", 'rb') as file:
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
        self.event = Clock.schedule_interval(partial(game.update, model = model(get_weights_from_encoded(trained_individual))), SPEED)
        return game

app = PongApp()
app.run()
