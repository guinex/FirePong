Both Algorithms use similar Neural Network with 4 input nodes, 1 hidden layer of 5 nodes and 1 output layer of 1 node.

With "DISCARD_TOO_GOOD" parameter we remove all the population member who have very high strength than the others.
With "TRAIN" parameter you can change the game mode from training to test or test to training.
With "SPEED" parameter speed of the game can be changed.
Population members are array of real numbers.


# FireFly Algorithm:
## Control Parameters: 
	INIT_POPULATION_SIZE = 5
	PICK_FIT_MAX = 3
	POPULATION_MAX = INIT_POPULATION_SIZE - 2
	MAX_GENERATION = 100

## Description:

We start with small population of "INIT_POPULATION_SIZE" i.e, generate "INIT_POPULATION_SIZE" set of individuals(weights of the neural network). Run game using these weights and check how many times paddle of AI player is able to hit the ball (this score is pseudo score) which is their strength, now sort the sets of population in descending order of their strength and pick top "PICK_FIT_MAX" (the currently selected population). 

Pick 2 individual from this population (one as parent with higher strenght and one with lower strength that moves towards the other) and change values at indices selected at random. To change we either add or subtract a proportion of parent individual value at that random index.
Now as this population is as similar as the pervious population(small variation), generate new population members(set of weights)
and add them to current population

Again the tournament is performed and same process is repeated for "MAX_GENERATIONS"

## Testing and Training:

Model has been trained and weights have been saved in weightsfirefly.pkl.
This file is loaded while testing model as well.
Try playing with control parameters and generate new files to check the difference.

# Genetic Algorithm:  
## Control Parameters:
	INIT_POPULATION_SIZE = 10
	PICK_FIT_MAX = INIT_POPULATION_SIZE
	POPULATION_MAX = INIT_POPULATION_SIZE - 2
	MAX_GENERATION = 50
	CRX2 = 0.3
	CRX = 0.6
	MUTATE = 0.1

## Description:

As before initial population is generated i.e set of weights of the neural network. Tournament strategy is same as before too.

Instead of randomly picking one with highest strenght as parent and moving the other towards it, we perform crossovers (of 2 types: singal point and multi-point) with probability "CRX" and "CRX2" and then we perform mutations with probability "MUTATE". Now new population members are introduced and are added to current population.

Again the tournament is performed and same process is repeated for "MAX_GENERATIONS"

## Testing and Training:

Model has been trained and weights have been saved in weights3.pkl.
This file is loaded while testing model as well.
Try playing with control parameters and generate new files to check the difference.

## Want to have some Fun?

Run AI trained using genetics algorithm against the one trained using FireFly Algorithm.
(Load pickle file for player 1 as well).


View Code <a href="https://github.com/guinex/FirePong">Here. </a>
