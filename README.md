Both Algorithms use similar Neural Network with 4 input nodes, 1 hidden layer of 5 nodes and 1 output layer of 1 node.

With "DISCARD_TOO_GOOD" parameter we remove all the population member who have very high strength than the others.

With "TRAIN" parameter you can change the game mode from training to test or test to training.

With "SPEED" parameter speed of the game can be changed.

Population members are array of real numbers.


# FireFly Algorithm:
## Control Parameters: 
	initial_population_size = 5
	PICK_FIT_MAX = 3
	POPULATION_MAX = initial_population_size - 2
	MAX_GENERATION = 100

# Description:

We start with small population of "initial_population_size" i.e, generate "initial_population_size" set of individuals(weights of the neural network). Run game using these weights and check how many times paddle of AI player is able to hit the ball (this score is pseudo score) which is their strength, now sort the sets of population in descending order of their strength and pick top "PICK_FIT_MAX" (the currently selected population). 

	Pick any 2 randomly from this population (one as parent and one that moves towards it) and change values at indices selected at random. To change we either add or subtract a proportion of parent individual value at that index.
	Now as this population is as similar as the pervious population(small variation), generate new population members(set of weights)
and add them to current population
	Again the tournament is performed and same process is repeated for "MAX_GENERATIONS"


###### Genetic Algorithm:  
## Control Parameters:
	initial_population_size = 10
	PICK_FIT_MAX = initial_population_size
	POPULATION_MAX = initial_population_size - 2
	MAX_GENERATION = 50
	CRX2 = 0.3
	CRX = 0.6
	MUTATE = 0.1
