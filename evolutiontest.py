#import libraries
import nn
import random

#have global storing variables
POPSIZE = 100
nets = []
num_of_gens = 0
outputs = []

#have data to determine what you want to do
house_inputs = [[0.2, 0.2, 0.175],[0.2, 0.2, 0.096],[0.3, 0.2, 0.18], [0.3, 0.3, 0.2238]]
house_targets = [[0.16],[0.053],[0.129],[0.178]]

def epoch():
    #nets is going to have arrays which have generations in the big array
    nets.append([])
    outputs.append([])

    #fill the first array in nets with randomly initialized neural nets
    if num_of_gens == 0:
        for i in range(POPSIZE):
            nets[num_of_gens].append(nn.NeuralNetwork([[3],[4],[1]], mutation_rate = 0.4))
    else:

    for i in range(POPSIZE):
        outputs[num_of_gens].append(nets[i].feedforward(house_inputs[0]))

    for i in range(POPSIZE):
        nets

#have an array with all the outputs
#have an array with 1 / (target - output)^2 (squared to get rid of sign)
#have the networks attempt all data
#with the score array, add them all up and then divide each one by the total for the probablility
#have an array with numbers equal to the index of the array which has the nets, number of numbers corresponds to the probablility