#import libraries
import nn
import random

#have global storing variables
POPSIZE = 100
nets = []
num_of_gens = 0

#have data to evolve the neural nets on
house_inputs = [[0.2, 0.2, 0.175],[0.2, 0.2, 0.096],[0.3, 0.2, 0.18], [0.3, 0.3, 0.2238]]
house_targets = [[0.16],[0.053],[0.129],[0.178]]

def epoch():
    #use the global variables
    global nets
    global num_of_gens

    #array to store scores
    scores = []

    #sum of all the scores
    total = 0

    #an array to store an amount of index numbers for each neural net according to their probability
    indicies_array = []

    #array with probablilites of each network based on its score / total score
    probabilities = []

    #nets is going to have arrays which have generations in the big array
    nets.append([])

    #fill the first array in nets with randomly initialized neural nets
    if num_of_gens == 0:
        for i in range(POPSIZE):
            nets[0].append(nn.NeuralNetwork([[3],[4],[1]], mutation_rate = 0.25))

    for i in range(POPSIZE):
        #get the score of each network by finding absolute value of the difference between the network's output and the target
        #this will give us a smaller number for a better score, so to have a higher number for a higher score, we can do 1 divided by the score so lower score becomes higher score
        scores.append(1 / abs(nets[num_of_gens][i].feedforward(house_inputs[0])[0] - house_targets[0][0]))
        total += scores[i]

    for i in range(POPSIZE):
        #get an array of probablilites for each neural net
        probabilities.append(scores[i] / total)

    for i in range(POPSIZE):
        for j in range(round(probabilities[i] * 100)):
            indicies_array.append(i)

    #add another array to store the next generation
    nets.append([])

    for i in range(POPSIZE):
        nets[num_of_gens + 1].append(nets[num_of_gens][random.choice(indicies_array)].copy().mutate())

    num_of_gens += 1

for i in range(50):
    epoch()


print(nets[num_of_gens][0].feedforward(house_inputs[0]))
# progress notes
# the copy function has a problem, and I can't get the get_data function to work

# have an array with all the outputs
# have an array with 1 / (target - output)^2 (squared to get rid of sign)
# have the networks attempt all data
# with the score array, add them all up and then divide each one by the total for the probablility
# have an array with numbers equal to the index of the array which has the nets, number of numbers corresponds to the probablility
