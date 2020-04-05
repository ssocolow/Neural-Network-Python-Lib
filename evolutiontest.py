#import libraries
import nn
import ga
import random

#have global storing variables
#POPSIZE is how many networks in each generation
POPSIZE = 100
#make a variable to hold the number of generations
num_of_gens = 0
#make an array to hold all of the nets
nets = []

#have data to evolve the neural nets on to make a house price predictor based on the number of bathrooms, bedrooms, and square footage
#taking the first input [0.2, 0.2, 0.175] as an example, that house has 2 bathrooms, 2 bedrooms, and 1,750 square feet
#this house has a price of 160,000 dollars
house_inputs = [[0.2, 0.2, 0.175],[0.2, 0.2, 0.096],[0.3, 0.2, 0.18], [0.3, 0.3, 0.2238]]
house_targets = [[0.16],[0.053],[0.129],[0.178]]


#make a fitness function that will return a score for the network based on how well it did
#higher is better
def evaluateNetwork(net):
    __sum = 0
    for i in range(len(house_inputs)):
        __sum += abs(1/(net.feedforward(house_inputs[i])[0] - house_targets[i][0]))
    return __sum


#make an evolve function
def evolve():
    #use the global variables
    global num_of_gens
    global nets
    global POPSIZE

    print(num_of_gens)

    #if it is the first generation, create the first nets
    if num_of_gens == 0:
        nets.append([])
        for i in range(POPSIZE):
            nets[0].append(nn.NeuralNetwork([[3],[32],[16],[1]]))

    #loop through all of the nets of the current generation
    for net in nets[num_of_gens]:
        net.score = evaluateNetwork(net)
        #print(str(nets[num_of_gens].index(net)) + ' / ' + str(POPSIZE))

    #get ready to spawn the next generation
    nets.append([])

    #make the probabilities for the networks to be picked
    nets[num_of_gens + 1].extend(ga.nextGeneration(nets[num_of_gens]))

    #increment the number of generations
    num_of_gens += 1




#run evolve certain number of times
def runEvolve(n):
    for i in range(n):
        evolve()

#run 25 epochs
runEvolve(25)

#weird issue that arises is the first networks have a prediction of 0 for the house price, could come from using too much memory?

#print what the first network thought that a house which has 2 bathrooms, 2 bedrooms, and 1,750 square feet should cost
print(nets[0][0].feedforward(house_inputs[0]))
#then print what one of the networks in the last generation thinks it costs
#the number should be closer to 0.16 than the first
print(nets[num_of_gens][0].feedforward(house_inputs[0]))
