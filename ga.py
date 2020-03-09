#genetic algorithm functions
#inspired by Dan Shiffman and the coding train
#Simon Socolow 2020
import random

#takes an array with objects that have a probability property and picks one based on its probability in an ingenious way
#returns the index of the chosen one in the list
def pickOne(list_):
    index = 0
    r = random.random()
    while r > 0:
        r = r - list_[index].probability
        index = index + 1
    index = index - 1
    return index

#takes an array of objects with a score property and assigns a probability value to their probability property
def calculateFitness(list_):
    sum_ = 0
    for obj in list_:
        sum_ += obj.score
    for obj in list_:
        obj.probability = obj.score / sum_
