import random

class NeuralNetwork:
    def __init__(self, shape):
        #get the shape as a two dimensional array
        self.shape = shape
        #initialize the weights array
        self.weights_arrays = []

        #get the length of self.shape so we don't need to repeatedly call it
        len_selfshape = len(self.shape)

        #interate over every layer in the neural net
        for i in range(len_selfshape):
            #add an empty array to the weights array
            #keep in mind that it is added to the end of the bigger array
            self.weights_arrays.append([])
            for j in range(self.shape[i][0]):
                self.weights_arrays[i].append(random.uniform(-1, 1))

    def print(self):
        n = len(self.shape)
        for i in range(n):
            print(self.shape[i])

p = NeuralNetwork([[2],[3],[4],[9],[2]])
print(p.weights_arrays)