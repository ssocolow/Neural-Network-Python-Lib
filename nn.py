import matrix2d
import math

inputs = 0

class NeuralNetwork:
    def __init__(self, shape):

        #get the shape as a two dimensional array
        self.shape = shape

        #store an activation function
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        self.activation_function = sigmoid
        #initialize the container weights array
        #this will store all of the matrices needed (from the matrix2d.py library)
        self.weight_matrices = []

        #initialize the container bias array
        #this will store all of the biases for all of the neurons (organized in matrices)
        self.bias_matrices = []

        #get the length of self.shape so we don't need to repeatedly call the len function
        self.len_selfshape = len(self.shape)

        #interate over every neuron layer in the neural net
        for i in range(self.len_selfshape - 1):
            #add a matrix to the weights matrix
            #keep in mind that it is added to the end of the bigger array
            #the weight matrix has rows equal to the number of nodes in the next layer, and columns equal to the number of inputs coming in
            self.weight_matrices.append(matrix2d.Matrix(self.shape[i + 1][0], self.shape[i][0]))

            #set random weights between 1 and -1
            self.weight_matrices[i].randomize(1,1)

            #add a bias vector to the bias_matrices container array
            #should have rows equal to the number of neurons in the next layer and one collumn (because it is a vector)
            self.bias_matrices.append(matrix2d.Matrix(self.shape[i + 1][0], 1))

            #set random biases between 1 and -1
            self.bias_matrices[i].randomize(1,1)



    #lots of matrix math
    #takes the inputs and feeds it through the network
    def feedforward(self, input_arr):
        #turn the input array into an input matrix
        inputs = matrix2d.Matrix.vectorize(input_arr)
        #iterate over every time we need to do O = a(W * I + B)
        for i in range(self.len_selfshape - 1):
            weighted_sum = matrix2d.Matrix.multiply(self.weight_matrices[i], inputs)
            weighted_sum.add(self.bias_matrices[i])
            inputs = weighted_sum.map(self.activation_function)

        return inputs


    #prints the shape of the neural net and all of its weights
    def print(self):
        print("Shape: " + str(self.shape))
        print()
        print("Weights:")
        for i in range(self.len_selfshape - 1):
            print(self.weight_matrices[i].data)
        print()
        print("Biases: ")
        for i in range(self.len_selfshape - 1):
            print(self.bias_matrices[i].data)

p = NeuralNetwork([[3], [2], [1]])
p.print()
print()
output = p.feedforward([2,2,2])
print("Output: " + str(output.data))