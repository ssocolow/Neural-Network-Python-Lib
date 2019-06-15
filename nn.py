import matrix2d
import math

class NeuralNetwork:
    def __init__(self, shape):

        #get the shape as a two dimensional array
        self.shape = shape

        #store an activation function
        #ADD YOUR OWN ACTIVATION FUNCTION BELOW THEN CHANGE self.activation_function to be equal to the name of your activation function
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        def twice(x):
            return x*2

        self.activation_function = twice

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

            #set random weights between -1 and 1
            self.weight_matrices[i].randomize(1,1)

            #add a bias vector (a Matrix object) to the bias_matrices container array
            #should have rows equal to the number of neurons in the next layer and one column (because it is a vector)
            self.bias_matrices.append(matrix2d.Matrix(self.shape[i + 1][0], 1))

            #set random biases between -1 and 1
            self.bias_matrices[i].randomize(1,1)

        #create an array with all of the transposed weight matrices(rows of original matrix = columns of transposed matrix)
        #we are doing it once so that we don't have to keep transposing all of the weight matrices whenever we call the train function
        self.weight_matrices_transposed = []

        for i in range(self.len_selfshape - 1):
            self.weight_matrices_transposed.append(matrix2d.Matrix.transpose(self.weight_matrices[i]))


    #lots of matrix math
    #takes the inputs and feeds it through the network
    def feedforward(self, input_arr):
        #turn the input array into an input matrix
        inputs = matrix2d.Matrix.vectorize(input_arr)

        #iterate over every time we need to do Output = activation_function(Weight_matrix * Input_matrix + Bias_vector)
        #we need to do this every gap between layers (which is the numbers of layers we have - 1)
        for i in range(self.len_selfshape - 1):

            weighted_sum = matrix2d.Matrix.multiply(self.weight_matrices[i], inputs)
            weighted_sum.add(self.bias_matrices[i])
            inputs = weighted_sum.map(self.activation_function)

        #every loop iteration, the inputs to the layer becomes the inputs to the next layer in the network
        #then the last inputs are inputs to the output
        #returns a two dimensional array with all the data of the matrix
        return inputs.data

    #the train function should use backpropagation and gradient descent to change the weights of the network
    #useful for supervised learning (where you have labels for data)
    #arguments should both be arrays and one dimensional (vectors)
    def train(self, inputs, targets):
        #get the network's guess
        outputs = self.feedforward(inputs)

        #make an array to store all of the error values for all the nodes
        errors = []

        #make the targets and the neural networks' guess into matrices so we can subtract them
        targets_matrix = matrix2d.Matrix.vectorize(targets)
        #the feedforward function returns a two dimensional array, and the vectorize function expects a one dimensional array
        outputs_matrix = matrix2d.Matrix.vectorize(outputs[0])

        #error = targets - guess
        errors.append(targets_matrix.subtract(outputs_matrix))
        for i in range(self.len_selfshape - 1):
            #calculate all of the error matrices for all of the nodes
            pass
        return errors


    #prints the shape of the neural net and all of its weights and biases and activation function
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
        print()
        print("Activation function: " + str(self.activation_function))

#testing and debugging
p = NeuralNetwork([[3], [2], [1]])
#p.print()
print()
error = p.train([2,2,2], [8])
output = p.feedforward([2,2,2])
print("Output: " + str(output))
print("Error: " + str(error))