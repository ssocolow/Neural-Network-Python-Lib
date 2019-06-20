import matrix2d
import math

class NeuralNetwork:
    def __init__(self, shape):

        #get the shape as a two dimensional array
        self.shape = shape

        #store an activation function
        #ADD YOUR OWN ACTIVATION FUNCTION BELOW THEN CHANGE self.activation_function to be equal to the name of your activation function
        #IF YOU WANT TO DO BACKPROPAGATION AND GRADIENT DESCENT WITH YOUR OWN ACTIVATION FUNCTION THEN PUT THE DERIVATIVE OF YOUR ACTIVATION FUNCTION BELOW EQUAL TO self.activation_function_derivative
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        #this should really be return sigmoid(x) * (1- sigmoid(x))
        #but the output of each layer has already been fed through the sigmoid function
        def dsigmoid(y):
            return y * (1 - y)

        def twice(x):
            return x*2

        self.activation_function = sigmoid
        self.activation_function_derivative = dsigmoid

        #set a learning rate
        #the learning rate might want to be set with a function argument or in the initialization of a neural network object
        self.lr = 0.1

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
            self.weight_matrices[i].randomize(-1,1)

            #add a bias vector (a Matrix object) to the bias_matrices container array
            #should have rows equal to the number of neurons in the next layer and one column (because it is a vector)
            self.bias_matrices.append(matrix2d.Matrix(self.shape[i + 1][0], 1))

            #set random biases between -1 and 1
            self.bias_matrices[i].randomize(-1,1)

        #create an array with all of the transposed weight matrices(rows of original matrix = columns of transposed matrix)
        #we are doing it once so that we don't have to keep transposing all of the weight matrices whenever we call the train function
        self.weight_matrices_transposed = []

        for i in range(self.len_selfshape - 1):
            self.weight_matrices_transposed.append(matrix2d.Matrix.transpose(self.weight_matrices[i]))

        #we are reversing the transposed weight matrices array because we need to iterate backwards through the network
        #we need to use the last weights of the network first so we can backpropagate the error
        self.weight_matrices_transposed.reverse()

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
        return inputs.matrix_vector_to_array()

    #the train function should use backpropagation and gradient descent to change the weights of the network
    #useful for supervised learning (where you have labels for data)
    #arguments should both be arrays and one dimensional (vectors)
    def train(self, inputs, targets):

        #turn the input array into an input matrix
        inputs = matrix2d.Matrix.vectorize(inputs)

        #make an array to contain all of the outputs of all of the layers
        #this is needed for calculating the change in weights
        layer_outputs = []

        #make an array to contain all of the transposed outputs of the layers
        #needed for calculating change in weights
        layer_outputs_transposed = []

        #iterate over every time we need to do Output = activation_function(Weight_matrix * Input_matrix + Bias_vector)
        #we need to do this every gap between layers (which is the numbers of layers we have - 1)
        for i in range(self.len_selfshape - 1):

            weighted_sum = matrix2d.Matrix.multiply(self.weight_matrices[i], inputs)

            weighted_sum.add(self.bias_matrices[i])

            inputs = weighted_sum.map(self.activation_function)

            #get that layer's output
            layer_outputs.append(inputs)

            if i != (self.len_selfshape - 1):
                #save the transposed version if it isn't the final output
                #we don't need the final ouput transposed for the backpropagation
                layer_outputs_transposed.append(matrix2d.Matrix.transpose(inputs))
            else:
                pass

        #the layer outputs and the transposed versions need to start with the last outputs for a backpropagation loop
        layer_outputs.reverse()
        layer_outputs_transposed.reverse()

        #get the network's guess
        #this is the last mapped weighted sum from the feedforward step
        #the difference from this to the feedforward step is that we want to keep it as a matrix
        outputs_matrix = inputs

        #make an array to store all of the error values for all the nodes
        errors = []

        #make the targets into a matrix so we can subtract the neural network's guess from it
        targets_matrix = matrix2d.Matrix.vectorize(targets)

        #error = targets - guess
        errors.append(targets_matrix.subtract(outputs_matrix))

        #calculate all of the error matrices for all of the nodes and save them in the errors array
        #the transposed weight matrix times the error matrix of the forward layer equals the error matrix of this layer
        for i in range(self.len_selfshape - 1):
            errors.append(matrix2d.Matrix.multiply(self.weight_matrices_transposed[i], errors[i]))

        #calculate the change in weight matrices
        #change in weight matrix = learning rate times error vector of the layer in front times the derivative of the activation function times the outputs of the behind layer
        for i in range(self.len_selfshape - 1):

            print("Layer ouput: " + str(layer_outputs[i].data))
            #calculate the gradient
            gradient = matrix2d.Matrix.static_map(layer_outputs[i], self.activation_function_derivative)
            gradient.multiply_elementwise(errors[i])
            gradient.multiply_elementwise(self.lr)

            #calculate the change in weights
            print("Layer ouputs transposed: " + str(layer_outputs_transposed[i].data))
            weight_deltas = matrix2d.Matrix.multiply(gradient, layer_outputs_transposed[i])

            #the weight_matrices start from the inputs to the first hidden layer
            #we are going backward through the network
            print(self.weight_matrices[self.len_selfshape - (2 + i)].data)
            #the self.weight_matrices is working as it should, but the weight_deltas are off, maybe they are from a different layer
            print(weight_deltas.data)
            self.weight_matrices[self.len_selfshape - (2 + i)].add(weight_deltas)

        #REMEMBER TO CHANGE THE TRANSPOSED WIEGHT MATRICES AFTER NEW WEIGHTS ARE CALCULATED
        for i in range(self.len_selfshape - 1):
            self.weight_matrices_transposed[i] = matrix2d.Matrix.transpose(self.weight_matrices[i])

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

# #testing and debugging
# p = NeuralNetwork([[2], [2], [2]])
# #p.print()
# output = p.feedforward([2,2])
# err = p.train([2,2], [8,6])
# print("Output: " + str(output))
# for i in range(p.len_selfshape - 1):
#     print("Error: " + str(err[i].data))
# print()
# err2 = p.train([2,2], [8,6])
# for i in range(p.len_selfshape - 1):
#     print("Error: " + str(err2[i].data))