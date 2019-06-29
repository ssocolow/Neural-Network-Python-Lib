#importing the library
#nn requires matrix2d.py and the math module and random module for dependencies
import nn
import random

#create the neural network to solve the XOR problem
#takes an array of arrays for argument
#the 2, 4 and 1 represent two nodes in the input and 4 nodes in the output layer and 1 node in the hidden layer
#you can add more layers by adding an array to the larger array with a number in it for the number of nodes you want like [[2],[3],[3],[4]]
#the learning rate is set to 0.1 (which is default)
example_neural_network = nn.NeuralNetwork([[2],[4],[1]], learning_rate = 0.1)

#have your inputs and targets in an array which match the number of inputs and outputs specificed in the initialization of the neural network
#if you want to use backpropagation and gradient descent in supervised learning
inputs = [[1,0],[0,1],[1,1],[0,0]]
targets = [[1],[1],[0],[0]]

#train the network on the inputs and the targets
for i in range(20000):
    index = random.randint(0,3)
    example_neural_network.train(inputs[index], targets[index])

#check what the network outputs after it has been trained
#this should be close to the targets
print(example_neural_network.feedforward(inputs[0]))
print(example_neural_network.feedforward(inputs[1]))
print(example_neural_network.feedforward(inputs[2]))
print(example_neural_network.feedforward(inputs[3]))

#print out some of the information in the network
example_neural_network.print()