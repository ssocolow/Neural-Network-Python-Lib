#importing the library
#nn requires matrix2d.py and the math module and random module for dependencies
import nn

#create the neural network
#takes an array of arrays for argument
#the 2's and 3 represent two nodes in the input and the output layer and three nodes in the hidden layer
#you can have more layers by adding an array to the array with a number in it for the number of nodes you want like [[2],[3],[3],[4]]
#the first 2 means the network will expect two inputs and the last two means the network will output two outputs in the example
example_neural_network = nn.NeuralNetwork([[2],[3],[2]])

#have your inputs and targets in an array
#if you want to use backpropagation and gradient descent
inputs = [2,2]
targets = [0.6,0]

#feed forward the inputs and put the output in the variable called output
output = nn.feedforward(inputs)
print(output)

#train the network 10000 times on the inputs and the targets
for i in range(10000):
    nn.train(inputs,targets)

#check what the network outputs after it has been trained
#this should be close to the targets
output = nn.feedforward(inputs)
print(output)