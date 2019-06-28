#importing the library
#nn requires matrix2d.py and the math module and random module for dependencies
import nn
import random

#create the neural network
#takes an array of arrays for argument
#the 2's and 4 represent two nodes in the input and the output layer and four nodes in the hidden layer
#you can add more layers by adding an array to the larger array with a number in it for the number of nodes you want like [[2],[3],[3],[4]]
#the first 2 of this new network would mean the network would expect two inputs and the last four means the network would output four outputs
example_neural_network = nn.NeuralNetwork([[2],[2],[1]], learning_rate = 0.1)
#example_neural_network.print()
#have your inputs and targets in an array which match the number of inputs and outputs specificed in the creation of the neural network
#if you want to use backpropagation and gradient descent in supervised learning
inputs = [[1,0],[0,1],[1,1],[0,0]]
targets = [[1],[1],[0],[0]]

#feed forward the inputs and put the output in the variable called output
#output = example_neural_network.feedforward(inputs)
#print(output)
errors = []
# #train the network 10000 times on the inputs and the targets
for i in range(1000):
    #save the error in a variable
    #the train function returns the error of the output layer equal to the target minus the network's guess
    index = random.randint(0,3)
    print(example_neural_network.train(inputs[index], targets[index]))
    #print(str(index) + str(errors[i]))

#check what the network outputs after it has been trained
#this should be close to the targets
#print(errors[0])
#print(errors[99])
example_neural_network.print()
print(example_neural_network.train(inputs[0], targets[0]))

print(example_neural_network.feedforward(inputs[0]))
print(example_neural_network.feedforward(inputs[1]))
print(example_neural_network.feedforward(inputs[2]))
print(example_neural_network.feedforward(inputs[3]))