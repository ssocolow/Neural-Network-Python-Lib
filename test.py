import nn

nn = nn.NeuralNetwork([[2],[2],[2]])

inputs = [2,2]
targets = [22,9]

for i in range(100):
    error = nn.train(inputs,targets)
    print(error[0].data)