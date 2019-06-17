import nn

nn = nn.NeuralNetwork([[2],[2],[2]])

inputs = [2,2]
targets = [0.6,0]

output = nn.feedforward(inputs)
print(output)

for i in range(10000):
    nn.train(inputs,targets)
output = nn.feedforward(inputs)
print(output)