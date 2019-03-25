import random

w0 = random.uniform(-1,1)
w1 = random.uniform(-1,1)
w2 = random.uniform(-1,1)

learning_rate = 0.01

#the 1 is the bias
inputs = [5,2,1]
target = 2

for i in range(1000):
    sum1 = w0*inputs[0] + w1*inputs[1] + w2*inputs[2]
    output = sum1/2

    error = target - output

    cw0 = error * inputs[0] * learning_rate
    cw1 = error * inputs[1] * learning_rate
    cw2 = error * inputs[2] * learning_rate

    w0 += cw0
    w1 += cw1
    w2 += cw2

    print(str(error) + '     ' + str(output))

print()
print(str(w0)+'  '+str(w1)+'   '+str(w2))