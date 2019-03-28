import random

w0 = random.uniform(-1,1)
w1 = random.uniform(-1,1)
#w2 = random.uniform(-1,1)


learning_rate = 0.01

#the 1 is the bias
inputs = [[10,6], [10,2], [12,8]]
targets = [8, 6, 10]

for i in range(1000):
    t = 0
    for j in inputs:
        print(float(j[0]))
        sum1 = w0*j[0] + w1*j[1] #+ w2*inputs[2]
        output = sum1/2

        error = targets[t] - output

        cw0 = error * j[0] * learning_rate
        cw1 = error * j[1] * learning_rate
        #cw2 = error * inputs[2] * learning_rate

        w0 += cw0
        w1 += cw1
        #w2 += cw2

        print(str(error) + '     ' + str(output))
        t+=1

print()
print(str(w0)+'  '+str(w1))#+'   '+str(w2))