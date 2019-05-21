import random

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []

        #fill the matrix with zeros
        for i in range(self.rows):
            self.matrix.append([])
            for j in range(self.cols):
                #filling the matrix with zeros as a default
                self.matrix[i].append(0)

    #for matrix scalar multiplication
    def scalar_mult(self, n):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] *= n

    #for matrix plus matrix and matrix plus scalar
    #only changes this matrix
    def add(self, n):
        if type(n) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += n.matrix[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += n

    #randomizes the numbers in the matrix to a floating point value between the two arguments
    #you can set the matrix with all equal values if you put in the same number for both parameters
    def randomize(self, lower, upper):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.uniform(lower, upper)

p = Matrix(3, 2)
u = Matrix(3, 2)

p.randomize(1,1)
u.randomize(2,2)

p.add(u)
print(p.matrix)
# p.randomize(-2,2)
# print(p.matrix)
# p.scalar_add(2)
# p.scalar_mult(4)
# print(p.matrix)