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

    #for matrix multiplied by scalar or matrix multiplied by matrix (matrix multiplication)
    #returns a new result matrix if matrix multiplication is done
    def mult(self, n):
        if type(n) == Matrix:
            #check if the columns of the first one equal the rows of the second one
            #if yes, then do a matrix multiplication
            if self.cols != n.rows:
                print("The columns of the first matrix need to be equal to the rows in the second matrix")
                return None
            else:
                #make a new matrix to return
                result = Matrix(self.rows, n.cols)

                #iterate over the rows of the result matrix
                for i in range(result.rows):
                    #iterate over the columns of the result matrix
                    #we are assembling the matrix by filling in each column of the first row, then filling in each column of the second row, ...
                    for j in range(result.cols):
                        #the sum is the numnber that will be filled into the spot of the result matrix (which is where we are in the two above loops)
                        sum = 0
                        for k in range(self.cols):
                            sum += self.matrix[i][k] * n.matrix[k][j]

                        #store the sum in the spot in the result matrix
                        result.matrix[i][j] = sum

                #return the matrix which is the product of the two matrices
                return result


        #for scalar multiplication
        #doesn't return anything but changes the matrix itself
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] *= n


    #for matrix plus scalar or matrix plus matrix
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

    #sets the matrix to the values in n
    def setMatrix(self, n):
        #store these values so we don't need to keep calling the len function
        number_of_rows = len(n)
        number_of_cols = len(n[0])

        if self.rows == number_of_rows and self.cols == number_of_cols:
            for i in range(number_of_rows):
                for j in range(number_of_cols):
                    self.matrix[i][j] = n[i][j]
        else:
            print("Matrix values can't be set, the argument to this function must have the same shape as this matrix")
            return None

    #adding a printing functionality for easier debugging
    def print(self):
        print(self.matrix)

p = Matrix(2,2)
r = Matrix(2,2)

p.setMatrix([[2,3],[3,4]])
r.setMatrix([[1,0],[2,6]])

x = p.mult(r)

x.print()