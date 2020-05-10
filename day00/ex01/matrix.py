import sys
sys.path.append('../ex00')
from vector import Vector

class Matrix:
    def __init__(self, arg):
        self.data = []
        self.shape = ()
        if isinstance(arg, tuple):
            if arg[0] < 1 or arg[1] < 1:
                print("Sorry, cannot create a matrix of those dimensions")
                return
            for r in range(arg[0]):
                nl = []
                for c in range(arg[1]):
                    nl.append(0)
                self.data.append(nl)
            self.shape = arg
        elif isinstance(arg,list):
            for nl in arg:
                self.data.append(nl)
                if len(nl) != len(arg[0]):
                    print("Sorry, cannot create a matrix of those dimensions")
                    self.data = []
                    return
            self.shape = (len(arg), len(nl))

    def __str__(self):
        return self.__class__.__name__ + str(self.data)

    def __repr__(self):
        return self.__class__.__name__ + str(self.data)

    def __add__(self, other):
        ret = []
        if isinstance(other, Matrix) and other.shape == self.shape:
            for r in range(self.shape[0]):
                nl = []
                for c in range(self.shape[1]):
                    nl.append(self.data[r][c] + other.data[r][c])
                ret.append(nl)
        else:
            print("Sorry, cannot add matrixes of those dimensions")
            return
        return(ret)
    
    def __radd__(self, other):
        return (self.__add__(other))
    
    def __sub__(self, other):
        ret = []
        if isinstance(other, Matrix) and other.shape == self.shape:
            for r in range(self.shape[0]):
                nl = []
                for c in range(self.shape[1]):
                    nl.append(self.data[r][c] - other.data[r][c])
                ret.append(nl)
        else:
            print("Sorry, cannot substract matrixes of those dimensions")
            return
        return(ret)
    
    def __rsub__(self, other):
        return (self.__sub__(other))

    def __truediv__(self,other):
        ret = []
        if isinstance(other, int) and other > 0:
            for r in range(self.shape[0]):
                nl = []
                for c in range(self.shape[1]):
                    nl.append(self.data[r][c] /other)
                ret.append(nl)
        else:
            print("Sorry, cannot substract matrixes of those dimensions")
            return
        return(ret)
    
    def __rtruediv__(self, other):
        return(self.__truediv__(other))

    def __mul__(self, other):
        ret = []
        if isinstance(other, int) and other > 0:
            for r in range(self.shape[0]):
                nl = []
                for c in range(self.shape[1]):
                    nl.append(self.data[r][c] * other)
                ret.append(nl)
        elif isinstance(other, Matrix) and self.shape[1] == other.shape[0]:
            for r in range(self.shape[0]):
                nl = []
                for c in range(self.shape[1]):
                    res = 0
                    for n in range(self.shape[1]):
                        res += self.data[r][n] * other.data[n][c]
                    nl.append(res)
                ret.append(nl)
        elif isinstance(other, Vector) and self.shape[1] == other.size:
            for r in range(self.shape[0]):
                nl = []
                res = 0
                for c in range(self.shape[1]):
                    res += self.data[r][c] * other.values[c]
                nl.append(res)
                ret.append(nl)
        else:
            print("Sorry, cannot multiply matrixes of those dimensions")
            return
        return(ret)