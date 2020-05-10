
class Vector:
    def __init__(self, num):
        self.values = []
        self.size = 0
        if isinstance(num, list):
            self.values = num
            self.size = len(num)
        elif isinstance(num,tuple):
            for i in range(num[0],num[1] + 1):
                self.values.append(i)
            self.size = num[1] - num[0]
        else:
            for i in range(num):
                self.values.append(i)
            self.size = num    

    def __str__(self):
        return self.__class__.__name__ + str(self.values)

    def __repr__(self):
        return self.__class__.__name__ + str(self.values)
    
    def __add__(self, other):
        ret = []
        n = 0
        if isinstance(other, int) and len(self.values) == 1:
            ret[0] = self.values[0] + other
        elif isinstance(other, Vector) and len(self.values) == len(other.values):
            for val in self.values:
                ret.append(other.values[n] + val)
                n += 1
        else:
            print("Sorry, it is impossible to add those values")
            return
        return(ret)
    
    def __radd__(self, other):
        return (self.__add__(other))

    def __sub__(self, other):
        ret = []
        n = 0
        if isinstance(other, int) and len(self.values) == 1:
            ret[0] = self.values[0] - other
        elif isinstance(other, Vector) and len(self.values) == len(other.values):
            for val in self.values:
                ret.append(val - other.values[n])
                n += 1
        else:
            print("Sorry, it is impossible to substract those values")
            return
        return(ret)
    
    def __rsub__(self, other):
        return (self.__sub__(other))

    def __truediv__(self,other):
        ret = []
        if isinstance(other, int):
            for val in self.values:
                ret.append(val/other)
        else:
            print("Sorry, it is impossible to divide those values")
            return
        return(ret)

    def __rtruediv__(self, other):
        return (self.__truediv__(other))

    def __mul__(self,other):
        ret = []
        n = 0
        if isinstance(other, int):
            for val in self.values:
                ret.append(val * other)
        elif isinstance(other, Vector) and len(self.values) == len(other.values):
            for val in self.values:
                ret.append(val * other.values[n])
                n += 1
        else:
            print("Sorry, it is impossible to multiply those values")
            return
        return(ret)

    def __rmul__(self, other):
        return (self.__mul__(other))


