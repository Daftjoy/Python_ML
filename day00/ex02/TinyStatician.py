import math

class TinyStatician:
    @staticmethod
    def mean(x):
        sum = 0
        for i in x:
            sum += i
        return(sum/len(x))
    
    @staticmethod
    def median(x):
        x.sort()
        return(x[int(len(x)/2)])

    @staticmethod
    def quartile(x, q):
        x.sort()
        if q == 25:
            return(x[int(len(x)/4)])
        if q == 75:
            return(x[3 * int(len(x)/4)])

    @staticmethod
    def var(x):
        err = 0
        sum = 0
        for i in x:
            sum += i
        mean =sum/len(x)
        for i in x:
            err += (i - mean)**2
        return(err/len(x))
    
    @staticmethod
    def std(x):
        err = 0
        sum = 0
        for i in x:
            sum += i
        mean =sum/len(x)
        for i in x:
            err += (i - mean)**2
        return(math.sqrt(err/len(x)))
