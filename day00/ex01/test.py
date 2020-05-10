from matrix import Matrix
import sys
sys.path.append('../ex00')
from vector import Vector

m1 = Matrix((3,3))
m2 = Matrix([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
m3 = Matrix([[14.0, 0.0, 2.0], [4.0, 4.0, 5.0], [6.0, 75.0, 8.0]])
m4 = Matrix([[1, 2, 3], [0, 0, 0]])
v1 = Vector([0.0, 1.0, 2.0, 4.0])
print(m4 * v1)
"""
print(a)
b = Vector([0.0, 1.0, 2.0, 3.0])
c = Vector(3)
print(a / 4)
print(a * c)
"""