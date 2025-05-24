import numpy as np
import dezero.functions as F
from dezero import Variable

x = Variable(np.array([[1,2,3], [4,5,6]]))
c = Variable(np.array([[10,20,30], [40,50,60]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)