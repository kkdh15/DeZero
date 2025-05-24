import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()
    
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
        
        
# 정렬된 x값으로 예측 보기
x_plot = np.linspace(0, 1, 100).reshape(100, 1)
x_plot_var = Variable(x_plot)
y_plot = predict(x_plot_var)

# Plot
plt.figure(figsize=(8, 4))
plt.scatter(x, y, label='Training data', color='orange', alpha=0.6)
plt.plot(x_plot, y_plot.data, label='Prediction', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression with 1-hidden-layer MLP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()