import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

# 테스트용 x 범위 정의
x_plot = np.linspace(0, 1, 100).reshape(100, 1)
x_plot_var = Variable(x_plot)
y_plot = predict(x_plot_var)

# 시각화
plt.figure(figsize=(8, 4))
plt.scatter(x.data, y.data, label='Training data', color='orange', alpha=0.6)
plt.plot(x_plot, y_plot.data, label='Prediction', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
