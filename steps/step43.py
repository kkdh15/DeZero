import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero import as_variable

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    y = F.linear_simple(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear_simple(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
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
