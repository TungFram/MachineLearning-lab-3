import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tabulate
import sklearn

plt.style.use('ggplot')


def print_dataframe(dataframe):
    print(tabulate.tabulate(dataframe, headers='keys', numalign='center', stralign='left', tablefmt='fancy_grid', ))


def generate_data(n_points=20):
    """
    Принимает на вход n_points точек 
    Возвращает данные для обучения и теста
  """
    X = np.linspace(-5, 5, n_points)
    y = 10 * X - 7

    X_train = X[0::2].reshape(-1, 1)
    # X_train = X[0::2]
    # X_train = np.insert(X_train, 0, 1).reshape(-1, 1)

    y_train = y[0::2] + np.random.randn(int(n_points / 2)) * 10
    # y_train = np.insert(y_train, 0, 1)

    X_test = X[1::2].reshape(-1, 1)
    # X_test = X[1::2]
    # X_test = np.insert(X_test, 0, 1).reshape(-1, 1)
    y_test = y[1::2] + np.random.randn(int(n_points / 2)) * 10
    # y_test = np.insert(y_test, 0, 1)

    print(f'Generated {len(X_train)} train samples and {len(X_test)} test samples.')
    return X, X_train, y_train, X_test, y_test


def get_determinant_of_matrix(matrix):
    det = np.linalg.det(matrix)
    return det


def get_inverse_matrix(matrix):
    inv = np.linalg.inv(matrix)
    return inv


def get_w_lsm(x, y):
    xT = x.T
    xTx = xT @ x

    if get_determinant_of_matrix(xTx) == 0:
        return np.NaN

    inv_xTx = get_inverse_matrix(xTx)
    w = inv_xTx @ xT @ y
    return w


def mse(true, pred):
    return np.mean((true - pred)**2)


def mae(true, pred):
    return np.mean(np.abs(true - pred))


X_original, X_train, y_train, X_test, y_test = generate_data(200)

### Реализуйте настройку w и b с помощью рассмотренного выше метода наименьших квадратов.
# Найдите значения метрик MSE и MAE. Сравните с результатами из sklearn

w_original = get_w_lsm(X_train, y_train)
y_train_predicted = X_train @ w_original
y_test_predicted = X_test @ w_original

train_absolute_error_original = mae(y_train, y_train_predicted)
train_squared_error_original = mse(y_train, y_train_predicted)

test_absolute_error_original = mae(y_test, y_test_predicted)
test_squared_error_original = mse(y_test, y_test_predicted)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
w_sklearn = model.coef_
bias_sklearn = model.intercept_

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae_sklearn = sklearn.metrics.mean_absolute_error(y_train, y_train_pred)
test_mae_sklearn = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)

train_mse_sklearn = sklearn.metrics.mean_squared_error(y_train, y_train_pred)
test_mse_sklearn = sklearn.metrics.mean_squared_error(y_test, y_test_pred)

print(f"\nMAE of manual calculation train = {round(train_absolute_error_original, 3)}")
print(f"MAE of sklearn calculation train = {round(train_mae_sklearn, 3)}")

print(f"\nMSE of manual calculation train = {round(train_squared_error_original, 3)}")
print(f"MSE of sklearn calculation train = {round(train_mse_sklearn, 3)}")

print(f"\nMAE of manual calculation test = {round(test_absolute_error_original, 3)}")
print(f"MAE of sklearn calculation test = {round(test_mae_sklearn, 3)}")

print(f"\nMSE of manual calculation test = {round(test_squared_error_original, 3)}")
print(f"MSE of sklearn calculation test = {round(test_mse_sklearn, 3)}")

print(f"\nw manual = {w_original}")
print(f"w sklearn = {w_sklearn}")

# print(f"\nEquation of the approximating line is: {w_original}x + {bias_sklearn}")

plt.figure(figsize=(16, 8))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X_original[1::2], X_original[1::2].reshape(-1, 1).dot(w_sklearn) + bias_sklearn, label='predicted')
plt.legend(loc='best')
plt.title('Our model')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

################################################ Task 3.2 ################################################
def generate_wave_set(n_support=1000, n_train=25, std=0.3):
    data = {}
    # выберем некоторое количество точек из промежутка от 0 до 2*pi
    data['support'] = np.linspace(0, 2*np.pi, num=n_support)
    # для каждой посчитаем значение sin(x) + 1
    # это будет ground truth
    data['values'] = np.sin(data['support']) + 1
    # из support посемплируем некоторое количество точек с возвратом, это будут признаки
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    # опять посчитаем sin(x) + 1 и добавим шум, получим целевую переменную
    data['y_train'] = np.sin(data['x_train']) + 1 + np.random.normal(0, std, size=len(data['x_train']))
    return data

data = generate_wave_set(1000, 250)

plt.figure(figsize=(16, 8))
margin = 0.3
plt.plot(data['support'], data['values'], 'b--', alpha=0.5, label='manifold')
plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
plt.xlim(data['x_train'].min() - margin, data['x_train'].max() + margin)
plt.ylim(data['y_train'].min() - margin, data['y_train'].max() + margin)
plt.legend(loc='upper right')
plt.title('True function and noised data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

### попробуйте реализовать настройку w и b с помощью рассмотренного выше метода наименьших квадратов.
### Найдите значения метрик MSE и MAE

X_train = data['x_train'].reshape(-1, 1)
X_original = data['support'].reshape(-1, 1)

w_original = get_w_lsm(X_train, data['y_train'])
y_train_predicted = X_train @ w_original
y_test_predicted = X_original @ w_original

train_absolute_error_original = mae(data['y_train'], y_train_predicted)
train_squared_error_original = mse(data['y_train'], y_train_predicted)

test_absolute_error_original = mae(data['values'], y_test_predicted)
test_squared_error_original = mse(data['values'], y_test_predicted)


model = LinearRegression()
model.fit(X_train, data['y_train'])
w_sklearn = model.coef_
bias_sklearn = model.intercept_

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_original)

train_mae_sklearn = sklearn.metrics.mean_absolute_error(data['y_train'], y_train_pred)
train_mse_sklearn = sklearn.metrics.mean_squared_error(data['y_train'], y_train_pred)

test_mae_sklearn = sklearn.metrics.mean_absolute_error(data['values'], y_test_pred)
test_mse_sklearn = sklearn.metrics.mean_squared_error(data['values'], y_test_pred)

print(f"\nMAE of manual calculation train = {round(train_absolute_error_original, 3)}")
print(f"MAE of sklearn calculation train = {round(train_mae_sklearn, 3)}")

print(f"\nMSE of manual calculation train = {round(train_squared_error_original, 3)}")
print(f"MSE of sklearn calculation train = {round(train_mse_sklearn, 3)}")

print(f"\nMAE of manual calculation test = {round(test_absolute_error_original, 3)}")
print(f"MAE of sklearn calculation test = {round(test_mae_sklearn, 3)}")

print(f"\nMSE of manual calculation test = {round(test_squared_error_original, 3)}")
print(f"MSE of sklearn calculation test = {round(test_mse_sklearn, 3)}")

print(f"\nw manual = {w_original}")
print(f"w sklearn = {w_sklearn}")

plt.figure(figsize=(16, 8))
margin = 0.3
plt.plot(data['support'], data['values'], 'b--', alpha=0.4, label='Original function')
plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.2, label='data')
plt.plot(data['support'], data['support'].reshape(-1, 1).dot(w_sklearn) + bias_sklearn, 'black', alpha=0.6, label='Approximating line')
plt.xlim(data['x_train'].min() - margin, data['x_train'].max() + margin)
plt.ylim(data['y_train'].min() - margin, data['y_train'].max() + margin)
plt.legend(loc='upper right')
plt.title('Our model')
plt.xlabel('x')
plt.ylabel('y')
plt.show()























