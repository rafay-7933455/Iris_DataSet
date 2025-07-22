import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

# OLS Method
def ols(x_train, y_train):
     x = np.matrix(x_train)
     y = np.matrix(y_train)
     b = np.linalg.inv(x.T @ x) @ x.T @ y
     return b
     
def ols_model(x_train, x_test, y_train):
     b = ols(x_train, y_train)
     y_pred = x_test @ b
     return y_pred

# Batch Gradient Descent Method
def batch_gd(x_train, y_train, iter = 10, n=0.01):
     x = np.matrix(x_train)
     y = np.matrix(y_train)
     b = np.matrix(np.zeros(x.shape[1]).reshape(-1, 1))
     for i in range(iter):
          y_pred = x @ b
          error = y_pred - y
          grad = (2/y.shape[0]) * x.T @ error
          b -= n * grad
     return b

def batch_gd_model(x_train, x_test, y_train, iter = 10, n=0.01):
     b = batch_gd(x_train, y_train, iter, n)
     y_pred = x_test @ b
     return y_pred

# Main Program
iris = load_iris()
iris_data = pd.DataFrame(columns=iris.feature_names)# type: ignore
iris_data[iris_data.columns] = iris.data# type: ignore

iris_data['target'] = iris.target#type:ignore

x=iris_data.drop(columns='target')
x.insert(loc=0, column='intercept', value=1)

y=iris_data['target']

#Training Models
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.3)

y_pred = ols_model(np.asarray(x_train), np.asarray(x_test), np.asarray(y_train).reshape(-1,1))
y_pred_ = (np.asarray(y_pred).reshape(1,y_pred.shape[0]))[0]

y_pred_gd = batch_gd_model(np.asarray(x_train), np.asarray(x_test), np.asarray(y_train).reshape(-1,1), iter=10000)
y_pred__ = (np.asarray(y_pred_gd).reshape(1,y_pred.shape[0]))[0]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

y_pred2=model.predict(x_test)

# Converting Outputs into DataFrame
y_dict={
     'y':y_test,
     'y_pred_':y_pred_,
     'y_pred__':y_pred__, 
     'y_pred2':y_pred2
}

out = pd.DataFrame(y_dict, columns=['y','y_pred_','y_pred__','y_pred2'])
print(out)

# Comparing MSE and r2_scores
from sklearn.metrics import mean_squared_error, r2_score, precision_score

print(f'My OLS model: MSE={mean_squared_error(y_test, np.asarray(y_pred_))}, r2_score={r2_score(y_test, np.asarray(y_pred_))}')
print(f'My Batch_GD model: MSE={mean_squared_error(y_test, np.asarray(y_pred__))}, r2_score={r2_score(y_test, np.asarray(y_pred__))}')
print(f'His model: MSE={mean_squared_error(y_test, np.asarray(y_pred2))}, r2_score={r2_score(y_test, np.asarray(y_pred2))}')

# Visualizing and Comparing The three models
import matplotlib.pyplot as plt

sorted_idx = x_test['sepal length (cm)'].argsort()

fig, ax = plt.subplots(3, 1, figsize=(8, 6))

ax[0].scatter(x_test['sepal length (cm)'].iloc[sorted_idx], y_test.iloc[sorted_idx])
ax[0].plot(x_test['sepal length (cm)'].iloc[sorted_idx], y_test.iloc[sorted_idx], linestyle='--', color='darkblue')
ax[0].plot(x_test['sepal length (cm)'].iloc[sorted_idx], y_pred__[sorted_idx], color='yellowgreen', linestyle='dashed')
ax[0].set_title('My Custom Built Linear Regression by Batch-GD Method')

ax[1].scatter(x_test['sepal length (cm)'].iloc[sorted_idx], y_test.iloc[sorted_idx])
ax[1].plot(x_test['sepal length (cm)'].iloc[sorted_idx], y_test.iloc[sorted_idx], linestyle='--', color='darkblue')
ax[1].plot(x_test['sepal length (cm)'].iloc[sorted_idx],y_pred_[sorted_idx] , color='red', linestyle='dashed')
ax[1].set_title('My Custom Built Linear Regression by OLS Method')

ax[2].scatter(x_test['sepal length (cm)'].iloc[sorted_idx], y_test.iloc[sorted_idx])
ax[2].plot(x_test['sepal length (cm)'].iloc[sorted_idx], y_test.iloc[sorted_idx], linestyle='--', color='darkblue')
ax[2].plot(x_test['sepal length (cm)'].iloc[sorted_idx], y_pred2[sorted_idx], color='green', linestyle="dashed")
ax[2].set_title('sklearn Linear Regression')

plt.tight_layout()
plt.show()