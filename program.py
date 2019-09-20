# -*- coding: utf-8 -*-
# Use CART for MNIST
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load Data
digits = load_digits()
data = digits.data

print("Size of data: ")
print(data.shape)
print("The Second metric: ")
print(digits.images[1])
print("Target: ")
print(digits.target[1])

# Split train set and test set
x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.25, random_state = 81)

# Standardization & Normalization
ps = preprocessing.StandardScaler()
train_ps_x = ps.fit_transform(x_train)
test_ps_x = ps.transform(x_test)

# CART Classification
CART_model = DecisionTreeClassifier()
CART_model.fit(train_ps_x, y_train)
predict_y = CART_model.predict(test_ps_x)
print('Accuracy: %0.4lf' % accuracy_score(predict_y, y_test))

# Display image
plt.gray()
plt.imshow(digits.images[1])
plt.show()