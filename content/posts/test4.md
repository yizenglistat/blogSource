---
date: 2012-09-28
lastmod: 2013-02-02
linktitle: Testing D
title: Testing D
weight: 1
draft: true
tags: ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11", "tag12", "tag13", "tag14", "tag15", "tag16", "tag17", "tag18", "pytorch", "neural networks", "python"]
categories: ["Series 1"]
next: /testing-b
prev: /testing-c
---

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

<!--more-->

One-Hot Encoding takes a single integer and produces a vector where **a single element is 1** and **all other elements are 0**, like $[0, 1, 0, 0]$.

For example, imagine we're working with **categorical data**, where only a limited number of colors are possible: red, green, or blue. One way we could represent this numerically is by assigning each color a number:

| Color | Value |
| --- | --- |
| Red | 0 |
| Green | 1 |
| Blue | 2 |

This is known as **integer encoding**. For Machine Learning, this encoding can be problematic - in this example, we're essentially saying "green" is the _average_ of "red" and "blue", which can lead to weird unexpected outcomes.

It's often more useful to use the **one-hot encoding** instead:

| Color | Integer Encoding | One-Hot Encoding |
| --- | --- | --- |
| Red | 0 | $[1, 0, 0]$ |
| Green | 1 | $[0, 1, 0]$ |
| Blue | 2 | $[0, 0, 1]$ |

This is much more useful to pass into something like a [neural network](/blog/intro-to-neural-networks/).

## One-Hot Encoding in Python

Below are several different ways to implement one-hot encoding in Python.

### scikit-learn

Using [scikit-learn](https://scikit-learn.org/stable/)'s [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html):

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
print(encoder.fit_transform([['red'], ['green'], ['blue']]))
'''
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
 '''
```

### Keras

Using [Keras](https://keras.io/)'s [to_categorical](https://keras.io/utils/#to_categorical):

```python
from keras.utils import to_categorical

print(to_categorical([0, 1, 2]))
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
 '''
```

### NumPy

Using [NumPy](https://numpy.org/):

```python
import numpy as np

arr = [2, 1, 0]
max = np.max(arr) + 1
print(np.eye(max)[arr])
'''
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
'''
```