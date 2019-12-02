# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### numPy


import numpy as np

# Python code
result = 0
for i in range(100):
    print(i)
    result += i

# List
L = list(range(10))

# string list
L2 = [str(c) for c in L]

# mixed list
L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]

# fixed arrays
import array as ar

L = list(range(10))
A = ar.array('i', L)

# create array from list
np.array([1, 4, 2, 5, 3])
np.array([3.14, 4, 2, 3])

# set type
np.array([1, 2, 3, 4], dtype='float32')

# nested list is a multidim array
np.array([[2, 3, 4],
          [4, 3, 2],
          [6, 7, 8]])

# create array from scratch
np.zeros(10, dtype=int)
np.ones((3, 5), dtype=float)
np.full((3, 5), 3.14)
np.arange(0, 20, 2)
np.linspace(0, 1, 5)
np.random.random((3, 3))
np.random.normal(0, 1, (3, 3))
np.random.randint(0, 10, (3, 3))
np.eye(6)

# numpy array attributes
np.random.seed(0)
x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")

# Array Indexing: Accessing Single Elements
x1[0]

x1[-1]

x2[2, 0]

x2[0, 0] = 12

# Array Slicing: Accessing Subarrays
x = np.arange(10)
x[5:]
x[4:7]
x[::2]
x[1::2]
x[::-1]
x[5::-2]

x2[:2,:3]
x2[:3, ::2]
x2[::-1, ::-1]

# Accessing array rows and columns
print(x2[:,0])
print(x2[0])

# Subarrays as no-copy views
x2_sub = x2[:2, :2]
print(x2_sub)

# Creating copies of arrays
x2_sub_copy = x2[:2,:2].copy()

# Reshaping of Arrays
grid = np.arange(1,10).reshape((3,3))
x = np.array([1,2,3])
x.reshape((1, 3))
x.reshape((3, 1))

# Concatenation of arrays
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
# concatenate along the first axis
np.concatenate([grid, grid])

np.concatenate([grid, grid], axis=1)

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])

y = np.array([[1],
             [2]])
np.hstack([grid, y])

# Splitting of arrays

x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
grid

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print(left)
print(right)

# not efficient looping
def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output


values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

big_array = np.random.randint(1, 100, size=1000000)
#%timeit compute_reciprocals(big_array)

# Introducing UFuncs
np.arange(5) / np.arange(1, 6)
x = np.arange(9).reshape((3, 3))
x ** 2

x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)
-(0.5*x + 1) ** 2

np.add(x, 2)

x = np.array([-2, -1, 0, 1, 2])
np.absolute(x)
abs(x)

# trigonometry
theta = np.linspace(0, np.pi, 3)
print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

# Exponents and logarithms
x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))

x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

# speciazed ufuncs

from scipy import special
# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))

x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)  =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))

# Specifying output
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

x = np.arange(1, 6)
np.add.reduce(x)

# keep intermediate results
np.add.accumulate(x)

# Outer products
x = np.arange(1, 6)
np.multiply.outer(x, x)

# sum value sin an array
L = np.random.random(100)
sum(L)

big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)

%timeit min(big_array)
%timeit np.min(big_array)


M = np.random.random((3, 4))
print(M)
#by column
M.sum(axis=0)
#by row
M.sum(axis=1)

#average height of us president
import pandas as pd

data = pd.read_csv('data/president_height.csv')
heights= np.array(data['height(cm)'])
print("Mean height:       ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height:    ", heights.min())
print("Maximum height:    ", heights.max())
print("25th percentile:   ", np.percentile(heights, 25))
print("Median:            ", np.median(heights))
print("75th percentile:   ", np.percentile(heights, 75))

import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # set plot style

plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');
plt.show()

# broadcasting
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

a + 5
M = np.ones((3,3))
M + a

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a + b

M = np.ones((2, 3))
a = np.arange(3)

M.shape=(2, 3)
a.shape=(1, 3)

a = np.arange(3).reshape((3, 1))
b = np.arange(3)

a+b

np.logaddexp(M, a[:, np.newaxis])

# center an array

X= np.random.random((10,3))
Xmean = X.mean(0)

X_centered = X - Xmean

# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
           cmap='viridis')
plt.colorbar();
plt.show()

#comparison operations
x = np.array([1, 2, 3, 4, 5])
x < 3
x != 3
x == 3
(2 * x) == (x ** 2)
np.less(x,3)

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
x
x < 6

np.count_nonzero(x < 6)
np.sum(x < 6)
# how many values less than 6 in each row?
np.sum(x < 6, axis=1)
np.any(x > 8)
np.all(x < 10)
np.all(x == 6)
np.all(x < 8, axis=1)
 # mask arrays
x[x<5]

#rainfall dataset
# use pandas to extract rainfall inches as a NumPy array
import numpy as np
import pandas as pd
rainfall = pd.read_csv('data/Seattle2014.csv',sep=',')['PRCP'].values
inches = rainfall / 254.0  # 1/10mm -> inches
inches.shape

np.sum((inches > 0.5) & (inches < 1))
print("Number days without rain:      ", np.sum(inches == 0))
print("Number days with rain:         ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches  :", np.sum((inches > 0) &
                                                (inches < 0.2)))

# construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2014 (inches):   ",
      np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):  ",
      np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ",
      np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):",
      np.median(inches[rainy & ~summer]))

# fancy indexing
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)

[x[3],x[7],x[9]]
ind = [3,6,7]
x[ind]

ind = np.array([[3, 7],
                [4, 5]])
x[ind]

X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

X[row[:, np.newaxis], col]


row[:, np.newaxis]*col

# combined indexing
print(X)
X[2,[2,0,1]]
X[1:,[2,0,1]]

# masking indexing
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:,np.newaxis], mask]

# select random points
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # for plot styling

plt.scatter(X[:, 0], X[:, 1]);


indices = np.random.choice(X.shape[0], 20, replace=False)
indices

selection = X[indices]  # fancy indexing here
selection.shape

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='none', s=200);

# modify values with indexing
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)

x[i] -= 10
print(x)

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)

# binning
np.random.seed(42)
x = np.random.randn(100)

# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)

# add 1 to each of these bins
np.add.at(counts, i, 1)

# plot the results
plt.plot(bins, counts, linestyle='steps')

# short arrays
import numpy as np
x=np.array([2,1,4,3,5])
x.sort()
np.sort(x)
print(x)

# indexes of sorted elements
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)

# sort along rows and columns
rand = np.random.RandomState(42)
X = rand.randint(0,10,(4, 6))
print(X)

np.sort(X,axis=0)
np.sort(X,axis=1)

# partial sorts partioning - smallest K values
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x ,3)

np.partition(X, 2, axis=1)


# K-nn
X=rand.rand(10,2)
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)

nearest = np.argsort(dist_sq, axis=1)
print(nearest)

# now remove first column - self point and get 2 nearest
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

# stuctured arrays
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

data[-1]['name']

data[data['age'] < 30]['name']

np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
