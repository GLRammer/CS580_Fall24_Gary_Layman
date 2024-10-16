import matplotlib.pyplot as plt
import numpy as np
# Temporarily store csv as single array
tempin = np.genfromtxt('linear_regression_data.csv', delimiter=',')
# Bool iterator and temporary array structures
i=False
tX,tY = [],[]
# Iterate over tempin to build temporary arrays
for val in tempin:
    if i:
        tY.append(val)
        i=False
    else:
        tX.append(val)
        i=True
# Build arrays into usable form and grab means
X=np.array(tX)
Y=np.array(tY)
XMean=np.mean(X)
YMean=np.mean(Y)
# Calculate covariance and variance
covariance = np.sum((X - XMean) * (Y - YMean)) / len(X)
variance_X = np.sum((X - XMean) ** 2) / len(X)
# Build slope maths
m = covariance / variance_X
b = YMean - m * XMean
# Print the linear model in question
print(f"Linear Model: Y = {m:.2f}X + {b:.2f}")
# Find best fits
XFit = np.linspace(np.min(X), np.max(X), 100)
YFit = m * XFit + b
# Plug variables into plot
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(XFit, YFit, color='red', label='Fitted Line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.title('Linear Model using Covariance Approach')
plt.legend()
plt.grid()
plt.show()