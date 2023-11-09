import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# soft thresholding operator for a vector
def Sk(x, k):
  """Applies soft thresholding to the vector x with threshold k."""
  return np.sign(x) * np.maximum(np.abs(x) - k, 0)


def admm(A, b, epsilon=10e-5, r=50, maxiter=1000):
  r = float(r)
  n = len(A[0])
  m = len(b)
  xk = np.zeros((n, 1))
  zk = np.zeros((m, 1))
  uk = np.zeros((m, 1))

  if m != A.shape[0]:
    raise ValueError("The number of rows in A must equal the number of elements in b")

  AtA_inv = np.linalg.inv(A.T.dot(A))
  At = A.T
  Atb = At.dot(b)

  for iteration in range(maxiter):
    xkp1 = AtA_inv.dot(Atb + At.dot(zk - uk))
    Axkp1 = A.dot(xkp1)
    zkp1 = Sk(Axkp1 - b + uk, 1./r)
    ukp1 = uk + Axkp1 - zkp1 - b

    # early stopping condition: (l2 norm of ((xkp1, zkp1, ukp1) - (xk, zk, uk))) / (l2 norm of (xkp1, zkp1, ukp1))  < epsilon
    if np.linalg.norm(np.concatenate([xkp1, zkp1, ukp1]) - np.concatenate([xk, zk, uk])) / np.linalg.norm(np.concatenate([xkp1, zkp1, ukp1])) < epsilon:
      break

    # Prepare for next iteration
    xk, zk, uk = xkp1, zkp1, ukp1

  print(f"ADMM iterations: {iteration + 1}")

  return xkp1, zkp1, ukp1

def main():
  if len(sys.argv) != 2:
    print("Usage: python3 admm.py <dataname>")
    sys.exit(1)
  dataname = sys.argv[1]
  if dataname != 'big':
    filename = dataname+'.csv'
    data = np.loadtxt(filename, delimiter=',')
    # A is the matrix composed of all the first columns of the data excluding the last one, with the first row filled with ones if data is 2D
    if len(data[0]) == 2:
      A = np.ones((len(data), len(data[0])))
      A[:, 1:] = data[:, :-1]
    else:
      A = data[:, :-1]

    # b is the last column
    b = data[:, -1:]

    dimension = len(data[0])
  elif dataname == 'big':
    A = np.loadtxt('LAV1000x100/lav_A_1000x100.dat', delimiter=' ')
    b = np.loadtxt('LAV1000x100/lav_b_1000x1.dat', delimiter=' ')
    b = b.reshape((len(b), 1))
    dimension = 100

  r = 50 # between 1 and 100
  epsilon = 0.0001 # 10e-5 or 10e-6
  maxiter = 1000

  x, z, u = admm(A, b, epsilon, r, maxiter)

  # Use regular least squares using sklearn
  reg = LinearRegression().fit(A, b)
  print(f"sklearn: {reg.coef_}")
  print(f"admm: {x.T}")

  admmPredictions = A.dot(x)
  sklearnPredictions = reg.predict(A)
  admmL2error = np.linalg.norm(admmPredictions - b)
  admmAbsError = np.linalg.norm(admmPredictions - b, ord=1)
  sklearnL2error = np.linalg.norm(sklearnPredictions - b)
  sklearnAbsError = np.linalg.norm(sklearnPredictions - b, ord=1)
  print("ADMM")
  print(f"  * L2 error: {admmL2error}")
  print(f"  * absolute error: {admmAbsError}")
  print("sklearn")
  print(f"  * L2 error: {sklearnL2error}")
  print(f"  * absolute error: {sklearnAbsError}")

  # Plot differently depending on dimension of data
  if dimension == 2:
    plt.title(f"ADMM vs sklearn on {dataname} dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    # Plot the data and the regression line of the admm model
    plt.scatter(data[:, 0], data[:, 1], color='black', label='data')
    plt.plot(data[:, 0], admmPredictions, color='red', linewidth=2, label='admm')

    # Plot the data and the regression line of the sklearn model
    plt.plot(data[:, 0], sklearnPredictions, color='blue', linewidth=2, label='sklearn')
    plt.legend()
    plt.show()
  elif dimension == 3:
    # Plot the data and the regression line of the admm model
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='black', label='data')
    ax.plot(data[:, 0], data[:, 1], admmPredictions, color='red', linewidth=2, label='admm')

    # Plot the data and the regression line of the sklearn model
    ax.plot(data[:, 0], data[:, 1], sklearnPredictions, color='blue', linewidth=2, label='sklearn')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(f"ADMM vs sklearn on {data} dataset")
    plt.legend()
    plt.show()


if __name__ == '__main__':
  main()
