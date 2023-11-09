import numpy as np

def Sk1(x, k):
    """Applies soft thresholding to the vector x with threshold k."""
    return np.sign(x - k) * np.maximum(np.abs(x) - k, 0)z

def Sk2(x, k):
  k = float(k)
  for i in range(len(x[0])):
    if x[i][0] > k:
      x[i][0] -= k
    elif x[i][0] < -k:
      x[i][0] += k
    else:
      x[i][0] = 0
  return x

def main():
  #compare Sk1 and Sk2 for a bunch of vectors
  for i in range(100):
    x = np.random.rand(100, 1)
    k = np.random.rand()
    assert np.allclose(Sk1(x, k), Sk2(x, k))
  print("Sk1 and Sk2 are equivalent.")

if __name__ == "__main__":
  main()