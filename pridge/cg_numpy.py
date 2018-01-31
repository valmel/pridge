def CGnumpyAAt(A, At, b, x, reg = 0.01, imax = 1000, eps = 1e-6):
  # project b by A^T
  rhs = At.dot(b)
  q = rhs.copy()
  
  i = 0
  aux = A.dot(x)
  r = rhs - At.dot(aux) - reg*x
  d = r.copy()
  delta_0 = r.dot(r)
  delta = delta_0
  while i < imax and \
    delta > delta_0 * eps**2:
    aux = A.dot(d)
    q = At*aux
    q = q + reg*d
    alpha = delta / d.dot(q)
    x = x + alpha*d
    r = r - alpha*q
    delta_old = delta
    delta = r.dot(r)
    beta = delta / delta_old
    d = r + beta*d
    i = i + 1
  return x, i, delta**0.5

def CGnumpyN(N, b, x, reg = 0.01, imax = 1000, eps = 1e-6):
  i = 0
  r = b - N.dot(x) - reg*x
  d = r.copy()
  delta_0 = r.dot(r)
  delta = delta_0
  while i < imax and \
    delta > delta_0 * eps**2:
    q = N.dot(d) + reg*d
    alpha = delta / d.dot(q)
    x = x + alpha*d
    r = r - alpha*q
    delta_old = delta
    delta = r.dot(r)
    beta = delta / delta_old
    d = r + beta*d
    i = i + 1
  return x, i, delta**0.5