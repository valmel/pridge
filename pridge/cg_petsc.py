def CGpetsc(A, b, x, reg = 0.01, imax = 1000, eps = 1e-6):
  # project b by A^T
  rhs = x.duplicate() # num of chemical features
  A.multTranspose(b, rhs)
  r = rhs.duplicate()
  d = rhs.duplicate()
  q = rhs.duplicate()
  aux = b.duplicate()
  
  i = 0
  A.mult(x, aux)
  A.multTranspose(aux, r)
  r.axpy(reg, x)
  r.aypx(-1, rhs)
  r.copy(d)
  delta_0 = r.dot(r)
  delta = delta_0
  while i < imax and \
    delta > delta_0 * eps**2:
    A.mult(d, aux)
    A.multTranspose(aux, q)
    q.axpy(reg, d)
    alpha = delta / d.dot(q)
    x.axpy(+alpha, d)
    r.axpy(-alpha, q)
    delta_old = delta
    delta = r.dot(r)
    beta = delta / delta_old
    d.aypx(beta, r)
    i = i + 1
  return i, delta**0.5
        