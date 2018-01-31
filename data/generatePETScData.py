from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix 

def run():
  m_train = 10000
  m_val = 1000
  m_test = 1000
  n = 1000
  
  A_train = csr_matrix(np.random.rand(m_train, n))
  A_val = csr_matrix(np.random.rand(m_val, n))
  A_test = csr_matrix(np.random.rand(m_test, n))

  d = 100 # the number of r.h.s 
  B_train = np.random.rand(m_train, d)
  B_val = np.random.rand(m_val, d)
  B_test = np.random.rand(m_test, d)
  
  # save operator TVT split in PETSc sparse format 
  A_train_PETSc = PETSc.Mat().createAIJ(size = A_train.shape,
                            csr = (A_train.indptr, A_train.indices,
                            A_train.data))
  viewer = PETSc.Viewer().createBinary('A_train.dat', 'w')
  viewer(A_train_PETSc)
  # now validation
  A_val_PETSc = PETSc.Mat().createAIJ(size = A_val.shape,
                            csr = (A_val.indptr, A_val.indices,
                            A_val.data))
  viewer = PETSc.Viewer().createBinary('A_val.dat', 'w')
  viewer(A_val_PETSc)
  # now test
  A_test_PETSc = PETSc.Mat().createAIJ(size = A_test.shape,
                            csr = (A_test.indptr, A_test.indices,
                            A_test.data))
  viewer = PETSc.Viewer().createBinary('A_test.dat', 'w')
  viewer(A_test_PETSc)
  
  #save the generated r.h.s. TVT split as PETSc dense format
  B_train_PETSc = PETSc.Mat()
  B_train_PETSc.createDense([B_train.shape[0], B_train.shape[1]], array = B_train)
  viewer = PETSc.Viewer().createBinary('B_train.dat', 'w')
  viewer(B_train_PETSc)
  
  B_val_PETSc = PETSc.Mat()
  B_val_PETSc.createDense([B_val.shape[0], B_val.shape[1]], array = B_val)
  viewer = PETSc.Viewer().createBinary('B_val.dat', 'w')
  viewer(B_val_PETSc)
  
  B_test_PETSc = PETSc.Mat()
  B_test_PETSc.createDense([B_test.shape[0], B_test.shape[1]], array = B_test)
  viewer = PETSc.Viewer().createBinary('B_test.dat', 'w')
  viewer(B_test_PETSc)
  
if __name__ == '__main__':
  run()