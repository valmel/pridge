import numpy as np
from pridge import LSRnumpy

def run():
  ddir = '../data/'
  
  print('generating operator...')
  m_train = 10000
  m_val = 1000
  m_test = 1000
  n = 1000
  A_train = np.random.rand(m_train,n)
  A_val = np.random.rand(m_val,n)
  A_test = np.random.rand(m_test,n)

  d = 1000 # the number of r.h.s 
  B_train = np.random.rand(m_train,d)
  B_val = np.random.rand(m_val,d)
  B_test = np.random.rand(m_test,d)
  
  print('setting up the least square model...')
  ls = LSRnumpy()
  ls.setVerbosity(0)
  ls.setOperator(A_train, A_val, A_test)
  ls.setRhs(B_train, B_val, B_test)
  ls.setReg(0.1)
  ls.setOutputDir(ddir)
  
  print('training the least square model...')
  ls.train()
  ls.saveTrainingInfo()
  X = ls.getEstimate()
  print(X)
  
  print('saving the estimate...')
  ls.saveEstimate('X_numpy')
  print('computing the prediction...')
  ls.computePrediction()
  print('saving the prediction...')
  ls.savePrediction('P_numpy')
  
if __name__ == '__main__':
  run()