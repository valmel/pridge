import numpy as np
from pridge import LSRnumpy

def run():
  print('defining test problem...')
  A_train = np.eye(2)
  A_val = np.asarray([[0.4, 0.7]])
  A_test = np.asarray([[0.2, 0.1]])
 
  B_train = np.asarray([[1.], [1.]])
  B_val = np.asarray([[1.1]])
  B_test = np.asarray([[0.3]])
  
  print('setting up the least square model...')
  ls = LSRnumpy()
  ls.setVerbosity(0)
  ls.setOperator(A_train, A_val, A_test)
  ls.setRhs(B_train, B_val, B_test)
  ls.setReg(0.1)
    
  print('training the least square model...')
  ls.train()
  X = ls.getEstimate()
  print('X = ', X)
  print(ls.regs)
  
if __name__ == '__main__':
  run()