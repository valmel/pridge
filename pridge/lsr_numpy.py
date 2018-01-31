import numpy as np
import sys as sys
import os as os
import time as time
from pridge.cg_numpy import CGnumpyN, CGnumpyAAt

class LSRnumpy(object):
  def __init__(self):
    self.A_train =  None # the operator - train rows
    self.A_traint =  None # the operator - train rows transposed
    self.N = None # AAt if used 
    self.A_val =  None # the operator - val rows
    self.A_test =  None # the operator - test rows
    
    self.B_train = None # the r.h.s. - train rows
    self.B_val = None # the r.h.s. - val rows
    self.B_test = None # the r.h.s. - test rows
    self.X = None # the solution
    self.P = None # the prediction
    self.reg = 0.1 # initial regularization
    self.verbosity = 0
    self.usecg_N = True

  def initialize(self):
    if(self.A_train[0, 0] == None or self.B_train[0, 0] == None):
      print('initialize: please provide the operator and r.h.s. first...')
      sys.exit()
            
    self.X = np.zeros((self.A_train.shape[1], self.B_train.shape[1])) # solution
    self.P = np.zeros((self.A_train.shape[0], self.B_train.shape[1])) # prediction
    
    self.x_train = np.zeros(self.A_train.shape[1])
    self.b_train = np.zeros(self.A_train.shape[0])
    self.x_train_old = self.x_train
   
    self.x_val = np.zeros(self.A_val.shape[1])
    self.b_val = np.zeros(self.A_val.shape[0])
    self.x_test = np.zeros(self.A_test.shape[1])
    self.b_test = np.zeros(self.A_test.shape[0])
     
  def setVerbosity(self, ver = 1):
    self.verbosity = ver
    
  def setReg(self, reg):
    self.reg = reg
    
  def setOutputDir(self, odir = ''):
    if (not os.path.isdir(odir)):
      os.makedirs(odir)
    if (not os.path.isdir(odir)):
      print('setDataDir(): directory does not exist and can not be created...')  
      sys.exit()
    self.odir = odir
      
  def setOperator(self, A_train, A_val, A_test):
    if(not (('numpy' in str(type(A_train))) or ('scipy' in str(type(A_train))))):
      print('Operator has to be a matrix of numpy or scipy type...')
      sys.exit()
    self.A_train = A_train
    self.A_traint = A_train.transpose()
    if self.usecg_N == True:
      self.N = self.A_traint.dot(self.A_train)
    self.A_val = A_val
    self.A_test = A_test
   
  def setRhs(self, B_train, B_val, B_test):
    if not hasattr(self, 'A_train'):
      print('Please use setOperator before setRhs...')
      sys.exit()
    if(not (('numpy' in str(type(B_train))) or ('scipy' in str(type(B_train))))):
      print('Right hand side has to be a matrix of numpy or scipy type...')
      sys.exit()
    self.B_train = B_train
    self.B_val = B_val
    self.B_test = B_test
    
    # check the consistency of the operator A with B
    if(self.A_train.shape[0] != self.B_train.shape[0]):
      print('Operator and the r.h.s. have to have the same row dimension...')
      sys.exit()
      
  def res(self, A, x, b):
    r = A.dot(x) - b
    res = (r.dot(r))**0.5
    norm = (b.dot(b))**0.5
    return res/norm # relative residum
      
  def train(self):
    self.initialize()
    if (self.verbosity > 0):
      start = time.time()
      
    self.regs = np.zeros(self.B_train.shape[1])
    self.trainErrors = np.zeros(self.B_train.shape[1])
    self.valErrors = np.zeros(self.B_train.shape[1])
    self.testErrors = np.zeros(self.B_train.shape[1])
    self.niters = np.zeros(self.B_train.shape[1])
    self.rnorms = np.zeros(self.B_train.shape[1])
    
    per = self.B_train.shape[1]/100
    for cB in range(self.B_train.shape[1]): # runs through the columns of the r.h.s. B
      if (self.verbosity > -1 and cB%per == 0):
	      print(str(cB//per) + ' percent done.' )
      self.b_train = self.B_train[:, cB]
      self.b_val = self.B_val[:, cB]
      self.b_test = self.B_test[:, cB]
      reg = self.reg;
      self.x_train.fill(0.)
      valError = self.res(self.A_val, self.x_train, self.b_val) # with solution = 0
      valError_old =  2.*valError
      iters = None
      rnorm = None
      trainError = self.res(self.A_train, self.x_train, self.b_train)
      while valError < valError_old:
        if (self.verbosity > 1):
          print('reg = %g' % (reg))
        norm_b_train = (self.b_train.dot(self.b_train))**0.5
        if (self.verbosity > 1):
          print('norm_b_train = %g' % (norm_b_train))
        self.x_train_old = self.x_train
        iters_old = iters
        rnorm_old = rnorm
        if self.usecg_N == True:
          self.x_train, iters, rnorm = CGnumpyN(self.N, self.A_traint.dot(self.b_train), self.x_train, reg*norm_b_train**2) 
        else:
	        self.x_train, iters, rnorm = CGnumpyAAt(self.A_train, self.A_traint, self.b_train, self.x_train, reg*norm_b_train**2)
        trainError_old = trainError
        trainError = self.res(self.A_train, self.x_train, self.b_train)
        valError_old = valError
        valError = self.res(self.A_val, self.x_train, self.b_val)
        if (self.verbosity > 1):
          print('iters = %g,  error norm = %g' % (iters, rnorm))
          print('valError = %g' % (valError))
          print('trainError = %g' % (trainError))  
        reg = reg/2.
      self.X[:, cB] = self.x_train_old
      testError = self.res(self.A_test, self.x_train_old, self.b_test)
  
      self.regs[cB] = 4.*reg
      self.trainErrors[cB] = trainError_old
      self.valErrors[cB] = valError_old
      self.testErrors[cB] = testError
      self.niters[cB] = iters_old
      self.rnorms[cB] = rnorm_old
        
      if(self.verbosity > 0):
        print('r.h.s. column = %g' % (cB))
        print('regs[' + str(cB)+']= ' + str(self.regs[cB]))
        print('trainError = %g' % (self.trainErrors[cB]))
        print('valError = %g' % (self.valErrors[cB]))
        print('testError = %g' % (self.testErrors[cB]))
        print('iters = %g' % (self.niters[cB]))
        print('rnorm = %g\n' % (self.rnorms[cB]))
        
    if (self.verbosity > 0):
      print('took ' + str(time.time() - start))
      
  def saveTrainingInfo(self):
    np.save(self.odir + 'regs', self.regs)
    np.save(self.odir + 'train_errors', self.trainErrors)
    np.save(self.odir + 'val_errors', self.valErrors)
    np.save(self.odir + 'test_errors', self.testErrors)
    np.save(self.odir + 'niters', self.niters)
    np.save(self.odir + 'rnorms', self.rnorms)
    
  def getEstimate(self):
    if(self.X[0, 0] == None):
      print('saveEstimate: please compute the estimate first...')
    return self.X  
   
  def saveEstimate(self, fname = 'estX'):
    if(self.X[0, 0] == None):
      print('saveEstimate: please compute the estimate first...')
      sys.exit()
    if (self.odir == None):
      print('saveEstimate: please use setOutputDir to define the output directory...')
      sys.exit()
    np.save(self.odir + fname, self.X)
    if not os.path.isfile(self.odir + fname + '.npy'):
        print('saveEstimate: saving not succesfull...')
        
  def computePrediction(self):
    # the prediction of the model P = A*X
    if(self.X[0, 0] == None):
      print('computePrediction: please first train the least squares model by running train...')
      sys.exit()
    self.P = self.A_train.dot(self.X)
      
  def savePrediction(self, fname):
    if(self.P[0, 0] == None):
      print('savePrediction: please first run computePrediction...')
      sys.exit()
    if (self.odir == None and self.rank == 0):
      print('savePrediction: please use setOutputDir to define the output directory...')

    np.save(self.odir + fname, self.P)
    if not os.path.isfile(self.odir + fname + '.npy'):
      print('savePrediction: saving not succesfull...')