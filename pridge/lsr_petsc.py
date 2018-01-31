from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import time as time
from pridge.cg_petsc import CGpetsc
import sys 
import os.path

class LSRpetsc(object):
  def __init__(self):
    self.comm = PETSc.COMM_WORLD
    self.size = PETSc.COMM_WORLD.size
    self.rank = PETSc.COMM_WORLD.rank
    self.odir = None # output directory
    self.A_train =  None # the operator - train rows
    self.A_val =  None # the operator - val rows
    self.A_test =  None # the operator - test rows
    
    self.B_train = None # the r.h.s. - train rows
    self.B_val = None # the r.h.s. - val rows
    self.B_test = None # the r.h.s. - test rows
    self.X = None # the solution
    self.P = None # the prediction
    self.reg = 0.1 # initial regularization
    self.verbosity = 0

  def initialize(self):
    if((self.A_train == None or self.B_train == None) and self.rank == 0):
      print('initialize: please provide the operator and  r.h.s. first...')
      sys.exit()
       
    self.X = PETSc.Mat()
    self.P = PETSc.Mat()
           
    self.X.createDense([self.numOfColsA_train, self.numOfColsB_train])
    self.P.createDense([self.numOfRowsA_train, self.numOfColsB_train])
    
    self.X.setUp()
    self.P.setUp()
    
    self.x_0_train, self.b_0_train = self.A_train.createVecs()
    self.x_0_train_old = self.x_0_train.duplicate()
    self.x_0_val, self.b_0_val = self.A_val.createVecs()
    self.x_0_test, self.b_0_test = self.A_test.createVecs()
  
    # MPI auxiliary arrays for gathering of results
    self.r_counts = np.array([0]*PETSc.COMM_WORLD.size, dtype = int)
    self.r_displs = np.array([0]*PETSc.COMM_WORLD.size, dtype = int)
     
  def setVerbosity(self, ver = 1):
    self.verbosity = ver
    
  def setReg(self, reg):
    self.reg = reg
    
  def setOutputDir(self, odir = ''):
    if (not os.path.isdir(odir) and self.rank == 0):
      os.makedirs(odir)
      time.sleep(0.1)  
    if (not os.path.isdir(odir) and self.rank == 0):
      print('setDataDir(): directory does not exist and can not be created...')  
      sys.exit()
    self.odir = odir
      
  def setOperator(self, A_train, A_val, A_test):
    if(type(A_train) is not PETSc.Mat):
      print('Operator has to be a matrix of petsc4py.PETSc.Mat type...')
      sys.exit()
    self.A_train = A_train
    self.A_val = A_val
    self.A_test = A_test
    self.numOfRowsA_train, self.numOfColsA_train = self.A_train.getSize()
    self.numOfRowsA_val, self.numOfColsA_val = self.A_val.getSize()
    self.numOfRowsA_test, self.numOfColsA_test = self.A_test.getSize()
    
  def setRhs(self, B_train, B_val, B_test):
    if not hasattr(self, 'A_train'):
      print('Please use setOperator before setRhs...')
      sys.exit()
    if(type(B_train) is not PETSc.Mat):
      # we are interested only in the case of multiple r.h.s.
      # otherwise simple numpy is more then enough ... 
      print('Right hand side has to be a matrix of petsc4py.PETSc.Mat type...')
      sys.exit()
    self.B_train = B_train
    self.B_val = B_val
    self.B_test = B_test
    self.numOfRowsB_train, self.numOfColsB_train = self.B_train.getSize()
    self.numOfRowsB_val, self.numOfColsB_val = self.B_val.getSize()
    self.numOfRowsB_test, self.numOfColsB_test = self.B_test.getSize()
    #TODO: check the consistency of TVT setup!!! like  self.numOfColsB_val == self.numOfColsB_test == self.numOfColsB_train etc
    
    # check the consistency of the operator A with B
    if(self.numOfRowsA_train != self.numOfRowsB_train):
      print('Operator and the r.h.s. have to have the same row dimension...')
      sys.exit()
      
  def res(self, A, x, b):
    r = b.duplicate()
    A.mult(x, r) # prediction
    r.aypx(-1., b) # residum
    res = (r.dot(r))**0.5
    norm = (b.dot(b))**0.5
    return res/norm # relative residum
      
  def train(self):
    self.initialize()
    r_start, r_end = self.X.getOwnershipRange()
    rows = range(r_start, r_end)
    if (self.verbosity > 0 and self.rank == 0):
      start = time.time()
    if (self.rank == 0):
      self.regs = np.zeros(self.numOfColsB_train)
      self.trainErrors = np.zeros(self.numOfColsB_train)
      self.valErrors = np.zeros(self.numOfColsB_train)
      self.testErrors = np.zeros(self.numOfColsB_train)
      self.niters = np.zeros(self.numOfColsB_train)
      self.rnorms = np.zeros(self.numOfColsB_train)
    
    for cB in range(self.numOfColsB_train): # runs through the columns of the r.h.s. B  
      self.B_train.getColumnVector(cB, self.b_0_train)
      self.B_val.getColumnVector(cB, self.b_0_val)
      self.B_test.getColumnVector(cB, self.b_0_test)
      reg = self.reg;
      self.x_0_train.set(0.)
      valError = self.res(self.A_val, self.x_0_train, self.b_0_val) # with solution = 0
      valError_old =  2.*valError
      iters = None
      rnorm = None
      trainError = self.res(self.A_train, self.x_0_train, self.b_0_train)
      while valError < valError_old:
        if (self.verbosity > 1 and self.rank == 0):
          print('reg = %g' % (reg))
        norm_b_0_train = (self.b_0_train.dot(self.b_0_train))**0.5
        if (self.verbosity > 1 and self.rank == 0):
          print('norm_b_0_train = %g' % (norm_b_0_train))
        self.x_0_train_old = self.x_0_train
        iters_old = iters
        rnorm_old = rnorm
        iters, rnorm = CGpetsc(self.A_train, self.b_0_train, self.x_0_train, reg*norm_b_0_train**2)
        trainError_old = trainError
        trainError = self.res(self.A_train, self.x_0_train, self.b_0_train)
        valError_old = valError
        valError = self.res(self.A_val, self.x_0_train, self.b_0_val)
        if (self.verbosity > 1 and self.rank == 0):
          print('iters = %g,  error norm = %g' % (iters, rnorm))
          print('valError = %g' % (valError))
          print('trainError = %g' % (trainError))
        reg = reg/2.
          #PETSc.Sys.Print('iters = %g,  error norm = %g' % (iters, rnorm), comm = PETSc.COMM_WORLD)
      self.X.setValues(rows, cB, self.x_0_train_old.getArray())
      testError = self.res(self.A_test, self.x_0_train_old, self.b_0_test)
      if(self.rank == 0):
        self.regs[cB] = 4.*reg
        self.trainErrors[cB] = trainError_old
        self.valErrors[cB] = valError_old
        self.testErrors[cB] = testError
        self.niters[cB] = iters_old
        self.rnorms[cB] = rnorm_old
        
      if(self.verbosity > 0 and self.rank == 0):
        print('r.h.s. column = %g' % (cB))
        print('regs[' + str(cB)+']= ' + str(self.regs[cB]))
        print('trainError = %g' % (self.trainErrors[cB]))
        print('valError = %g' % (self.valErrors[cB]))
        print('testError = %g' % (self.testErrors[cB]))
        print('iters = %g' % (self.niters[cB]))
        print('rnorm = %g\n' % (self.rnorms[cB]))
        
    self.X.assemblyBegin()
    self.X.assemblyEnd()
    if (self.verbosity > 0 and self.rank == 0):
      print('took ' + str(time.time() - start))
      
  def saveTrainingInfo(self):
    if(self.rank == 0):
      np.save(self.odir + 'regs', self.regs)
      np.save(self.odir + 'train_errors', self.trainErrors)
      np.save(self.odir + 'val_errors', self.valErrors)
      np.save(self.odir + 'test_errors', self.testErrors)
      np.save(self.odir + 'niters', self.niters)
      np.save(self.odir + 'rnorms', self.rnorms)
      
  def saveEstimatePETSc(self, fname = 'estX'):
    if(self.X == None and self.rank == 0):
      print('saveEstimatePETSc: please compute the estimate first...')
      sys.exit()
    if (self.odir == None and self.rank == 0):
      print('saveEstimatePETSc: please use setOutputDir to define the output directory...')
      sys.exit()
    viewer = PETSc.Viewer().createBinary(self.odir + fname, 'w')
    viewer.pushFormat(viewer.Format.NATIVE)
    viewer(self.X)  
   
  def saveEstimateNumpy(self, fname = 'estX'):
    if(self.X == None and self.rank == 0):
      print('saveEstimateNumpy: please compute the estimate first...')
      sys.exit()
    if (self.odir == None and self.rank == 0):
      print('saveEstimateNumpy: please use setOutputDir to define the output directory...')
      sys.exit()
    # local numpy array (np.ascontiguousarray: Fortran -> C)
    self.Xlocal = np.ascontiguousarray(self.X.getDenseArray()) 
    self.cXlocal = self.Xlocal.shape[0]*self.Xlocal.shape[1]
    ranges = self.X.getOwnershipRanges()
    disp = 0
    for i in range(self.size):
      size = (ranges[i+1] - ranges[i])*self.numOfColsB_train
      self.r_counts[i] = size
      self.r_displs[i] = disp
      disp += size
    
    if(self.rank == 0):
      self.Xglobal = np.zeros((self.numOfColsA_train, self.numOfColsB_train))
    else:
      self.Xglobal = None
    #s_msg = [self.Xlocal, (self.cXlocal, 0), MPI.DOUBLE]
    s_msg = self.Xlocal    
    r_msg = [self.Xglobal, (self.r_counts, self.r_displs), MPI.DOUBLE]
    
    MPI.COMM_WORLD.Barrier()
    MPI.COMM_WORLD.Gatherv(s_msg, r_msg, root = 0)
    MPI.COMM_WORLD.Barrier()
    
    if(self.rank == 0):
      np.save(self.odir + fname, self.Xglobal)
      if not os.path.isfile(self.odir + fname + '.npy'):
        print('saveEstimate: saving not succesfull...')
        
  def computePrediction(self):
    # the prediction of the model P = A*X
    if(self.X == None and self.rank == 0):
      print('computePrediction: please first train the least squares model by running train...')
      sys.exit()
    self.A_train.matMult(self.X, self.P)
    self.P.assemblyBegin()
    self.P.assemblyEnd()
    
  def savePredictionPETSc(self, fname = 'predP'):
    if(self.P == None and self.rank == 0):
      print('savePredictionPETSc: please first run computePrediction...')
      sys.exit()
    if (self.odir == None and self.rank == 0):
      print('savePredictionPETSc: please use setOutputDir to define the output directory...')
    viewer = PETSc.Viewer().createBinary(self.odir + fname, 'w')
    viewer.pushFormat(viewer.Format.NATIVE)
    viewer(self.P)
  
  def savePredictionNumpy(self, fname = 'predP'):
    if(self.P == None and self.rank == 0):
      print('savePredictionNumpy: please first run computePrediction...')
      sys.exit()
    if (self.odir == None and self.rank == 0):
      print('savePredictionNumpy: please use setOutputDir to define the output directory...')
        
    self.Plocal = np.ascontiguousarray(self.P.getDenseArray())
    self.cPlocal = self.Plocal.shape[0]*self.Plocal.shape[1]
    ranges = self.P.getOwnershipRanges()
    disp = 0
    for i in range(self.size):
      size = (ranges[i+1] - ranges[i])*self.numOfColsB_train
      self.r_counts[i] = size
      self.r_displs[i] = disp
      disp += size
    
    if(self.rank == 0):
      self.Pglobal = np.zeros((self.numOfRowsA_train, self.numOfColsB_train))
    else:
      self.Pglobal = None
    s_msg = [self.Plocal, (self.cPlocal, 0), MPI.DOUBLE]
    r_msg = [self.Pglobal, (self.r_counts, self.r_displs), MPI.DOUBLE]
    
    MPI.COMM_WORLD.Barrier()
    MPI.COMM_WORLD.Gatherv(s_msg, r_msg, root = 0)
    MPI.COMM_WORLD.Barrier()
    
    if(self.rank == 0):
      np.save(self.odir + fname, self.Pglobal)
      if not os.path.isfile(self.odir + fname + '.npy'):
        print('savePrediction: saving not succesfull...')
            
if __name__ == '__main__':
  pass