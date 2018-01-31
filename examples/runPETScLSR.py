def run():
  from petsc4py import PETSc
  from pridge import LSRpetsc 
  
  #OptDB = PETSc.Options()
  #N     = OptDB.getInt('N', 100)
  #draw  = OptDB.getBool('draw', False)

  ddir = '../data/'  
  
  # load the "sparse" operator
  viewer = PETSc.Viewer().createBinary(ddir +'A_train.dat', 'r')
  A_train = PETSc.Mat(PETSc.COMM_WORLD).load(viewer)
  viewer = PETSc.Viewer().createBinary(ddir +'A_val.dat', 'r')
  A_val = PETSc.Mat(PETSc.COMM_WORLD).load(viewer)
  viewer = PETSc.Viewer().createBinary(ddir +'A_test.dat', 'r')
  A_test = PETSc.Mat(PETSc.COMM_WORLD).load(viewer)
  
  # load the r.h.s.
  viewer = PETSc.Viewer().createBinary(ddir +'B_train.dat', 'r')
  B_train = PETSc.Mat(PETSc.COMM_WORLD).load(viewer)
  viewer = PETSc.Viewer().createBinary(ddir +'B_val.dat', 'r')
  B_val = PETSc.Mat(PETSc.COMM_WORLD).load(viewer)
  viewer = PETSc.Viewer().createBinary(ddir +'B_test.dat', 'r')
  B_test = PETSc.Mat(PETSc.COMM_WORLD).load(viewer)
  
  # set up the least square model 
  ls = LSRpetsc()
  ls.setOperator(A_train, A_val, A_test)
  ls.setRhs(B_train, B_val, B_test)
  ls.setReg(0.01)
  ls.setOutputDir(ddir)
  
  # train the least square model
  ls.train()
  ls.saveTrainingInfo()
  
  # save the results
  ls.saveEstimateNumpy('X')
  ls.computePrediction()
  ls.savePredictionNumpy('P')

if __name__ == '__main__':
  import sys, petsc4py
  petsc4py.init(sys.argv)
  run()