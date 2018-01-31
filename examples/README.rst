Examples
========

1. ```python runNumpyLSR.py``` allows you to run Numpy-version of the linear regression
code with randomly generated synthetic data
 
2. ``mpirun -np n python runPETScLSR.py`` allows you to run PETSc-version of the linear regression
code with randomly generated synthetic data. Here ``n`` is number of independent mpi processes. 
Please use `../data/generatePETScData.py`_ python script before running this.


.. _`../data/generatePETScData.py`: ../data/generatePETScData.py