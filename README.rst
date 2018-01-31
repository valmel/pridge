PRIDGE: Ridge regression using PETSc with automatic level of regularization
===========================================================================

The package implements ridge regression (regularized least squares) with
an automatic detection of regularization hyperparameter. The focus lies
on problems with sparse operator, potentially a huge one, where the right hand
side is a matrix as well. Thus multiple regressions are to be solved.
The P- in the name of the package stands for parallel.

Two versions are supplied: PETSc- and Numpy- one. The employment of PETSc_ allows
for compact and readable high-level code with well-known excellent out-of-the-box
scalability to potentially thousands of processors (MPI). The single point of weakness
is loading of data, since PETSc is not made with machine learning (ML) in mind
(think rather PDEs). The PETSc matrices are usually locally assembled and not loaded.
This issue is however hardly avoidable in ML and usually not such a big problem
in practice.

Then, for smaller problems is supplied a Numpy_ version of the algorithm. Many
operations in Numpy are multithreaded (OpenMP).

Installation
------------

In the main folder run:

```python setup.py install --user```

For a working PETSc version of the ridge regression, it is necessary that both PETSc 
and petsc4py are properly installed. Moreover, for a full functionality, a recent mpi4py 
has to be installed from the source_ in the classical ```git clone ...``` 
+ ```python setup.py install --user``` fashion.

Examples
--------

Examples are located in ``examples`` directory. See the corresponding
`examples/README.rst`_ file there

.. _PETSc: https://www.mcs.anl.gov/petsc/
.. _Numpy: http://www.numpy.org/
.. _source: https://bitbucket.org/mpi4py/mpi4py/
.. _`examples/README.rst` : examples/README.rst