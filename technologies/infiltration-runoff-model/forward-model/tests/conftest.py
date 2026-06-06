"""Pytest configuration for the forward-model suite.

Force single-threaded BLAS / MUMPS for the tests. The verification meshes are tiny, so the direct
solves are trivial; with default threading, BLAS/MUMPS spawn threads that BUSY-WAIT on those tiny
matrices -- pure overhead that made the stiff multi-step overland-coupling tests ~9x slower (e.g.
7.4 min vs 51 s for one lateral-routing test, with all cores pegged spinning). Real parallelism for
production-scale problems comes from MPI, not BLAS threads on sub-meter verification meshes.

Set here (and BEFORE numpy/PETSc/DOLFINx import) so the suite is fast without needing the caller to
export the vars; ``setdefault`` lets an explicit override stand.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
