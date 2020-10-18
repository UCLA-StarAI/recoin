
from pysmt.shortcuts import *
from rcwmi import RCWMI
from rcwmi.rcwmiold import RCWMIold

from time import time
from sys import argv


if len(argv) != 2:
    print("USAGE: python3 compare.py NVARS")
    exit()

NVARS = int(argv[1])
VARS = [Symbol(f'x{i}', REAL) for i in range(NVARS)]

LBOUND, UBOUND = 0, 1

STRATEGY = 'everything'
NCOMPLITS = 2
MAXITERS = 5
NPROC = 4

clauses = []
for i in range(NVARS):
    clauses.append(LE(Real(LBOUND), VARS[i]))
    clauses.append(LE(VARS[i], Real(UBOUND)))
for i in range(NVARS - 1):
    for j in range(i, NVARS):
        # create a new clause        
        cl = Or(LE(Times(Real(2), VARS[i]), VARS[j]),
                LE(VARS[i], Times(Real(2), VARS[j])))
        clauses.append(cl)


formula = And(clauses)
weight = Real(1)

oldsolver = RCWMIold(formula, weight,
                     n_comp_lits=NCOMPLITS,
                     max_compensate_iters=MAXITERS,
                     relax_strategy=STRATEGY)

t0 = time()
Z_old_noca, _ = oldsolver.compute_volumes(cache=False)
T_old_noca = time() - t0

t0 = time()
Z_old_sica, _ = oldsolver.compute_volumes(cache=True)
T_old_sica = time() - t0

newsolver = RCWMI(formula, weight,
                  n_comp_lits=NCOMPLITS,
                  max_compensate_iters=MAXITERS,
                  relax_strategy=STRATEGY,
                  n_processes=NPROC)

t0 = time()
Z_new_noca, _ = newsolver.compute_volumes(cache=False)
T_new_noca = time() - t0

t0 = time()
Z_new_sica, _ = newsolver.compute_volumes(cache=True)
T_new_sica = time() - t0

print("solver\tZ\ttime")
print(f"old-noca\t{Z_old_noca}\t{T_old_noca}")
print(f"old-sica\t{Z_old_sica}\t{T_old_sica}")
print(f"new-noca\t{Z_new_noca}\t{T_new_noca}")
print(f"new-sica\t{Z_new_sica}\t{T_new_sica}")
