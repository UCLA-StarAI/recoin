
import argparse
from experiments.runners import run_pa, run_rcwmi, run_rej, run_xsdd, run_trigt
from numpy.random import RandomState
from os import listdir, mkdir
from os.path import isfile, join
from pywmi import Density


# EXECUTION DEFAULTS
DEF_SEED = 666

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--root-dir", help="Directory containing the densities",
                    type=str, required=True)
parser.add_argument("--alg", help="Algorithm",
                    type=str, required=True)
parser.add_argument("--n-bins", help="Number of bins for the marginals",
                    type=int, required=True)

# optional
parser.add_argument("-s", "--seed", help="Seed number",
                    type=int, default=DEF_SEED)

args = parser.parse_args()

rand_gen = RandomState(DEF_SEED)

densities = sorted([f for f in listdir(args.root_dir)
                    if isfile(join(args.root_dir, f)) and f.endswith(".density")],
                   key=lambda x: int(x.replace(".density","").split('-')[-1]))

for density_path in densities:

    density = Density.from_file(join(args.root_dir, density_path))

    name = density_path.replace(".density", "")
    log_path = join(args.root_dir, f"{name}-{args.alg}.results")

    if args.alg == 'pa':
        run_pa(density, args.n_bins, log_path)

    elif args.alg.startswith('rej'):
        NSAMPLES = int(args.alg.split("-")[1])
        run_rej(density, args.n_bins, NSAMPLES, log_path)

    elif args.alg.startswith('xsdd'):
        NSAMPLES = int(args.alg.split("-")[1])
        run_xsdd(density, args.n_bins, NSAMPLES, log_path)

    elif args.alg.startswith('rcwmi'):

        rcwmi_params = args.alg.strip().split("-")[1:]

        if len(rcwmi_params) < 2:
            print("RCWMI: unspecified parameters")
            print("rcwmi-complits-maxiters[-n_processes]")

        n_comp_lits = int(rcwmi_params[0])
        max_iters = int(rcwmi_params[1])
        nproc = None

        if len(rcwmi_params) == 3:
            nproc = int(rcwmi_params[2])

        run_rcwmi(density, args.n_bins, n_comp_lits, max_iters,
                  rand_gen, log_path, nproc=nproc)

    elif args.alg.startswith('trigt'):
        run_trigt(args.n_bins, args.root_dir, name, log_path)

    else:
        print(f"Algorithm '{args.alg}' not recognized")
