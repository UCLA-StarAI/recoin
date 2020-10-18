
import argparse
from experiments.generate import generate_problem
import json
from numpy.random import RandomState
from os import mkdir
from os.path import isdir, join
from sys import argv
from time import time

# GENERATION DEFAULTS
DEF_SEED = 666
DEF_REP = 5
DEF_GAP = 0.3

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output-path", help="Output path of the density",
                    type=str, required=True)
parser.add_argument("-g", "--shape", help="Shape of the dependecy graph",
                    type=str, required=True)
parser.add_argument("--min-vars", help="Min # of variables",
                    type=int, required=True)
parser.add_argument("--max-vars", help="Max # of variables",
                    type=int, required=True)

# optional
parser.add_argument("--gap", help="Gap?",
                    type=float, default=DEF_GAP)
parser.add_argument("--rep", help="How many instances for each config",
                    type=int, default=DEF_REP)
parser.add_argument("-s", "--seed", help="Seed number",
                    type=int, default=DEF_SEED)

args = parser.parse_args()
if not isdir(args.output_path):
    print(f"Not a directory: {args.output_path}")
    exit(1)

print("Generating suite w/ params:")
print(f"seed:\t\t{args.seed}")
print(f"output_path:\t{args.output_path}")
print(f"shape:\t\t{args.shape}")
print(f"min/max vars:\t[{args.min_vars},{args.max_vars}]")
print(f"gap:\t\t{args.gap}")



rand_gen = RandomState(args.seed)

for n in range(args.rep):

    exp_dir = f'{args.shape}-gap-{args.gap}-inst-{n}-from-{args.min_vars}-to-{args.max_vars}'
    root_dir = join(args.output_path, exp_dir)
    
    if not isdir(root_dir):
        mkdir(root_dir)
        
    for nvars in range(args.min_vars, args.max_vars + 1):
        t0 = time()
        name = f'{args.shape}-gap-{args.gap}-inst-{n}-vars-{nvars}'
        
        density = generate_problem(nvars, rand_gen, args.shape, gap=args.gap)
        
        density_path = join(root_dir, name + '.density')
        density.to_file(density_path)
        
        gentime = time() - t0
        print(f"Generated {density_path} in {gentime} s.")
