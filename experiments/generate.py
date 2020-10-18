
from pysmt.shortcuts import And, Ite, LE, Or, Plus, REAL, Real, Symbol, Times
from pywmi import Density, Domain
import networkx as nx


def triangle(nvars, rand_gen):
    # alpha = rand_gen.uniform(0.05, 0.25)
    alpha = rand_gen.uniform(0.2, 0.25)
    remain = nvars % 3
    n_tris = int(nvars / 3)

    variables = [Symbol(f"x{i}", REAL) for i in range(1, nvars + 1)]

    lbounds = [LE(Real(0), x) for x in variables]
    ubounds = [LE(x, Real(1)) for x in variables]

    clauses = []
    potentials = []

    for i in range(n_tris):
        x, y, z = variables[3*i], variables[3*i+1], variables[3*i+2]
        xc = None if 3*i+3 >= nvars else variables[3*i+3]
        # x_i
        clauses.append(
            Or(LE(x, Real(alpha)), LE(Real(1 - alpha), x))
        )
        # x_i -- y_i
        clauses.append(
            Or(LE(y, Plus(x, Real(-alpha))), LE(Plus(x, Real(alpha)), y))
        )
        # x_i -- z_i
        clauses.append(
            Or(LE(Real(1 - alpha), x), LE(Real(1 - alpha), z))
        )
        clauses.append(
            Or(LE(x, Real(alpha)), LE(z, Real(alpha)))
        )
        # z_i -- y_i
        clauses.append(LE(z, y))
        # x_i -- x_i+1
        if xc:
            clauses.append(Or(LE(x, Real(alpha)), LE(Real(1 - alpha), xc)))
            clauses.append(Or(LE(Real(1 - alpha), x), LE(xc, Real(alpha))))

        pot_yz = Ite(LE(z, y), Times([z, y, Real(100)]), Real(1))
        pot_xy = Ite(LE(y, Plus(x, Real(-alpha))),
                     Times(Real(100), Plus(x, y)), Real(1))
        potentials.append(pot_xy)
        potentials.append(pot_yz)

    if remain == 1:
        x = variables[3*n_tris]
        clauses.append(
            Or(LE(x, Real(alpha)), LE(Real(1 - alpha), x))
        )
    if remain == 2:
        x, y = variables[3*n_tris], variables[nvars-1]
        # x_n
        clauses.append(
            Or(LE(x, Real(alpha)), LE(Real(1 - alpha), x))
        )
        # x -- y
        clauses.append(
            Or(LE(y, Plus(x, Real(-alpha))), LE(Plus(x, Real(alpha)), y))
        )
        potentials.append(
            Ite(LE(y, Plus(x, Real(-alpha))),
                Times(Real(100), Plus(x, y)), Real(1))
        )

    domain = Domain.make([],  # no booleans
                         [x.symbol_name() for x in variables],
                         [(0, 1) for _ in range(len(variables))])
    support = And(lbounds + ubounds + clauses)
    weight = Times(potentials) if len(potentials) > 1 else potentials[0]

    return Density(domain, support, weight, []), alpha


def generate_problem(nvars, rand_gen, graph, gap=0.2):

    assert(nvars > 1)

    variables = [Symbol(f"x{i}", REAL) for i in range(1,nvars+1)]

    lbounds = [LE(Real(0), x) for x in variables]
    ubounds = [LE(x, Real(1)) for x in variables]

    all_deps = [(variables[i], variables[j])
                for i in range(len(variables)-1)
                for j in range(i+1, len(variables))]

    if graph == "TRI":
        return triangle(nvars, rand_gen)

    elif graph.startswith("RANDOM-"):
        ratio = float(graph.partition('-')[-1])
        assert(0.0 <= ratio and ratio <= 1.0)
        ndeps = max(1, int(ratio*len(all_deps))) # at least 1 density
        dependencies = [all_deps[i] for i in
                        rand_gen.choice(list(range(len(all_deps))),
                                        ndeps, replace=False)]
    elif graph == "WATTS":
        G = nx.watts_strogatz_graph(n=nvars, k=2, p=0.5)
        edges = list(G.edges)
        dependencies = [(variables[i], variables[j]) for i, j in edges]

    elif graph == "LADDER":
        G = nx.generators.classic.ladder_graph(int(nvars / 2))
        edges = list(G.edges)
        dependencies = [(variables[i], variables[j]) for i, j in edges]
        if nvars % 2 == 1:
            dependencies.append((variables[nvars-2], variables[nvars-1]))

    else:
        raise NotImplementedError("graph shape not supported")

    clauses = []
    potentials = []
    alpha = rand_gen.random()
    for x1, x2 in dependencies:

        # cl = Or(LE(Plus(x1, x2), Real(2*rand_gen.random())),
        #         LE(Real(2*rand_gen.random()), Plus(x1, x2)))

        # pot = Ite(LE(Plus(x1, x2), Real(2 * rand_gen.random())),
        #           Real(rand_gen.randint(1, 10)), Real(1))

        cl = Or(LE(Plus(x1, x2), Real(1 + alpha)),
                LE(Real(1 - alpha), Plus(x1, x2)))

        pot = Ite(LE(Plus(x1, x2), Real(1 + alpha)),
                  Real(rand_gen.randint(1, 10)), Real(1))

        clauses.append(cl)
        potentials.append(pot)

    for x in variables:
        p = rand_gen.random()
        if p > 0.5:  # gap half of the variables
            continue
        cl = Or(LE(x, Real(0.5 - gap)), LE(Real(0.5 + gap), x))
        clauses.append(cl)

    domain = Domain.make([], # no booleans
                         [x.symbol_name() for x in variables],
                         [(0, 1) for _ in range(len(variables))])
    support = And(lbounds + ubounds + clauses)
    weight = Times(potentials) if len(potentials) > 1 else potentials[0]

    return Density(domain, support, weight, [])



if __name__ == '__main__':
    import argparse
    from numpy.random import RandomState
    from mpwmi.utils import RAND_SEED


    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", help="Output path of the density",
                        type=str, required=True)
    parser.add_argument("-v", "--n-vars", help="Number of variables",
                        type=int, required=True)
    parser.add_argument("-g", "--shape", help="Shape of the dependecy graph",
                        type=str, required=True)
    parser.add_argument("-s", "--seed", help="Seed number",
                        type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        seed = RAND_SEED

    rand_gen = RandomState(seed)

    density = generate_problem(args.n_vars, rand_gen, args.shape)
    density.to_file(args.output_path)  # Save to file
