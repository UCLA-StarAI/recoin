from sympy import Poly, Symbol
import matplotlib.pyplot as plt

from os import mkdir
from os.path import isdir, isfile, join

import json
from pysmt.shortcuts import And, Bool, LE, Real
from rcwmi import RCWMI
try:
    from wmipa import WMI
except ImportError:
    print("couldn't import wmipa")
    WMI = None

try:
    from pywmi import RejectionEngine, XsddEngine
    from pywmi.engines.rejection import RejectionIntegrator
except ImportError:
    print("couldn't import pywmi")
    RejectionEngine, XsddEngine, RejectionIntegrator = None, None, None
    
    
          
import time

def plot_all_marginals(x, xvals, marginals, path=None):
    plt.ylim(-0.01, 1.01)

    for m in marginals:
         print("METHOD", m)
         print(marginals[m])
         print("==================================================")

    for method in marginals:
        plt.step(xvals, marginals[method], '--', where='mid',
                 label=method)

    plt.legend()
    plt.title(f'Marginals for variable {x}')
    plt.xlabel(f'{x}')
    plt.ylabel(f'p({x})')

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.clf()


def run_pa(density, n_bins, log_path):
    formula = density.support
    weight = density.weight
    CACHE = False

    #log_path = join(output_folder, f"{name}-gt-PA.json")
    if not isfile(log_path):
        print("Running WMI-PA")

        wmipa = WMI(density.support, density.weight)
        Z_pa, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA, cache=CACHE)

        gt_marginals = {}
        for xvar in density.domain.get_real_symbols():
            x = xvar.symbol_name()
            gt_marginals[x] = []
            low, up = density.domain.var_domains[x]
            slices = [(i / n_bins) * (up - low) + low for i in range(0, n_bins + 1)]
            for i in range(len(slices) - 1):
                l, u = slices[i], slices[i + 1]
                gt_q = And(LE(Real(l), xvar), LE(xvar, Real(u)))
                gt_vol, _ = wmipa.computeWMI(gt_q, mode=WMI.MODE_PA, cache=CACHE)
                gt_marginals[x].append((l, u, gt_vol / Z_pa))

        with open(log_path, 'w') as log_file:
            json.dump(gt_marginals, log_file, indent=2)

    else:
        print(f"Found {log_path}")
        with open(log_path, 'r') as log_file:
            gt_marginals = json.load(log_file)

    return gt_marginals


def run_rej(density, n_bins, n_samples, log_path):

    #log_path = join(output_folder, f"{name}-rej.json")
    if not isfile(log_path):
        print(log_path)
        rej_log = {}
        # create queries
        qs = {}
        for xvar in density.domain.get_real_symbols():
            x = xvar.symbol_name()
            qs[x] = []
            low, up = density.domain.var_domains[x]
            slices = [(i / n_bins) * (up - low) + low for i in range(0, n_bins + 1)]
            for i in range(len(slices) - 1):
                l, u = slices[i], slices[i + 1]
                qs[x].append(
                    (l, u, And(LE(Real(l), xvar), LE(xvar, Real(u))))
                )

        t0 = time.perf_counter()
        rejection_engine = RejectionEngine(
            density.domain, density.support, density.weight, sample_count=n_samples
        )
        rej_bin_probs = {}
        for x in qs:
            rej_bin_probs[x] = []
            for l, u, q in qs[x]:
                prob_q = rejection_engine.compute_probability(q)
                rej_bin_probs[x].append((l, u, prob_q))
        t1 = time.perf_counter()
        rej_log['time'] = t1 - t0
        rej_log['query_marginals'] = rej_bin_probs

        with open(log_path, 'w') as f:
            json.dump(rej_log, f)

    else:
        print(f"Found {log_path}")
        with open(log_path, 'r') as f:
            rej_bin_probs = json.load(f)['query_marginals']

    return rej_bin_probs


def run_xsdd(density, n_bins, n_samples, log_path):

    #log_path = join(output_folder, f"{name}-xsdd.json")
    print(f"running XSDD(Sampling) with {n_bins} bins and {n_samples} samples")
    print(f"dumping the result in {log_path}")
    if not isfile(log_path):
        xsdd_log = {}
        # create queries
        qs = {}
        for xvar in density.domain.get_real_symbols():
            x = xvar.symbol_name()
            qs[x] = []
            low, up = density.domain.var_domains[x]
            slices = [(i / n_bins) * (up - low) + low for i in range(0, n_bins + 1)]
            for i in range(len(slices) - 1):
                l, u = slices[i], slices[i + 1]
                qs[x].append(
                    (l, u, And(LE(Real(l), xvar), LE(xvar, Real(u))))
                )

        t0 = time.perf_counter()
        backend = RejectionIntegrator(n_samples, 0)
        xsdd_engine = XsddEngine(density.domain, density.support, density.weight,
                                 convex_backend=backend, factorized=True, ordered=False)
        xsdd_bin_probs = {}
        for x in qs:
            xsdd_bin_probs[x] = []
            for l, u, q in qs[x]:
                prob_q = xsdd_engine.compute_probability(q)
                xsdd_bin_probs[x].append((l, u, prob_q))
        t1 = time.perf_counter()
        xsdd_log['time'] = t1 - t0
        xsdd_log['query_marginals'] = xsdd_bin_probs

        with open(log_path, 'w') as f:
            json.dump(xsdd_log, f)

    else:
        print(f"Found {log_path}")
        with open(log_path, 'r') as f:
            xsdd_bin_probs = json.load(f)['query_marginals']

    return xsdd_bin_probs


def run_rcwmi(density, n_bins, complits, maxiters, rand_gen, log_path, cache=True, nproc=None):

    if nproc is None:
        nproc = 1

    #log_path = join(output_folder, "comp_log.json")

    if not isfile(log_path):

        print("Running RCWMI")
        print(f"N. compensating literals: {complits}")
        print(f"Max. iterations: {maxiters}")
        print(f"N. processes: {nproc}")

        # create bin queries
        debug_queries = []

        xvals = {}
        for xvar in density.domain.get_real_symbols():
            x = xvar.symbol_name()
            low, up = density.domain.var_domains[x]
            slices = [(i / n_bins) * (up - low) + low for i in range(0, n_bins + 1)]
            xvals[x] = slices[:-1]
            for i in range(len(slices) - 1):
                l, u = slices[i], slices[i + 1]
                debug_queries.append(LE(Real(l), xvar))
                debug_queries.append(LE(xvar, Real(u)))

        t0 = time.perf_counter()
        rcwmi = RCWMI(density.support, density.weight, n_comp_lits=complits, rand_gen=rand_gen,
                      max_compensate_iters=maxiters, log_path=log_path,
                      debug_queries=debug_queries, n_processes=nproc)
        t1 = time.perf_counter()

        # adding xvals and runtime to the log
        with open(log_path, 'r') as f:
            log = json.load(f)

        log['runtime'] = t1 - t0
        log['xvals'] = xvals

        with open(log_path, 'w') as f:
            json.dump(log, f)
    else:
        print(f"Found {log_path}")


def tri_f1(alpha):
    """
    :param alpha:
    :return: int 10^4*(x+y)*y*z dz dy dx, x = (1-a) to 1, y = 0 to a, z = 0 to y
    """
    return 125 * (alpha**5) * (10 + 3 * alpha)


def tri_f2(alpha):
    """
    :param alpha:
    :return: int 10^4*(x+y)*y*z dz dy dx, x=(1-a) to 1, y=a to (x-a), z=0 to a
    """
    return -625/3 * (alpha**3) \
        * (-20 + 78 * alpha - 92 * (alpha**2) + 49 * alpha**3)


def tri_f3(alpha):
    """
    :param alpha:
    :return: int 100*y*z dz dy dx, x = 0 to a, y = (1-a) to 1, z = (1-a) to y
    """
    return 25/2 * ((-2 + alpha)**2) * alpha**3


def tri_f4(alpha):
    """
    :param alpha:
    :return: int 100*(x+y) dy dx, x = (1-a) to 1, y = 0 to x - a
    """
    return 50 * alpha * (3 - 7 * alpha + 4 * alpha**2)


def tri_f5(alpha):
    """
    :param alpha:
    :return: int 1 dy dx, x = 0 to a, y = (x + a) to 1
    """
    return alpha - (3 * alpha**2) / 2


def tri_f6(alpha):
    """
    :param alpha:
    :return: int 10^4*(x+y)*y*z dz dy dx, x=(1-a) to 1, y=a to 1/2, z=0 to a
    """
    return -625/6 * (alpha**3) * (-8 + 3 * alpha + 24 * alpha**2 + 4 * alpha**3)


def tri_f7(alpha):
    """
    :param alpha:
    :return: int 100*(x+y) dy dx, x = (1-a) to 1, y = 0 to 1/2
    """
    return 25/2 * (5 - 2 * alpha) * alpha


def run_trigt(NBINS, ROOT_DIR, name, log_path):
    assert NBINS == 2, "ground truth only for two bins!!"

    if not isfile(log_path):
        print("Running TRI-GT")
        param_path = join(ROOT_DIR, name + ".alpha")
        with open(param_path, 'r') as f:
            param = json.load(f)
        nvars = param["nvars"]
        alpha = param["alpha"]

        assert nvars >= 3, "number of vars must >= 3!!"

        remain = nvars % 3
        n_tris = int(nvars / 3)

        # for all x, Pr(x > .5) are the same.
        # moreover, Pr(x > .5) = Pr(z < .5) always holds.
        if remain == 0 or remain == 1:
            Z = (tri_f3(alpha)) ** n_tris + \
                (tri_f1(alpha) + tri_f2(alpha)) ** n_tris
            pr_x_less = (tri_f3(alpha)) ** n_tris / Z
            pr_y_less = ((tri_f6(alpha) + tri_f1(alpha)) *
                         (tri_f1(alpha) + tri_f2(alpha))**(n_tris - 1)) / Z

        else:
            Z = tri_f5(alpha) * (tri_f3(alpha))**n_tris + \
                tri_f4(alpha) * (tri_f1(alpha) + tri_f2(alpha))**n_tris
            pr_x_less = tri_f5(alpha) * (tri_f3(alpha))**n_tris / Z
            pr_y_less = ((tri_f6(alpha) + tri_f1(alpha)) * tri_f4(alpha) *
                         (tri_f1(alpha) + tri_f2(alpha))**(n_tris - 1)) / Z

        marginals = {}
        for i in range(n_tris):
            x, y, z = f"x{3*i+1}", f"x{3*i+2}", f"x{3*i+3}"
            marginals[x] = [[0, 0.5, pr_x_less], [0.5, 1, 1 - pr_x_less]]
            marginals[y] = [[0, 0.5, pr_y_less], [0.5, 1, 1 - pr_y_less]]
            marginals[z] = [[0, 0.5, 1 - pr_x_less], [0.5, 1, pr_x_less]]

        if remain == 1 or remain == 2:
            x = f"x{3*n_tris + 1}"
            marginals[x] = [[0, 0.5, pr_x_less], [0.5, 1, 1 - pr_x_less]]
        if remain == 2:
            y = f"x{nvars}"
            pr_y_less = (tri_f7(alpha) *
                         (tri_f1(alpha) + tri_f2(alpha))**n_tris) / Z
            marginals[y] = [[0, 0.5, pr_y_less], [0.5, 1, 1 - pr_y_less]]

        with open(log_path, 'w') as log_file:
            json.dump(marginals, log_file, indent=2)

    else:
        print(f"Found {log_path}")
        with open(log_path, 'r') as log_file:
            marginals = json.load(log_file)

    return marginals


def plot_marginals(output_folder):

    raise NotImplementedError()

    results = [f for f in listdir(output_folder)
               if isfile(join(output_folder, f)) and f.endswith(".results")]

    named_results = {}
    for res in results:
        name = res.split('-')[0]
        if name not in named_results :
            named_results[name] = []

        named_results[name].append(join(output_folder, res))


    for name in named_results:
        marginals = {}

        for log_path in named_results[name]:

            method = log_path.split('-')[-1].replace(".results", "")
            if 'rcwmi' in method:

                # read RCWMI results
                with open(log_path, 'r') as f:
                    log = json.load(f)
                    #mars = log['real_marginals']
                    epsilons = log['epsilon']
                    bin_probs = log['query_marginals']
                    xvals = log['xvals']

            elif 'pa' in method:
                pass

    for xvar in density.domain.get_real_symbols():
        x = xvar.symbol_name()

        
        marginals_x = {'RC-WMI' : [bin_probs[-1][x][2*i] + bin_probs[-1][x][2*i+1] - 1
                                 for i in range(n_bins)]}

        for method in other_marginals:
            marginals_x[method] = [p for _,_,p in other_marginals[method][x]]

        """ we're not storing this anymore cause NeurIPS is coming AWAWAWAWA
        for it in range(len(epsilons)):
            plot_path = join(output_folder, f"marginals_{x}_{it}.png")
            # approx_marginals = mars[it][x]
            # plot_marginals(x, it, approx_marginals, gt_marginals, plot_path)
            plot_marginals_at_iter(x, it, bin_probs, other_marginals['PA'][x], plot_path)
        """

        plot_path = join(output_folder, f"ALL_marginals_{x}.png")
        plot_all_marginals(x, xvals[x], marginals_x, path=plot_path)

    epsilon_path = join(output_folder, "epsilons.png")
    plt.plot(epsilons)
    plt.xlabel('iteration')
    plt.ylabel('epsilon')        
    plt.savefig(epsilon_path)
    plt.clf()
