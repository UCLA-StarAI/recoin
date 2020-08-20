
import networkx as nx
from pysmt.shortcuts import Symbol, LE, Real, And, Times, substitute, Ite, Plus, \
    Bool, is_sat, Equals
from pysmt.typing import REAL
from wmipa import WMI

from mpwmi import MP2WMI
from mpwmi.utils import SMT_SOLVER, safeexp, safelog
from mpwmi import logger

import time
import numpy as np
import json



class RCWMI(MP2WMI):
    """
    A class that implements the Relax-Compensate Model Integration approximate 
    inference algorithm.

    Attributes
    ----------
    TODO

    Methods
    -------
    compute_volumes(queries=None, evidence=None, cache=False)
        Computes the partition function value and the unnormalized probabilities
        of uni/bivariate literals in 'queries'.

    """

    def __init__(self, formula, weight, n_comp_lits=1, smt_solver=SMT_SOLVER,
                 rand_gen=None, tolerance=1.49e-8, relax_strategy='tree',
                 max_compensate_iters=10, comp_cache=True, log_path=None,
                 debug_queries=None, n_processes=1):
        """
        Parameters
        ----------
        formula : pysmt.FNode instance
            The input formula representing the support of the distribution
        weight : pysmt.FNode instance
            Polynomial potentials attached to literal values
        rand_gen : np.random.RandomState instance, optional
            The random number generator (default: RandomState(mpwmi.RAND_SEED))
        """
        super().__init__(formula, weight,
                         smt_solver=smt_solver,
                         rand_gen=rand_gen,
                         tolerance=tolerance,
                         n_processes=n_processes)

        self.relax_strategy = relax_strategy
        self.max_compensate_iters = max_compensate_iters
        self.recover_iters = 0
        self.relaxations = dict()
        self.relax()
        self.compensate(n_comp_lits, cache=comp_cache, log_path=log_path, debug_queries=debug_queries)


    def compute_volumes(self, queries=None, evidence=None, cache=False):
        """Computes the unnormalized probabilities of univariate and
        bivariate literals in 'queries' associated to univariate
        literals and a list of uni/bivariate clauses representing the
        'evidence'.

        Returns (Z given evidence, list[volumes of queries given evidence]).

        Raises NotImplementedError if the literals are not uni/bivariate.

        Parameters
        ----------
        queries : list of pysmt.FNode instances (optional)
            Uni/bivariate literals
        evidence : iterable of pysmt.FNode instances (optional)
            Uni/bivariate clauses, default: None
        cache : bool (optional)
            If True, integrals are cached, default: False
        """
        if queries is None:
            queries = []
        if evidence is None:
            evidence = []
        
        queries2 = []
        n_copies = []
        for q in queries:
            qvars = list(q.get_free_variables())
            if len(qvars) == 2:  # bivariate query
                qvars.sort(key=lambda v : v.symbol_name())
                x, y = qvars[0].symbol_name(), qvars[1].symbol_name()
                if (x,y) in self.relaxations:
                    xcopy = self.relaxations[(x,y)]['copy_name']
                    q = q.substitute({qvars[0] : self.primal.nodes()[xcopy]['var']})
                queries2.append(q)
                n_copies.append(1)
            elif len(qvars) == 1:  # univariate query
                x = qvars[0].symbol_name()
                if x in self.relaxations:
                    copies = self.relaxations[x]['copies']
                    qs = [q] + [q.substitute(
                        {qvars[0]: self.primal.nodes()[xcopy]['var']}
                    ) for xcopy in copies]
                    queries2.extend(qs)
                    n_copies.append(len(copies) + 1)
                else:
                    queries2.append(q)
                    n_copies.append(1)

        evidence2 = []
        for e in evidence:
            evars = list(e.get_free_variables())
            if len(evars) == 2:
                evars.sort(key=lambda v : v.symbol_name())
                x, y = evars[0].symbol_name(), evars[1].symbol_name()
                if (x,y) in self.relaxations:
                    xcopy = self.relaxations[(x,y)]['copy_name']
                    e = e.substitute({evars[0] : self.primal.nodes()[xcopy]['var']})

            evidence2.append(e)
        
        Z, vols = super().compute_volumes(queries2, evidence2, cache=cache)
        mean_vols = []
        id = 0
        for i, v in enumerate(vols):
            mean_vols.append(sum(vols[id: id + n_copies[i]]) / n_copies[i])
            id = id + n_copies[i]
        print(f"len of queries {len(queries)} and len of mean vols {len(mean_vols)}")
        return Z, mean_vols


    def relax(self):

        original = list(self.primal.edges())

        # TODO: add more strategies here
        if self.relax_strategy == 'everything':
            to_remove = original

        elif self.relax_strategy == 'tree':
            # TODO: weight edges according to # potentials
            keep = nx.minimum_spanning_tree(self.primal.G).edges()
            to_remove = [e for e in original if e not  in keep]

        else:
            msg = f"Relaxing strategy not implemented {self.relax_strategy}"
            raise NotImplementedError(msg)

        logger.debug(f"strategy: {self.relax_strategy}")        
        logger.debug(f"relaxing: {len(to_remove)}/{len(original)}")
        logger.debug(f"edges: {to_remove}")
        for edge in to_remove:
            self.relax_edge(edge)

    def relax_edge(self, edge):
        
        u, v = min(edge), max(edge) # lexicographic order
        ucopy = self.copy_name(u, v)
        
        assert(edge in self.primal.edges()), f"{edge} not in G"
        assert((u, v) not in self.relaxations), f"{edge} already relaxed"

        # create a new pysmt var
        old_var = self.primal.nodes()[u]['var']
        new_var = Symbol(ucopy, REAL)

        # copy univariate clauses and potentials for u_c
        new_uniclauses = {substitute(c, {old_var: new_var})
                          for c in self.primal.nodes()[u]['clauses']}
        new_unipotentials = []
        # are these lit in potentials all univariate?
        for lit, w in self.primal.nodes()[u]['potentials']:
            new_unipotentials.append((substitute(lit, {old_var: new_var}),
                                      w.subs(u, ucopy)))

        # add the node u_c to the primal graph
        self.primal.G.add_node(ucopy,
                               var=new_var,
                               clauses=new_uniclauses,
                               potentials=new_unipotentials)

        # copy bivariate clauses and potentials for (u_c, v)
        new_biclauses = {substitute(c, {old_var: new_var})
                         for c in self.primal.edges()[(u, v)]['clauses']}
        new_bipotentials = []
        for lit, w in self.primal.edges()[(u, v)]['potentials']:
            new_bipotentials.append((substitute(lit, {old_var: new_var}),
                                     w.subs(u, ucopy)))

        self.primal.G.add_edge(ucopy, v,
                               clauses=new_biclauses,
                               potentials=new_bipotentials)

        # store the relaxation (s.t. we can invert it)
        self.relaxations[(u, v)] = {'copy_name' : ucopy,
                                   'old_edge' : dict(self.primal.edges()[(u, v)]),
                                   'new_edge' : dict(self.primal.edges()[(ucopy, v)])}

        if (u,) not in self.relaxations:
            self.relaxations[(u,)] = {'copies' : [ucopy]}
        else:
            self.relaxations[(u,)]['copies'].append(ucopy)

        # finally remove the (u, v) dependency
        self.primal.G.remove_edge(u, v)

    def sample_comp_literal(self, var):
        while True:
            k = int(self.rand_gen.choice([-1, 1]))
            b = float(self.rand_gen.uniform(0, 1))
            
            f = And(self.primal.get_full_formula(),
                    Equals(Times(Real(k), var), Real(b)))

            if is_sat(f, solver_name=self.smt_solver):
                # we should actually check SAT for both (kx <= b) and (kx > b)
                break

        return LE(Times(Real(k), var), Real(b))


    def compensating_potentials(self, n_comp_lits):
        winit = lambda x: safeexp(self.rand_gen.random())
        comp_pots = {}
        init_w = 1
        for k in self.relaxations:
            if len(k) == 1:
                continue
            x, y = k
            xc = self.relaxations[(x,y)]['copy_name']
            var_x = self.primal.nodes()[x]['var']
            var_xc = self.primal.nodes()[xc]['var']
            
            if x not in comp_pots:
                # REM model for 'x'
                rem = MP2WMI(self.primal.get_univariate_formula(x), Real(1),
                             n_processes=self.n_processes)

                # sampling compensating potentials for 'x'
                # original = [[self.sample_comp_literal(var_x), winit(None)]
                #             for _ in range(n_comp_lits)]
                # try uniform weights
                original = [[self.sample_comp_literal(var_x), init_w]
                            for _ in range(n_comp_lits)]
                print(f"%%%%%%% var {x} compensating literals:")
                for lit, w in original:
                    print(f"%%%%%%% {lit}")
                comp_pots[x] = (rem, original, {})
            else:
                original = comp_pots[x][1]

            # using the same comp lits for the copies of 'x'
            # comp_pots[x][2][xc] = [
            #     [substitute(lit, {var_x: var_xc}), winit(None)]
            #     for lit, _ in original]
            comp_pots[x][2][xc] = [
                [substitute(lit, {var_x: var_xc}), init_w]
                for lit, _ in original]

        return comp_pots

    def compensate(self, n_comp_lits, damping=0.5, tolerance=1e-3, cache=False,
                   log_path=None, debug_queries=None):
        # queries is put here for debugging purpose

        comp_pots = self.compensating_potentials(n_comp_lits)
        original_pots = {}

        sorted_vars = sorted(comp_pots.keys())
        sorted_copies = {}
        comp_lits = []
        for x in sorted_vars:
            original_pots[x] = self.primal.nodes()[x]['potentials']
            _, x_comp_pots, xc_comp_pots = comp_pots[x]
            comp_lits.extend([l for l, _ in x_comp_pots])
            self.primal.nodes()[x]['potentials'] = \
                original_pots[x] + x_comp_pots

            sorted_copies[x] = sorted(xc_comp_pots.keys())
            for xc in sorted_copies[x]:
                original_pots[xc] = self.primal.nodes()[xc]['potentials']
                comp_lits.extend([l for l, _ in xc_comp_pots[xc]])
                self.primal.nodes()[xc]['potentials'] = \
                    original_pots[xc] + xc_comp_pots[xc]

        old_comp_probs, rem_probs = None, None
        it, epsilon = 0, np.inf

        if log_path is not None:
            log = {'epsilon': [], 'real_marginals': [], 'query_marginals': []}

        while it < self.max_compensate_iters and epsilon > tolerance:
            print(f"at iteration {it}, with epsilon = {epsilon}")

            t1 = time.perf_counter()
            Z, vol = super().compute_volumes(queries=comp_lits, cache=cache)
            t2 = time.perf_counter()
            print(f"inference on relaxed formula takes {t2 - t1} seconds")

            comp_probs = [v/Z for v in vol]
            if old_comp_probs and rem_probs:
                id = 0
                for x in sorted_vars:
                    print(f"***** Probs for var {x} and its copies*****")
                    for i in range(1 + len(sorted_copies[x])):
                        print(f"\tprobabilities for {x}_{i}: "
                              f"{old_comp_probs[id : id + n_comp_lits]}")
                        id += n_comp_lits
                    print(f"\tremain probs for var {x}: {rem_probs[x]}")

                diff = [abs(p - q) for p, q in zip(old_comp_probs, comp_probs)]
                epsilon = max(diff)  # or mean
                print(f"epsilon = {epsilon}")
            old_comp_probs = comp_probs

            # update potentials in remaining formula
            ratios = dict()
            rem_probs = dict()
            for x in sorted_vars:

                remx, x_comp_pots, xc_comp_pots = comp_pots[x]
                rem_comp_pots = []
                rem_queries = []
                
                for k, xpot in enumerate(x_comp_pots):
                    lit_x, w_x = xpot
                    for xc in xc_comp_pots:
                        _, w_xc = xc_comp_pots[xc][k]
                        w_x *= w_xc

                    rem_comp_pots.append((lit_x, w_x))
                    print(f"\t\t\tIn remaining formula, the {k}-th literal:"
                          f"{lit_x} with potentials: {w_x}")
                    rem_queries.append(lit_x)

                x_ratios = []
                for k in range(len(x_comp_pots)):
                    lit_rem_comp_pots = [xpot for i, xpot in enumerate(rem_comp_pots) if i != k]
                    remx.primal.nodes()[x]['potentials'] = original_pots[x] + lit_rem_comp_pots
                    Z_remx, vol_remx = remx.compute_volumes(queries=[rem_queries[k]], cache=cache)

                    print(f"\t\t========== debug float division zero =========")
                    print(f"\t\t Z_remx = {Z_remx: .6f}, vol_remx[0] = {vol_remx[0]: .6f}")

                    x_ratios.append((Z_remx, vol_remx[0]))
                ratios[x] = x_ratios

                remx.primal.nodes()[x]['potentials'] = original_pots[x] + rem_comp_pots
                Z_remx, vol_remx = remx.compute_volumes(queries=rem_queries, cache=cache)
                rem_probs[x] = [vol_remx[k] / Z_remx for k in range(len(x_comp_pots))]

            t3 = time.perf_counter()
            print(f"inference on remaining formula takes {t3 - t2} seconds")

            # update potential parameters
            offset = 0
            for x in sorted_vars:
                Nx = 1 + len(sorted_copies[x])
                _, x_comp_pots, xc_comp_pots = comp_pots[x]
                vol_x = vol[offset : offset + n_comp_lits * Nx]
                offset += n_comp_lits * Nx

                # new update
                for k in range(n_comp_lits):
                    vol_xk = [vol_x[n_comp_lits * i + k] for i in range(Nx)]
                    old_pot_xk = [x_comp_pots[k][1]]
                    old_pot_xk.extend(
                        [xc_comp_pots[xc][k][1] for xc in sorted_copies[x]]
                    )

                    new_pots_xk = []
                    for j in range(Nx):
                        vol_diff = vol_xk[(j + 1) % Nx]
                        pot_diff = [old_pot_xk[jd] for jd in range(Nx)
                                    if jd != j]
                        prod_pots = np.prod(pot_diff)
                        Z_rem, vol_rem = ratios[x][k][0], ratios[x][k][1]

                        new_pot_xkj = (Z_rem - vol_rem) * vol_diff \
                                      / (vol_rem * (Z - vol_diff) * prod_pots)

                        # clip
                        high_clip = 50
                        low_clip = 1e-6 ** (1 / (n_comp_lits * Nx)) + 0.05
                        new_pot_xkj = max(
                            min(new_pot_xkj, old_pot_xk[j] * high_clip),
                            low_clip,
                            old_pot_xk[j] / high_clip
                        )

                        new_pots_xk.append(
                            (new_pot_xkj ** damping)
                            * (old_pot_xk[j] ** (1 - damping))
                        )

                    # update potentials x,k
                    print(
                        f"updating theta_{x}^({k},0) : {comp_pots[x][1][k][1]} ==> {new_pots_xk[0]}")
                    comp_pots[x][1][k][1] = new_pots_xk[0]
                    for j, xc in enumerate(sorted_copies[x]):
                        print(
                            f"updating theta_{x}^({k},{j + 1}) : {comp_pots[x][2][xc][k][1]} ==> {new_pots_xk[j + 1]}")
                        comp_pots[x][2][xc][k][1] = new_pots_xk[j + 1]

            # update the relaxed model with the new potentials
            for x in comp_pots:
                _, x_comp_pots, xc_comp_pots = comp_pots[x]
                self.primal.nodes()[x]['potentials'] = original_pots[x] + x_comp_pots
                for xc in xc_comp_pots:
                    self.primal.nodes()[xc]['potentials'] = original_pots[xc] + xc_comp_pots[xc]

            if epsilon < tolerance:
                print(f"with tolerence = {tolerance} and epsilon = {epsilon} the updates converge.")


            # for debugging, output marginals for all x, xc
            
            if log_path is not None:
                log['epsilon'].append(epsilon)
                """
                log['real_marginals'].append({})
                log['query_marginals'].append({})
                #for x in sorted_vars:
                for x in self.primal.nodes()
                    log['real_marginals'][-1][x] = []

                    x_mar = [[float(m[0]), float(m[1]), str(m[2])] for m in super()._get_full_marginal(x)]
                    log['real_marginals'][-1][x].append(x_mar)

                    if x not in sorted_vars:
                        continue

                    print(f"for var {x}, its copies: {sorted_copies[x]}")                    
                    for xc in sorted_copies[x]:
                        xc_mar = [[float(m[0]), float(m[1]), str(m[2])] for m in super()._get_full_marginal(xc)]
                        log['real_marginals'][-1][x].append(xc_mar)

                    qvar = list(debug_queries[0].get_free_variables())
                    xqs = [q.substitute(
                        {qvar[0]: self.primal.nodes()[x]['var']}
                    ) for q in debug_queries]
                    Zx, volx = self.compute_volumes(xqs)
                    log['query_marginals'][-1][x] = [v / Zx for v in volx]
                """


            t4 = time.perf_counter()
            print(f"at iter {it}, it takes {t4 - t1} seconds")
            it += 1


        if log_path is not None:
            log['real_marginals'].append({})
            log['query_marginals'].append({})


            Zx, volx = self.compute_volumes(debug_queries)


            for x in self.primal.nodes():

                xvar = self.primal.nodes()[x]['var']

                """
                log['real_marginals'][-1][x] = []

                x_mar = [[float(m[0]), float(m[1]), str(m[2])] for m in super()._get_full_marginal(x)]
                log['real_marginals'][-1][x].append(x_mar)

                if x not in sorted_vars:
                    continue

                print(f"for var {x}, its copies: {sorted_copies[x]}")                    
                for xc in sorted_copies[x]:
                    xc_mar = [[float(m[0]), float(m[1]), str(m[2])] for m in super()._get_full_marginal(xc)]
                    log['real_marginals'][-1][x].append(xc_mar)

                """
                log['query_marginals'][-1][x] = [volx[i] / Zx
                                                 for i in range(len(debug_queries))
                                                 if xvar in debug_queries[i].get_free_variables()]


            
            logger.info(f"Dumping log at: {log_path}")
            with open(log_path, 'w') as log_file:
                json.dump(log, log_file, indent=2)


    @staticmethod
    def copy_name(n1, n2):
        return f"{n1}_{n2}"


if __name__ == '__main__':
    from mpwmi import set_logger_debug
    set_logger_debug()

    import matplotlib.pyplot as plt
    import json
    import numpy as np
    from numpy.random import RandomState

    x = Symbol("x", REAL)
    y = Symbol("y", REAL)
    z = Symbol("z", REAL)

    """
        example of triangle-shape formula with a loop
    """
    f = And(LE(Real(0), x), LE(x, Real(1)),
            LE(Real(0), y), LE(y, Real(1)),
            LE(Real(0), z), LE(z, Real(1)),
            LE(Plus(z, x), Real(1)),
            LE(Plus(x, y), Real(1)),
            LE(Plus(y, z), Real(1)))

    per_lit_w = [
        Ite(LE(x, y), Real(2), Real(1)),
        Ite(LE(y, z), Real(3), Real(1)),
        Ite(LE(z, x), Real(5), Real(1)),
        #Ite(LE(x, y), x, Real(1)),
        #Ite(LE(y, z), y, Real(1)),
        #Ite(LE(z, x), z, Real(1)),
        Ite(LE(x, Real(0.5)), Real(2), Real(1)),
        #Ite(LE(x, Real(0.5)), x, Real(1)),
        Ite(LE(y, Real(0.5)), Real(2), Real(1)),
        Ite(LE(z, Real(0.5)), Real(2), Real(1))
        ]
    w = Times(per_lit_w)
    queries = [LE(x, Real(0.5)),
               LE(y, Real(0.5)),
               LE(z, Real(0.5)),
               LE(Plus(x, y), Real(0.5)),
               LE(Plus(y, z), Real(0.5))]


    rand_gen = RandomState(666)

    MAX_NCOMPLITS = 2 # 5
    REPEAT = 1

    MAX_ITERS = 30 # 20

    diffs = []
    # xs = list(range(1, MAX_NCOMPLITS+1))
    xs = list(range(MAX_NCOMPLITS, MAX_NCOMPLITS + 1))

    wmipa = WMI(f, w)
    Z_pa, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)


    vol_pa_res = []
    for i, q in enumerate(queries):
        vol_pa, _ = wmipa.computeWMI(q, mode=WMI.MODE_PA)
        vol_pa_res.append(vol_pa)

    for n_comp_lits in xs:

        diffn = []
        for _ in range(REPEAT):

            rcmi = RCWMI(f, w, n_comp_lits=n_comp_lits, rand_gen=rand_gen, max_compensate_iters=MAX_ITERS,
                         n_processes=4)
            Z_mp, vol_mp = rcmi.compute_volumes(queries=queries, cache=False)

            print("==================================================")
            print(f"Z\t\t\t{Z_mp}\t{Z_pa}")

            diffq = []
            for i, q in enumerate(queries):
                # vol_pa, _ = wmipa.computeWMI(q, mode=WMI.MODE_PA)
                vol_pa = vol_pa_res[i]
                print(f"Prob of query: {q}\t\t\t{vol_mp[i] / Z_mp}\t{vol_pa / Z_pa}")

                diffq.append(np.abs(vol_mp[i] / Z_mp - vol_pa / Z_pa))

            diffn.append(np.mean(diffq))

        diffs.append(diffn)


    with open("results.json", 'w') as f:
        json.dump(diffs, f)



    avg = [np.mean(d) for d in diffs]
    std = [np.std(d) for d in diffs]

    plt.plot(xs, avg)
    plt.fill_between(xs,
                     [avg[i] - std[i] for i in range(len(xs))],
                     [avg[i] + std[i] for i in range(len(xs))],
                     alpha=0.5)

    plt.xlabel("# compensating literals")
    plt.ylabel("Avg. query error")

    plt.savefig("results.png")
    plt.show()
    plt.clf()
