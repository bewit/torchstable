import torch
from scipy.stats import levy_stable
import numpy as np
import itertools
from matplotlib import pyplot as plt
from tabulate import tabulate
from time import time

import torchstable


def _product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def _to_tensors(d):
    return {key: torch.tensor(d[key]) for key in d}


def MAE(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2))


def mci_evaluation_nodes():
    repetitions = 20
    Ns = (10**1, 10**2, 10**3, 10**4, 10**5, 10**6)
    data_size = 1000
    error_function = MAE

    alphas = (1.9, 1.4, 0.9)
    betas = (0.0,) # 0.5, 1.0)
    locs = (0.0,)
    scales = (1.0,)
    options = {"alpha": alphas, "beta": betas, "loc": locs, "scale": scales}

    parameter_set = _product_dict(**options)

    for params in parameter_set:
        torch_results = torch.empty(len(Ns), repetitions, data_size)
        scipy_results = torch.empty(len(Ns), repetitions, data_size)
        errors = torch.empty(len(Ns), repetitions)
        times = torch.empty(len(Ns))

        for n, N in enumerate(Ns):
            start_time = time()
            torchstable.N = N

            for repetition in range(repetitions):
                data = torch.randn(data_size)

                torch_stable = torchstable.TorchStable(**_to_tensors(params))
                torch_densities = torch_stable.pdf(data)

                torch_results[n, repetition, :] = torch_densities

                scipy_stable = levy_stable(**params)
                scipy_densities = scipy_stable.pdf(data)
                scipy_results[n, repetition, :] = torch.tensor(scipy_densities)

                errors[n, repetition] = error_function(torch_densities, scipy_densities)
            
            time_per_repetition = (time() - start_time) /  repetitions
            times[n] = time_per_repetition

        means = torch.mean(errors, dim=1)
        error = [torch.abs(means - torch.min(errors, dim=1)[0]), torch.abs(means - torch.max(errors, dim=1)[0])]
        print(f"params: {params}")
        print(tabulate({"N": Ns, "means": means, "error_min": torch.min(errors, dim=1)[0], "error_max": torch.max(errors, dim=1)[0], "time": times}, headers="keys"))
        x_scatter = np.random.uniform(0.8, 1.2)
        plt.errorbar(np.array(Ns) * x_scatter, means, yerr=error, fmt="o", label=f"S({params['alpha']},{params['beta']})")
        
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("error")
    plt.title(f"MCI: error of levy_stable vs TorchStable with N evaluation nodes, {error_function.__name__}, i={repetitions}, |X|={data_size}")
    plt.savefig(f"./MCI_levystable_vs_TorchStable_{error_function.__name__}_i{repetitions}_d{data_size}")
    plt.show()
    plt.close()




if __name__ == "__main__":
    mci_evaluation_nodes()