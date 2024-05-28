import math
import torch
from torchstable import TorchStable
from matplotlib import pyplot as plt
import seaborn as sns
from tabulate import tabulate

linestyles = ["-", "--", "-.", ":", ][::-1]



def plot_characteristic_function(grouped_parameter_set: list[list[dict[str, torch.Tensor]]], parametrization: str, filename: str, linspace: torch.Tensor = torch.linspace(-10, 10, 200)) -> None:
    number_of_groups = len(grouped_parameter_set)
    fig, axes = plt.subplots(nrows = number_of_groups, ncols=2)
    fig.set_size_inches(10, 3 * number_of_groups)
    fig.set_dpi(2000)
    
    for i, group in enumerate(grouped_parameter_set):
        group_axis = axes[i]
        for j, params in enumerate(group):
            alpha = torch.tensor(params["alpha"])
            beta = torch.tensor(params["beta"])
            loc = torch.tensor(params["loc"])
            scale = torch.tensor(params["scale"])
            representation = f"S({alpha}, {beta})"
            torch_stable = TorchStable(alpha, beta, loc, scale)
            torch_stable.parametrization = parametrization

            cf_values = torch_stable.characteristic_function(linspace)

            reals = cf_values.real
            imaginaries = cf_values.imag

            group_axis[0].plot(linspace, reals, label=representation, linestyle=linestyles[-j-1])
            group_axis[1].plot(linspace, imaginaries, label=representation, linestyle=linestyles[-j-1])
        
        group_axis[0].legend()
        group_axis[0].set_ylim((-0.6, 1.1))
        group_axis[0].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        group_axis[1].legend()
        group_axis[1].set_ylim((-0.6, 0.6))
        group_axis[1].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])

    fig.suptitle(f"Real (left) and imaginary (right) part of characteristic_function\n of  {parametrization}-parametrized stable distribution")
    fig.savefig(filename)
    plt.close()


def plot_empirical_characteristic_function(grouped_parameter_set: list[list[dict[str, torch.Tensor]]], parametrization: str, filename: str, n_samples: int, seed: int, linspace: torch.Tensor = torch.linspace(-10, 10, 200)) -> None:
    from scipy.stats import levy_stable
    def empirical_characteristic_function(data, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.exp(1j * (data @ t.T)), axis=0)
    
    number_of_groups = len(grouped_parameter_set)
    fig, axes = plt.subplots(nrows = number_of_groups, ncols=2)
    fig.set_size_inches(10, 3 * number_of_groups)
    fig.set_dpi(2000)
    
    for i, group in enumerate(grouped_parameter_set):
        group_axis = axes[i]
        for j, params in enumerate(group):
            alpha = torch.tensor(params["alpha"])
            beta = torch.tensor(params["beta"])
            loc = torch.tensor(params["loc"])
            scale = torch.tensor(params["scale"])
            representation = f"ECF(S({alpha}, {beta}))"

            levy_stable.parameterization = parametrization
            scipy_stable = levy_stable(alpha=alpha, beta=beta, loc=loc, scale=scale)
            scipy_stable.parameterization = parametrization
            samples = torch.tensor(scipy_stable.rvs((n_samples, 1), random_state=seed))
            cf_values = empirical_characteristic_function(data=samples, t=linspace.reshape(-1, 1))

            reals = cf_values.real
            imaginaries = cf_values.imag

            group_axis[0].plot(linspace, reals, label=representation, linestyle=linestyles[-j-1])
            group_axis[1].plot(linspace, imaginaries, label=representation, linestyle=linestyles[-j-1])
        
        group_axis[0].legend()
        group_axis[0].set_ylim((-0.6, 1.1))
        group_axis[0].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        group_axis[1].legend()
        group_axis[1].set_ylim((-0.6, 0.6))
        group_axis[1].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])

    fig.suptitle(f"Real (left) and imaginary (right) part of empirical characteristic function \n estimated from {n_samples} {parametrization}-parametrized stable distribution")
    fig.savefig(filename)
    plt.close()


def plot_densities(grouped_parameter_set: list[list[dict[str, torch.Tensor]]], parametrization: str, filename: str, linspace: torch.Tensor = torch.linspace(-10, 10, 200)) -> None:
    number_of_groups = len(grouped_parameter_set)
    fig, axes = plt.subplots(nrows = number_of_groups)
    fig.set_size_inches(5, 3 * number_of_groups)
    fig.set_dpi(2000)

    for i, group in enumerate(grouped_parameter_set):
        group_axis = axes[i]
        for j, params in enumerate(group):
            alpha = torch.tensor(params["alpha"])
            beta = torch.tensor(params["beta"])
            loc = torch.tensor(params["loc"])
            scale = torch.tensor(params["scale"])
            representation = f"S({alpha}, {beta})"
            torch_stable = TorchStable(alpha, beta, loc, scale)
            torch_stable.parametrization = parametrization

            densities = torch_stable.pdf(linspace)

            group_axis.plot(linspace, densities, label=representation, linestyle=linestyles[-j-1])

        group_axis.legend()
        group_axis.set_ylim((-0.1, 0.6))
        group_axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
    fig.suptitle(f"Probability density function of \n {parametrization}-parametrized stable distribution")
    fig.savefig(filename)
    plt.close()


def plot_densities_parametrizations(grouped_parameter_set: list[list[dict[str, torch.Tensor]]], filename: str, linspace: torch.Tensor = torch.linspace(-10, 10, 200)) -> None:
    number_of_groups = len(grouped_parameter_set)
    fig, axes = plt.subplots(nrows = number_of_groups, ncols=2)
    fig.set_size_inches(10, 3 * number_of_groups)
    fig.set_dpi(2000)

    for i, group in enumerate(grouped_parameter_set):
        group_axis = axes[i]
        for j, params in enumerate(group):
            alpha = torch.tensor(params["alpha"])
            beta = torch.tensor(params["beta"])
            loc = torch.tensor(params["loc"])
            scale = torch.tensor(params["scale"])
            representation = f"S({alpha}, {beta})"
            torch_stable = TorchStable(alpha, beta, loc, scale)

            torch_stable.parametrization = "S0"
            densities_S0 = torch_stable.pdf(linspace)
            group_axis[0].plot(linspace, densities_S0, label=representation, linestyle=linestyles[-j-1])

            torch_stable.parametrization = "S1"
            densities_S1 = torch_stable.pdf(linspace)
            group_axis[1].plot(linspace, densities_S1, label=representation, linestyle=linestyles[-j-1])

        group_axis[0].legend()
        group_axis[0].set_ylim((-0.1, 0.6))
        group_axis[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        group_axis[1].legend()
        group_axis[1].set_ylim((-0.1, 0.6))
        group_axis[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
    fig.suptitle(f"Probability density function of S0- (left) and \n S1- (right) parametrization of stable distribution")
    fig.savefig(filename)
    plt.close()


def plot_distribution_parametrizations(grouped_parameter_set: list[list[dict[str, torch.Tensor]]], filename: str, linspace: torch.Tensor = torch.linspace(-10, 10, 200)) -> None:
    number_of_groups = len(grouped_parameter_set)
    fig, axes = plt.subplots(nrows = number_of_groups, ncols=2)
    fig.set_size_inches(10, 3 * number_of_groups)
    fig.set_dpi(2000)

    for i, group in enumerate(grouped_parameter_set):
        group_axis = axes[i]
        for j, params in enumerate(group):
            alpha = torch.tensor(params["alpha"])
            beta = torch.tensor(params["beta"])
            loc = torch.tensor(params["loc"])
            scale = torch.tensor(params["scale"])
            representation = f"S({alpha}, {beta})"
            torch_stable = TorchStable(alpha, beta, loc, scale)

            torch_stable.parametrization = "S0"
            densities_S0 = torch_stable.cdf(linspace)
            group_axis[0].plot(linspace, densities_S0, label=representation, linestyle=linestyles[-j-1])

            torch_stable.parametrization = "S1"
            densities_S1 = torch_stable.cdf(linspace)
            group_axis[1].plot(linspace, densities_S1, label=representation, linestyle=linestyles[-j-1])

        group_axis[0].legend()
        group_axis[0].set_ylim((-0.1, 1.1))
        group_axis[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        group_axis[1].legend()
        group_axis[1].set_ylim((-0.1, 1.1))
        group_axis[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(f"Cumulative distribution function of S0- (left) and \n S1- (right) parametrization of stable distribution")
    fig.savefig(filename)
    plt.close()



if __name__ == "__main__":
    params_grouped_by_alpha = [
        [
            {"alpha": 2.0, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.9, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.9, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.9, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
        ], 
        [
            {"alpha": 1.5, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.5, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.5, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
        ], 
        [
            {"alpha": 1.0, "beta": 0.0, "loc": 0.0, "scale": 1.0},        
            {"alpha": 1.0, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.0, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
        ],
        [
            {"alpha": 0.5, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 0.5, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 0.5, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
        ],
    ]
    params_grouped_by_beta = [
        [
            {"alpha": 1.9, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.5, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.0, "beta": 0.0, "loc": 0.0, "scale": 1.0},
            {"alpha": 0.5, "beta": 0.0, "loc": 0.0, "scale": 1.0},
        ], 
        [
            {"alpha": 1.9, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.5, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 1.0, "beta": 0.5, "loc": 0.0, "scale": 1.0},
            {"alpha": 0.5, "beta": 0.5, "loc": 0.0, "scale": 1.0},
        ],
        [
            {"alpha": 1.9, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
            {"alpha": 1.5, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
            {"alpha": 1.0, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
            {"alpha": 0.5, "beta": 1.0, "loc": 0.0, "scale": 1.0}, 
        ],
    ]


    plot_characteristic_function(grouped_parameter_set=params_grouped_by_alpha, parametrization="S0", filename="./visualization/characteristic_function_splitted_S0_grouped_by_alpha.png")
    plot_characteristic_function(grouped_parameter_set=params_grouped_by_alpha, parametrization="S1", filename="./visualization/characteristic_function_splitted_S1_grouped_by_alpha.png")

    plot_empirical_characteristic_function(grouped_parameter_set=params_grouped_by_alpha, parametrization="S0", n_samples=100, seed=49, filename="./visualization/ecf100_splitted_S0_grouped_by_alpha.png")
    plot_empirical_characteristic_function(grouped_parameter_set=params_grouped_by_alpha, parametrization="S1", n_samples=100, seed=49, filename="./visualization/ecf100_splitted_S1_grouped_by_alpha.png")

    plot_densities_parametrizations(grouped_parameter_set=params_grouped_by_beta, filename="./visualization/densities_parametrizatons_grouped_by_beta.png")

    plot_distribution_parametrizations(grouped_parameter_set=params_grouped_by_beta, filename="./visualization/distribution_parametrizations_grouped_by_beta.png")