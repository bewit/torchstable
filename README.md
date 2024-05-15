# torchstable
This is a torch implementation of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html.
Thanks to the authors of SciPy and of the referenced file and its dependencies in particular.
For docs, please see scipy.stats.levy_stable.
This implementation only provides the file stable.py, which offers the computation of the pdf and the cdf of stable distributed random variables within the torch.distributions interface.
A proper package structure might be provided in the future, but you know how promises like these work out.

I tried to stay as close to the original numpy implementation as possible (and kind of sensible).
Nonetheless, I'm not that familiar with the computational and mathematical tweaks here, and the implemented solution might appear hacky.
Only the piecewise integration works, and as all integrations are carried out using MonteCarlo integration, the results are approximations and the respective errors are not quantified here (this is partly due to me not knowing how to do this and not having the time for researching it right now).
Precision can be increased by choosing more MCIntegration evaluation nodes or increasing the number of repetitions per integration.
According to the WLLN, the results converge in the infinite, but I cannot derive the convergence rate here; and the computation gets heavy quite quickly.
Please note that comparison to the scipy implementation still has to be carried out.

If you want to contact me, you can find me via ORCID: https://orcid.org/0000-0002-6861-7301.
If you find this implementation helpful, feel free to let me know and/or acknowledge it in your work :)
