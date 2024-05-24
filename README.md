# torchstable
This is a torch implementation of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html.
Thanks to the authors of SciPy and of the referenced file and its dependencies in particular.
For docs, please see scipy.stats.levy_stable.

I tried to stay as close to the original numpy implementation as possible (and kind of sensible).
Nonetheless, I'm not that familiar with the computational and mathematical tweaks here, and the implemented solution might appear hacky.
Only the piecewise integration works, and as all integrations are carried out using MonteCarlo integration, the results are approximations and the respective errors are only inspected in a small analysis (see analysis.py and the respective png/txt)
Precision can be increased by choosing more MCIntegration evaluation nodes
According to the WLLN, the results converge in the infinite, but I cannot derive the convergence rate here; and the computation gets heavy quite quickly.

If you want to contact me, you can find me via ORCID: https://orcid.org/0000-0002-6861-7301.
If you find this implementation helpful, feel free to let me know :)
