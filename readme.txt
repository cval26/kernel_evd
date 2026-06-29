Generating the numerical results included in

Chris Vales & Dimitrios Giannakis.
Accelerated decomposition of bistochastic kernel matrices by
low rank approximation.

- Run the "ks_datagen.py" file to generate the simulation results.
- Run the "ks_preproc.py" file to process the generated simulation
results.
- Run the "kevd.py" file to compute the approximate EVD.
- Use the notebook "ks_plots.ipynb" to generate the plots based on the
numerical results.

- The numerical results based on the smaller, reference training dataset
can be generated following the analogous procedure, modifying the
simulation parameters as needed.
The true reference EVD results can be produced by running the
"ks_reference.py" file.

The numerical results were generated using
- Python 3.12.3
- JAX 0.10.0
- Numpy 2.4.4
- Scipy 1.17.1
- Matplotlib 3.10.9
