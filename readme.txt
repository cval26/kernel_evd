Generating the numerical results included in

Chris Vales & Dimitrios Giannakis.
Accelerated decomposition of bistochastic kernel matrices by
low rank approximation.

- Run the "ks_datagen.py" file to generate the simulation results.
- Run the "ks_preproc.py" file to process the generated simulation
results.
- Run the "ks_median.py" file to compute the kernel bandwidth using
the median heuristic.
- Run the "ks_subsample.py" file to extract 8192 samples from the
full training dataset, using the median bandwidth value.
- Run the "ks_bandwidth.py" file to use the extracted training subset
to calibrate the kernel bandwidth.
- Run the "ks_basis.py" file to compute the approximate eigenvalue
decompositions using the dilution and subsampling methods.
- Use the notebook "ks_plots.ipynb" to generate the plots based on the
numerical results.

- The numerical results based on the larger training dataset can be
generated following the analogous procedure, modifying the simulation
parameters as needed.

The numerical results were generated using
- Python 3.12.3
- Numpy 1.26.4
- Scipy 1.11.4
- Matplotlib 3.6.3
