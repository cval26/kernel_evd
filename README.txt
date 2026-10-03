Generating the numerical results included in

Chris Vales & Dimitrios Giannakis.
Accelerated decomposition of bistochastic kernel matrices by
low rank approximation.

- Run "ks_datagen.py" to generate the simulation results.
- Run "ks_bandwidth.cu" to calibrate the kernel bandwidth.
- Run "ks_arpclm.cu" to compute the low rank kernel matrix
approximation.
- Run "ks_kevd.cu" to compute the approximate EVD of the
bistochastic normalization.
- Use "ks_plots.ipynb" to produce the plots based on the results.

- For the reference EVD using the smaller dataset, run
"ks_datagen.py" followed by "ks_preproc.py" and "ks_reference.py".
The corresponding plots can be produced using "ks_plots.ipynb".
