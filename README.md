# AMVAR
Adaptive Multivariate AutoRegregression for spectral analysis
- an alternative to Fourier analysis as it allows for BOTH, high spectral AND temporal resolution
- particularly useful when many multiple trials are present
- implemented using the least-squares method
- allows to calculate the following spectral quantities: power, coherence, partial coherence, coherence phase, phase of partial coherence, multiple coherence
- for further information on the method see e.g.:
Ding, M., Bressler, S., Yang, W. et al.: Short-window spectral analysis of cortical event-related potentials by adaptive multivariate autoregressive modeling: data preprocessing, model validation, and variability assessment. Biol Cybern 83, 35–45 (2000). https://doi.org/10.1007/s004229900137

## File info
- _usage.py_ contains an example using surrogate data and basic visualization
- _amvar.py_ contains the core functions
