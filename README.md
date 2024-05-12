requirements:

python >= 3.10
qiskit < 1
Also, as a feature of qiskit, calculations using GPU is not supported on windows. (Using CPU 
calculations on windows is fine.) Linux/WSL2 supports GPU (with CUDA) calculations.  

how to run: (shell script for example)
$ python noise_sweep.py
After this is completed, run
$ python readnplot.py 
to plot all results. 


