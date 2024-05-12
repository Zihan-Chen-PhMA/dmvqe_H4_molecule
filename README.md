requirements:

python >= 3.10

joblib >= 1.4

qiskit < 1

Also, as a feature of qiskit<1, calculations using GPU is not supported on windows. (Using CPU 
calculations on windows is fine.) Linux/WSL2 supports GPU (with CUDA) calculations.  

To parallelize things, joblib is used and the number of threads is set to be 25. You may change it according 
to your own liking and the number of cores of your computer. Piece of advice: running this on a cheap laptop may be very slow. 

how to run: (shell script for example)

$ python noise_sweep.py

After this is completed, run

$ python readnplot.py 

to plot all results. 


