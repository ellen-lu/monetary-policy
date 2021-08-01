# monetary-policy
Working code for Monetary Policy paper with Giri Parameswaran and Carola Binder. The 'rrm' folder contains the data from Riboni and Ruge-Murcia's paper, *Monetary Policy by Committee: Consensus, Chairman Dominance, or Simple Majority?*. The ipython notebook file 'rrm_main' contains the most recently updated code, and references both papers linked to in this repo. Both papers are still in progress. 

The porject is built on Jupyter notebook. Once dowloaded, 'rrm_main' can be run on Jupyter notebook and it calls 'table_builder.py' and 'class_children.py'. 'table_builder.py' contains functions that read and clean the data. 'class_children.py' defines Bank class.

'rrm_main' contains three classes and some preliminary results:

1. Nash class is a child of Bank class defined in 'class_children.py'. First few methods within the Nash class defines variables (e.g. ideal interest rate, left and right extreme constants, left and right extereme ideal interest rate). Methods guess_rho, give_z, give_rho, and get_rho_z_helper recursively solve rho and zeta (equations 7, 8 & 9 in the paper). Method grad_ascent_nash implements gradient ascent using Adam method (adapted from: https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c). Two methods compare_grad_to_lln and check_grad_improving_lln are built to ensure the results are plausible. Hessian matrix, Fisher information, and errors are still in progress.

2. Continuous class is a child class of Nash class, and it is the case where there is a continuum of players. It defines log likelihhod function (equation 10 in the paper) and evaluates the gradients by breaking the log likelihood function into three pieces, calculating the gradient of each piece, then summing the three pieces together.

3. Discrete class is a child class of Nash class, and it is the case where there is a discrete number of players. It defines the log liklihood function and evaluates the gradients. Different from the continuous case, the discrete case has a simpler log liklihood function because there isn't an one-to-one mapping between i and zeta.


