import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import expon
import matplotlib.pyplot as plt

# target distribution mixture components
MIX = []
mu_1 = 20
sigma_1 = 3
MIX.append(norm(mu_1, sigma_1))
mu_2 = 40
sigma_2 = 10
MIX.append(norm(mu_2, sigma_2))

# assign component weights
w = 1.0 / len(MIX)


def eval_target(x):
    """
    evaluates the often un-normalised target distribution p_star(x). Arbitrarily create a mixture of gaussians
    :param x: possible points
    :return: probability distribution function
    """
    # compute the weighted sum of all components
    p_x = 0.0
    for component in MIX:
        p_x += w * component.pdf(x)
    return p_x


def main():
    """
    Initialise the main function and implements it
    :return: Implements the sampling process
    """
    # First define proposed distribution q
    q = uniform(0, 100).pdf
    q_sample = uniform(0, 100).rvs
    k = 8

    # Number of samples to be drawn
    N = 50000

    # Perform N rounds of rejection sampling - leading to a lower number of accepted points
    x = []
    accepted = 0
    for i in range(N):

        # produce some status output
        if i % 1000 == 0:
            print('Running iteration number ', i)

        # start sampling from q
        x_0 = q_sample()  # Taking a sample from the proposed uniform distribution





if __name__ == "__main__":
    main()



