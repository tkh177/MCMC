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
w = 1.0 / len(MIX)  # this is for the normalisation of the un-normalised function


def eval_target(x):
    """
    evaluates the often un-normalised target distribution p_star(x). Arbitrarily create a mixture of gaussians
    :param x: possible points
    :return: probability distribution function
    """
    # compute the weighted sum of all components
    p_x = 0.0
    for component in MIX:
        p_x += w * component.pdf(x)  # w normalised the monte carlo result
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
    N = 10000

    # Perform N rounds of rejection sampling - leading to a lower number of accepted points
    x = []
    accepted = 0
    for i in range(N):

        # produce some status output to tell us how much has been calculated so that it keeps printing
        if i % 1000 == 0:
            print('Running iteration number ', i)

        # start sampling from q
        x_0 = q_sample()  # Taking a sample from the proposed uniform distribution

        # Sample from the uniform distribution in the range [0, k * q(s)]
        z = uniform.rvs()

        # check whether to accept x_0
        if z <= eval_target(x_0):
            x.append(x_0)
            accepted += 1.0  # meaning we then increase it by 1

    acceptance_rate = (accepted / N) * 100
    print('The Acceptance Rate is ', acceptance_rate)

    # Plot a histogram of the samples
    plt.hist(x, 50, normed=1, alpha=0.75)

    # show actual target distribution
    state_space = np.linspace(0, 100, 10000)
    p_x = eval_target(state_space)
    plt.plot(state_space, p_x, linewidth=2.0)

    # plot proposal distribution for reference
    plt.plot(state_space, k * q(state_space), 'r', linewidth=2.0)

    plt.show()


if __name__ == "__main__":
    main()



