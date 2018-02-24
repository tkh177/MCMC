import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# define the target distribution mixture components
target_mix = []  # creating empty list
mu_1 = 20
sigma_1 = 3
target_mix.append(norm(mu_1, sigma_1))
mu_2 = 40
sigma_2 = 10
target_mix.append(norm(mu_2, sigma_2))

# normalising factor w
w = 1.0 / len(target_mix)


def eval_target(x):
    """
    Evaluates the often un-normalised target distribution p(x)
    :param x:
    :return:
    """
    p_x = 0.0
    for component in target_mix:
        p_x += w * component.pdf(x)

    return p_x


def eval_proposal(x, mu, std):
    """
    evaluates the proposal distribution at x, which over here is an isotropic gaussian with mean mu and standard
    deviation std
    :param x: value of something
    :param mu: mean of normal distribution
    :param std: standard deviation of normal distribution
    :return: probability density at that point in the
    """
    q_x = norm(mu, std). pdf(x)

    return q_x


def sample_from_proposal(mu, std, no_samples = 1):
    """
    Samples from the proposal distribution q
    :param mu:
    :param std:
    :param no_samples:
    :return:
    """
    # evaluate proposal distribution for a number of samples
    x = norm(mu, std).rvs(no_samples)

    return float(x)


def main():

    # define the number of samples
    n = 10000
    # define standard deviation
    std = 10.0

    # initialise at mode of target distribution(might have been found using optimization)
    x_old = mu_1
    samples = []
    accepted = 0.0

    # start sampling
    for i in range(n):

        # produce some status output
        if i % 200 == 0:
            print('Running iteration ', i)

        # draw samples from the proposal distribution
        x = sample_from_proposal(x_old, std)  # take one sample from proposal distribution

        # generate random number from uniform distribution
        z = np.random.uniform(0, 1.0, 1)  # one random number from a normal distribution with mean 0 and std 1

        # define A which is the threshold for z
        fraction_a = eval_target(x) * eval_proposal(x_old, x, std)
        a = min(1.0, fraction_a)

        # accept or reject new sample
        if z < a:
            samples.append(x)
            x_old = x  # update the previous value of x_0, which is the first value in the markov chain
            accepted += 1  # add 1 to the total number of samples accepted
        else:
            samples.append(x_old)  # take the old value if rejected

    # print the state
    acceptance_rate = (accepted / n) * 100
    print('The Acceptance Rate is ', acceptance_rate)

    # show actual target distribution using a plot
    state_space = np.linspace(0, 100, 10000)
    p_x = eval_target(state_space)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(n * np.ones(len(state_space)), state_space, p_x, linewidth=2.0, label='Actual Distribution p(x)')
    ax1.plot(range(n), samples, np.zeros(n), label='Samples Drawn')
    ax1.set_xlabel('Sample Numer')
    ax1.set_ylabel('State Space (x)')
    ax1.set_zlabel('p(x)')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(state_space, p_x, linewidth=2.0, label='Actual Distribution p(x)')
    ax2.hist(samples, 50, normed=1, alpha=0.75, label='Sampled Distribution p(x)')
    ax2.set_xlabel('State Space (x)')
    ax2.set_ylabel('p(x)')
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()







