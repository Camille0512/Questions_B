from scipy.stats import norm
from numpy import array, mean, std, arange
import numpy as np
from timeit import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt


def print_info(text):
    def decorate(func):
        def wrap(*args, **kwargs):
            print("\n========== {} ==========".format(text))
            res = func(*args, **kwargs)
            print("========================================================\n")
            return res

        return wrap

    return decorate


def variance_calculator(left_range: int, right_range: int, n: int):
    """
    Calculate the possible variance of a given range.
    left_range: The left side of the interval.
    right_range: The right side of the interval.
    n: The amount of numbers within the range.
    """
    sim_n, diff, mean_p, prices = n, right_range - left_range, mean([left_range, right_range]), []
    for i in range(sim_n + 1):
        prices.append(left_range + i * diff / sim_n)
    prices = array(prices) - mean_p
    print(round(std(prices), 3))


def expon_pdf(x: int, lambda_val: int):
    """
    The PDF function of the exponential distribution.
    Get the corresponding probability according to given number and lambda value.
    :params x: The given number.
    :params lambda_val: The lambda value.
    :return: The probability of the given number under given exponential distribution.
    """
    return lambda_val * np.exp(-lambda_val * x) if x >= 0 else 0


@print_info("Generate Exponential Distributed Random Numbers")
def expon_num_gen(lambda_val: int, amt: int, scale: int, std_scale=True):
    """
    Generate exponential distributed random number according to a given exponential distribution.
    :params lambda_val: The lambda value.
    :params amt: The generated amount of random numbers.
    :params scale: The target scale, which is the generated object.
    :params std_scale: Whether to scale the generated exponential number to [0, 1] range.
    :return: A list of generated numbers.
    """
    gen_num_list = []
    for i in range(amt):
        x = np.random.rand()
        if std_scale:
            prob = expon_pdf(x, lambda_val) / lambda_val  # Scale the probability to [0,1]
        else:
            prob = expon_pdf(x, lambda_val)
        gen_num_list.append(round(prob * scale, 6))
    return gen_num_list


@print_info("Plot Generated Exponential Distributed Random Numbers")
def distribution_gen_variance(gen_list: list, save_fig=False, fig_name="Distrib_variance.png"):
    """
    Graph the distribution of the generated variances.
    :params gen_list: The list of generated numbers.
    :params save_fig: Whether to save the graph.
    :params fig_name: Graph file name.
    """
    fig = plt.figure(figsize=(8, 5))
    plt.hist(gen_list, bins=50, edgecolor="white", label="variance")
    plt.legend()
    plt.xlabel("Variance")
    plt.ylabel("Frequencies")
    plt.title("The Distribution of the Generated Variance")
    if save_fig:
        plt.savefig(fig_name)
    else:
        plt.show()


def price_fake_prob(p: int, mu: float, sigma: float):
    """
    Calculate the possibility of counterfeit on given price point.
    :params p: The given price.
    :params mu: The mean value of the normal distribution.
    :params sigma: The standard deviation of the normal distribution.
    :return: The probability of counterfeit product the given price.
    """
    prob = norm.pdf((p - mu) / sigma)
    return round(prob, 6)


@print_info("Get fake probabilities on prices")
def gen_fake_prob_on_prices(prices: list, variances: list, mu: int):
    """
    Generate the probability of counterfeit products on each price.
    :params prices: The price list to be investigated.
    :params variances: The variance list of the normal distribution.
    :params mu: The mean value of the normal distribution.
    :return: The dictionary of counterfeit product probability on each price.
    """
    p_fake_prob = {}
    for price in tqdm(prices):
        p_fake_prob[price] = [price_fake_prob(price, mu, var) for var in variances]
    return p_fake_prob


@print_info("Monte Carlo Simulation for Probability Estimation")
def MCS_price_prob(price_range: list, **kwargs):
    """
    Monte Carlo Simulation to calculate the average probability of counterfeiting on each price.
    :params price_range: The price range list includes all the prices to be investigated.
    :params args: The counterfeit probabilities.
    :return: The simulation estimated probability of each price.
    """
    avg_prob = {}
    for price in price_range:
        group_prob = {}
        for name, prob_dict in kwargs.items():
            prob_list = prob_dict[0].get(price) if prob_dict[0].get(price) is not None else [0]
            mean_prob = mean(prob_list)
            group_prob[name] = mean_prob * prob_dict[1]
        avg_prob[price] = sum(group_prob.values())
        print(f'${price: 3} : {avg_prob[price]: 6f}')
    return avg_prob


def graph_price_prob_distrib(price_prob: dict, display_p_label=True, interval=1, save_fig=False,
                             figname="Prob_distrib.png"):
    """
    Graph the probability on each price.
    :params price_prob: The price probability dictionary with price as key, probability as values.
    :params display_p_label: Whether to display the point label on the graph.
    :params interval: The intervals for each labeled point.
    :params save_fig: Whether to save the graph.
    :params figname: Graph file name.
    """
    x, y = list(price_prob.keys()), list(price_prob.values())
    fig = plt.figure(figsize=(16, 8))
    plt.plot(x, y, marker="*")
    if display_p_label:
        for e, (a, b) in enumerate(zip(x, y)):
            if e % interval == 0:
                plt.text(a, round(b, 3), (a, round(b, 3)))
    plt.xlabel("Prices", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("The Probability of Counterfeit Products on Different Prices", fontsize=20)
    if save_fig:
        plt.savefig(figname)
    else:
        plt.show()


@print_info("Find optimal price and probability")
def find_best_prob(probability: dict):
    """
    Find the local and global optimized price and corresponding  probabilities.
    :params probability: The dictionary of prices and corresponding probabilities.
    :return: Both the local and global optimized price and corresponding probabilities.
    """
    local_opt, global_opt, local_p, global_p = float("inf"), float("inf"), 0, 0
    for e, (price, prob) in enumerate(probability.items()):
        if local_opt > prob and local_p == e:
            local_opt = prob
            local_p += 1
        if global_opt > prob:
            global_opt = prob
            global_p = e
    prob_list = list(probability.keys())
    return prob_list[local_p - 1], round(local_opt, 4), prob_list[global_p], round(global_opt, 4)


if __name__ == "__main__":
    # Check possible variances
    p1, p2 = 48, 68
    print("Possible variances are:")
    for i in arange(1, 50, 5):
        variance_calculator(p1, p2, i)

    # Settings
    # # The probability of $30 and $50 counterfeits.
    prob30, prob50 = 0.06, 0.04
    # # The mean of the counterfeit distributions
    mu30, mu50 = 30, 50
    # # The simulation times
    sn = 10000
    # # The lambda value for the $30 and $50 scenario.
    lambda30, lambda50 = 3, 2
    # # The maximum of variance
    max_var = 10
    # # Price range
    P30 = np.arange(30, 51)
    P50 = np.arange(30, 71)
    investigate_P = np.arange(30, 71)

    # Random simulator number generation
    x = expon_num_gen(3, sn, 10)
    distribution_gen_variance(x)

    # Calculate the probability of counterfeit product on each given price
    # # Generate variance list of the $30 and $50 correspondingly.
    vars30 = expon_num_gen(lambda30, sn, max_var)
    vars50 = expon_num_gen(lambda50, sn, max_var)
    p30_fake_prob = gen_fake_prob_on_prices(P30, vars30, mu30)
    p50_fake_prob = gen_fake_prob_on_prices(P50, vars50, mu50)
    # # Get the estimated probability of each price.
    avg_prob = MCS_price_prob(investigate_P, price30=[p30_fake_prob, prob30], price50=[p50_fake_prob, prob50])
    print(avg_prob)
    graph_price_prob_distrib(avg_prob, interval=3)

    # Find the optimal prices and probabilities
    local_price, local_opt, global_price, global_opt = find_best_prob(avg_prob)
    print("The optimized local price is {}, and the probability is {}".format(local_price, local_opt))
    print("The optimized global price is {}, and the probability is {}".format(global_price, global_opt))