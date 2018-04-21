from __future__ import division
import numpy as np
import random, pdb
import operator
import math



def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    print(cxpoint)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


def swapMutation(ind1):
    size = len(ind1)
    swpoint = random.sample(range(1, size - 1), 2)
    print(swpoint)

    ind1[swpoint[0]], ind1[swpoint[1]] = ind1[swpoint[1]], ind1[swpoint[0]]
    return ind1


def roulette_selection(weights):
    '''performs weighted selection or roulette wheel selection on a list
    and returns the index selected from the list'''

    # sort the weights in ascending order
    sorted_indexed_weights = sorted(enumerate(weights), key=operator.itemgetter(1));
    indices, sorted_weights = zip(*sorted_indexed_weights);
    # calculate the cumulative probability
    tot_sum = sum(sorted_weights)
    prob = [x / tot_sum for x in sorted_weights]
    cum_prob = np.cumsum(prob)
    # select a random a number in the range [0,1]
    random_num = random.random()

    for index_value, cum_prob_value in zip(indices, cum_prob):
        if random_num < cum_prob_value:
            return index_value


l1 = [1, 2, 3, 4, 5, 6]
l2 = [7, 8, 9, 10, 11, 12]

print(cxOnePoint(l1[:], l2[:]))

print(l1)
print(l2)
print(swapMutation(l1[:]))

weights = [1, 2, 6, 4, 3, 7, 20]
print (roulette_selection(weights))


a = set()

i = 0
while i < 10:
    for ii in range(0, 20):
        print(math.fmod(ii, 2))
        if math.fmod(ii, 2) == 0:
            print("hi")
            break

    i += 1


