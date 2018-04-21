from __future__ import division
import random
from sklearn.metrics.pairwise import euclidean_distances
from ann import NeuralNetwork
from numpy import array
import numpy as np
import math
import os


class GA(object):
    def __init__(self, n_population, pc, pm, bankruptcy_data, non_bankruptcy_data, clusters_data, cluster_centers, threshold_list,
                 population=None):
        self.threshold_list = threshold_list
        self.bankruptcy_data = bankruptcy_data
        self.non_bankruptcy_data = non_bankruptcy_data
        self.neural_network = NeuralNetwork(n_inputs=6, n_outputs=2, n_neurons_to_hl=6, n_hidden_layers=1)
        self.n_population = n_population
        self.p_crossover = pc  # percent of crossover
        self.p_mutation = pm  # percent of mutation
        self.population = population or self._makepopulation()
        self.saved_cluster_data = clusters_data
        self.cluster_centers = cluster_centers
        self.predict_bankruptcy = []
        self.predict_non_bankruptcy = []
        self.fitness_list = []  # list of  chromosome and fitness
        self.currentUnderSampling = None
        self.predict_chromosome = None
        self.fitness()

    def init_neural_network(self, chromosome):
        # remove threshold from chromosome list
        primary_weights = chromosome[5:]
        matrix_list = []
        for i in range(0, int(len(primary_weights) / 6) - 1):
            matrix_list.append(primary_weights[i * 6:(i + 1) * 6])

        weights_matrix = array(matrix_list)

        layers = self.neural_network.layers

        i = 0
        for neuron in layers[0].neurons:
            neuron.set_weights(weights_matrix[:, i])
            i += 1

        layers[1].neurons[0].set_weights(primary_weights[-6:])

    def performance_measure(self):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for item in self.bankruptcy_data:
            if self.predict(item) > 0.5:
                fp += 1
            else:
                tp += 1
        for item in self.non_bankruptcy_data:
            if self.predict(item) > 0.5:
                tn += 1
            else:
                fn += 1

        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)


        print("TP is : %s" % (str(tp)))
        print("FP is : %s" % (str(fp)))
        print("FN is : %s" % (str(fn)))
        print("TN is : %s" % (str(tn)))
        print("G-MEAN : %s" % (str(math.sqrt(sensitivity * specificity))))

        print("Hit-ratio : %s" % (str((tp + tn) / (tp + fn + fp + tn))))

    def predict(self, data):
        self.init_neural_network(self.predict_chromosome)
        return self.neural_network.update(data)[0]


    def _makepopulation(self):
        pop_list = []
        for i in range(0, self.n_population):
            weights = [random.uniform(-5, 5) for _ in range(0, 36)]
            out_weights = [random.uniform(-5, 5) for _ in range(0, 12)]

            # make threshold list
            threshold1 = [random.uniform(threshold[0], threshold[1]) for threshold in self.threshold_list]
            chromosome = threshold1 + weights + out_weights

            pop_list.append(chromosome)

        return pop_list

    '''
    b : the number of bankruptcy firms
    BAi : the classification accuracy of ith instances of bankruptcy firms
    n : the number of non-bankruptcy firms
    NAj : the classification accuracy of jth instances of non-bankruptcy firms
    POi : the predicated output of ith instances of bankruptcy firms
    AOi : the actual output of ith instances of non-bankruptcy firms
    POj : the predicated output of jth instances of non-bankruptcy firms
    AOj : the actual output of jth instances of non-bankruptcy firms

    '''

    def ba_i(self, poi):
        if poi < 0.5:
            return 1
        return 0

    def na_j(self, poj):
        if poj > 0.5:
            return 1
        return 0

    def cbeus(self, thresholds):  # the rule structure for the cluster-based underSampling base on GA

        i = 0
        undersampling_clusters = []
        for cluster in self.saved_cluster_data:
            for instance in cluster:
                if euclidean_distances([instance], [self.cluster_centers[i]]) < thresholds[i]:
                    undersampling_clusters.append(instance)

            i += 1

        return undersampling_clusters

    def fitness(self):

        fitness_sum = 0
        trials = 3

        for index in range(0, trials):

            for item in self.population:
                print("underSampling : Cut off % s" % str(item[:5]))
                self.currentUnderSampling = self.cbeus(item[:5])
                self.init_neural_network(item)
                self.predict_non_bankruptcy = []
                self.predict_bankruptcy = []
                for instance in self.currentUnderSampling:
                    self.predict_non_bankruptcy.append(self.neural_network.update(instance)[0])
                for instance in self.bankruptcy_data:
                    self.predict_bankruptcy.append(self.neural_network.update(instance)[0])

                fitness_value = self.g_mean(len(self.currentUnderSampling))

                self.fitness_list.append([item, fitness_value])
                fitness_sum += fitness_value

            self.fitness_list.sort(key=lambda x: x[1])
            self._select_parents(fitness_sum)
            self.fitness_list.sort(key=lambda x: x[1])
            self.population = []

            for item in self.fitness_list:
                self.population.append(item[0])
                if len(self.population) == 5:
                    break

            if index == trials-1:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("The Optimization Weights For Predict Is: %s " % str(self.population[0][5:]))
                self.predict_chromosome = self.population[0][5:]

        self.performance_measure()




    def g_mean(self, n):

        b = len(self.bankruptcy_data)

        sum_bankruptcy = 0
        sum_non_bankruptcy = 0

        for item in self.predict_non_bankruptcy:
            sum_non_bankruptcy += self.ba_i(item)

        for item in self.predict_bankruptcy:
            sum_bankruptcy += self.na_j(item)

        return math.sqrt((1 / b) * sum_bankruptcy * (1 / n) * sum_non_bankruptcy)

    def cxOnePoint(self, ind1, ind2):
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
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

        return ind1, ind2

    def swapMutation(self, ind1):

        size = len(ind1)
        swpoint1 = random.randint(1, size - 1)
        swpoint2 = random.randint(1, size - 1)

        ind1[swpoint1], ind1[swpoint2] = ind1[swpoint2], ind1[swpoint1]
        return ind1

    def _select_parents(self, fitness_sum):
        """
        Roulette wheel selection
        Selects parents from the given population

        Args :
        population (list) : Current population from which parents will be selected
        fitness_sum (number) : Summation of all fitness value

        Returns :
        parents (IndividualGA, IndividualGA) : selected parents
        """

        probability = []

        for item in self.fitness_list:
            probability.append(item[1] / fitness_sum)
            item.append(item[1] / fitness_sum)

        ncrossover = math.ceil(self.n_population * self.p_crossover / 2)  # number of crossover offspring
        nmutation = math.ceil(self.n_population * self.p_mutation)  # number of mutation offspring

        selection_probability = set()

        while len(selection_probability) < ncrossover:
            selection_probability.add(random.uniform(0, 1))

        probability = np.cumsum(probability).tolist()



        def roulette(prob):
            for i in range(0, len(probability)):
                if prob < probability[i]:
                    return self.fitness_list[i][0]



        crossover_list = []

        for item in list(selection_probability):
            crossover_list.append(roulette(item))
            if len(crossover_list) == 2:
                inde1, inde2 = self.cxOnePoint(crossover_list[0][:], crossover_list[1][:])

                # init the neural network with the individual 1

                self.currentUnderSampling = self.cbeus(inde1[:5])
                self.init_neural_network(inde1)
                self.predict_non_bankruptcy = []
                self.predict_bankruptcy = []
                for instance in self.currentUnderSampling:
                    self.predict_non_bankruptcy.append(self.neural_network.update(instance)[0])
                for instance in self.bankruptcy_data:
                    self.predict_bankruptcy.append(self.neural_network.update(instance)[1])

                fitness_value = self.g_mean(len(self.currentUnderSampling))

                self.fitness_list.append([inde1, fitness_value])


                # init the neural network with the individual 2

                self.currentUnderSampling = self.cbeus(inde2[:5])
                self.init_neural_network(inde2)
                self.predict_non_bankruptcy = []
                self.predict_bankruptcy = []
                for instance in self.currentUnderSampling:
                    self.predict_non_bankruptcy.append(self.neural_network.update(instance)[0])
                for instance in self.bankruptcy_data:
                    self.predict_bankruptcy.append(self.neural_network.update(instance)[1])

                fitness_value = self.g_mean(len(self.currentUnderSampling))

                self.fitness_list.append([inde2, fitness_value])

                crossover_list = []


        # create individual with mutation

        selection_probability = set()
        while len(selection_probability) < nmutation:
            selection_probability.add(random.uniform(0, 1))

        for item in list(selection_probability):

            inde3 = self.swapMutation(roulette(item))
            self.currentUnderSampling = self.cbeus(inde3[:5])
            self.init_neural_network(inde3)
            self.predict_non_bankruptcy = []
            self.predict_bankruptcy = []
            for instance in self.currentUnderSampling:
                self.predict_non_bankruptcy.append(self.neural_network.update(instance)[0])
            for instance in self.bankruptcy_data:
                self.predict_bankruptcy.append(self.neural_network.update(instance)[1])
            fitness_value = self.g_mean(len(self.currentUnderSampling))
            self.fitness_list.append([inde3, fitness_value])

