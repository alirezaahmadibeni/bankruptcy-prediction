from __future__ import division
import math
import random
from sklearn.metrics.pairwise import euclidean_distances
from ann import NeuralNetwork
from numpy import array
import os


class GWO(object):
    def __init__(self, n_population, bankruptcy_data, non_bankruptcy_data, clusters_data, cluster_centers,
                 threshold_list,
                 population=None):
        self.threshold_list = threshold_list
        self.bankruptcy_data = bankruptcy_data
        self.non_bankruptcy_data = non_bankruptcy_data
        self.neural_network = NeuralNetwork(n_inputs=6, n_outputs=2, n_neurons_to_hl=6, n_hidden_layers=1)
        self.n_population = n_population
        self.population = population or self._makepopulation()
        self.saved_cluster_data = clusters_data
        self.cluster_centers = cluster_centers
        self.predict_bankruptcy = []
        self.predict_non_bankruptcy = []
        self.fitness_list = []  # list of  chromosome and fitness
        self.currentUnderSampling = None
        self.predict_position = None
        self.search_main()



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
        self.init_neural_network(self.predict_position)
        return self.neural_network.update(data)[0]

    def _makepopulation(self):
        pop_list = []
        for i in range(0, self.n_population):
            weights = [random.uniform(-5, 5) for _ in range(0, 36)]
            out_weights = [random.uniform(-5, 5) for _ in range(0, 12)]

            # make threshold list
            threshold1 = [random.uniform(threshold[0], threshold[1]) for threshold in self.threshold_list]
            position = threshold1 + weights + out_weights

            pop_list.append(position)

        return pop_list

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




    def search_main(self):
        Max_iter = 3

        for index in range(0, Max_iter):


            for position in self.population:
                print("underSampling : Cut off % s" % str(position[:5]))
                self.currentUnderSampling = self.cbeus(position[:5])
                self.init_neural_network(position)
                self.predict_non_bankruptcy = []
                self.predict_bankruptcy = []
                for instance in self.currentUnderSampling:
                    self.predict_non_bankruptcy.append(self.neural_network.update(instance)[0])
                for instance in self.bankruptcy_data:
                    self.predict_bankruptcy.append(self.neural_network.update(instance)[0])

                fitness_value = self.g_mean(len(self.currentUnderSampling))

                self.fitness_list.append([position, fitness_value])

            self.fitness_list.sort(key=lambda x: x[1])

            # Update Alpha, Beta, and Delta

            Alpha_pos = self.fitness_list[0][0] # Update alpha

            Beta_pos = self.fitness_list[1][0] # Update beta

            Delta_pos = self.fitness_list[2][0] # Update delta

            a = 2-index*(( 2 ) / Max_iter)  # a decreases linearly from 2 to 0


            for position in self.population:
                for j in range(0, len(position)):

                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]

                    A1 = 2 * a * r1 - a  # Equation (3.3)
                    C1 = 2 * r2  # Equation (3.4)

                    D_alpha = abs(C1 * Alpha_pos[j]-position[j])  # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1*D_alpha  # Equation (3.6)-part 1

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a  # Equation (3.3)
                    C2 = 2 * r2  # Equation (3.4)

                    D_beta = abs(C2 * Beta_pos[j] - position[j])  # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a  # Equation (3.3)
                    C3 = 2 * r2  # Equation (3.4)

                    D_delta = abs(C3 * Delta_pos[j] - position[j])  # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3

                    position[j] = (X1+X2+X3) / 3  # Equation (3.7)

            if index == Max_iter - 1:
                os.system('cls' if os.name == 'nt' else 'clear')
                self.fitness_list.sort(key=lambda x: x[1])
                print("The Optimization Weights For Predict Is: %s " % str(self.fitness_list[0][0][5:]))
                self.predict_position = self.fitness_list[0][0][5:]

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
