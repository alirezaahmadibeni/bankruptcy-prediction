# *-* coding:utf-8 *-*
"""
Data Set Information:

Data Set Url : https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

The dataset is about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service (EMIS, [Web Link]), which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013.
Basing on the collected data five classification cases were distinguished, that depends on the forecasting period:
- 1stYear â€“ the data contains financial rates from 1st year of the forecasting period and corresponding class label that indicates bankruptcy status after 5 years. The data contains 7027 instances (financial statements), 271 represents bankrupted companies, 6756 firms that did not bankrupt in the forecasting period.
- 2ndYear â€“ the data contains financial rates from 2nd year of the forecasting period and corresponding class label that indicates bankruptcy status after 4 years. The data contains 10173 instances (financial statements), 400 represents bankrupted companies, 9773 firms that did not bankrupt in the forecasting period.
- 3rdYear â€“ the data contains financial rates from 3rd year of the forecasting period and corresponding class label that indicates bankruptcy status after 3 years. The data contains 10503 instances (financial statements), 495 represents bankrupted companies, 10008 firms that did not bankrupt in the forecasting period.
- 4thYear â€“ the data contains financial rates from 4th year of the forecasting period and corresponding class label that indicates bankruptcy status after 2 years. The data contains 9792 instances (financial statements), 515 represents bankrupted companies, 9277 firms that did not bankrupt in the forecasting period.
- 5thYear â€“ the data contains financial rates from 5th year of the forecasting period and corresponding class label that indicates bankruptcy status after 1 year. The data contains 5910 instances (financial statements), 410 represents bankrupted companies, 5500 firms that did not bankrupt in the forecasting period.



"""

from ann import *
from numpy import array
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from ga.ga import GA
from gwo.gwo import GWO
import arff
import time

print(__doc__)

timerStart = time.time()
startTime = time.strftime("%Y-%m-%d-%H-%M-%S")


dataset = arff.load(open('1year.arff', 'rb'))

dataset_values = []

bankruptcy_data = []

for item in dataset['data']:
    if int(item[-1]) == 0:
        dataset_values.append([item[x] for x in [4, 60, 10, 50, 25, 62]])
    elif int(item[-1]) == 1:
        bankruptcy_data.append([item[x] for x in [4, 60, 10, 50, 25, 62]])

k = 5

all_data_len = len(bankruptcy_data) + len(dataset_values)
print("The len of all data is: %s " % str(all_data_len))
bankruptcy_data_len = len(bankruptcy_data)
non_bankruptcy_data_len = len(dataset_values)

print("The len of bankruptcy company is : %s" % str(bankruptcy_data_len))
print("The len of non bankruptcy company is : %s" % str(non_bankruptcy_data_len))

bankruptcy_data = np.array(bankruptcy_data)

X = np.array(dataset_values)

clusterer = KMeans(n_clusters=k, random_state=10)
cluster_labels = clusterer.fit_predict(X)

clusters_data = []

cluster_centers = clusterer.cluster_centers_

for i in range(0, 5):
    clusters_data.append(X[cluster_labels == i])
    print("The Cluster%s len is :  %s" % (str(i + 1), str(len(clusters_data[i]))))

min_max_distance = []

# find threshold for each cluster
i = 0


for cluster in clusters_data:
    distances = []
    for instance in cluster:
        distances.append(euclidean_distances([instance], [cluster_centers[i]])[0][0])
    distances.sort()
    min_max_distance.append([distances[0], distances[-1]])

    i += 1


# ga = GA(5, .6, .1, bankruptcy_data, dataset_values, clusters_data, cluster_centers, min_max_distance)
gwo = GWO(5, bankruptcy_data, dataset_values, clusters_data, cluster_centers, min_max_distance)


timerEnd = time.time()
endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
executionTime = timerEnd - timerStart
print("Execution Time : %s" % (executionTime))