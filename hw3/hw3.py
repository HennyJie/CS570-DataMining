'''
@Author: Hejie Cui
@Date: 2020-02-24 14:40:35
@LastEditTime: 2020-02-26 22:46:15
@FilePath: /CS570-DataMining/hw3/hw3.py
'''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import random
import numpy as np
import time
from collections import defaultdict
import math


# funnction for filtering out only numerical type attributes
def is_numerical(all_attributes):
    try:
        float(all_attributes)
        return True
    except ValueError:
        return False


# parse the input data file
def parse_dataset(input_dasaset):
    dataset = []
    with open(input_dasaset, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            numerical_attributes = []
            all_attributes = line.split(",")
            numerical_attributes = filter(is_numerical, all_attributes)
            numerical_attributes = [float(f) for f in numerical_attributes]
            if numerical_attributes != []:
                dataset.append(numerical_attributes)

    return dataset


# write to the output file in the required format
def write_to_output_file(assignments, sum_squared_error, silhouette_coefficient, output_file):
    print('sum squared error: {}, silhouette_coefficient: {} \n'.format(
        sum_squared_error, silhouette_coefficient))

    with open(output_file, "w") as f:
        for i in range(len(assignments)):
            f.write('{} \n'.format(assignments[i]))
        f.write('sum squared error: {}, silhouette_coefficient: {} \n'.format(
            sum_squared_error, silhouette_coefficient))


# max min normalization
def max_min_normalization(X):
    X_arr = np.array(X)
    x_min = np.min(X_arr, axis=0)
    x_max = np.max(X_arr, axis=0)
    m = X_arr.shape[1]
    for i in range(m):
        X_arr[:, i] = (X_arr[:, i] - x_min[i]) / (x_max[i] - x_min[i])
    X = X_arr.tolist()
    return X


# z score normalization
def z_score(X):
    X_arr = np.array(X)
    x_mu = np.average(X_arr, axis=0)
    x_sigma = np.std(X_arr, axis=0)
    m = X_arr.shape[1]
    for i in range(m):
        X_arr[:, i] = (X_arr[:, i] - x_mu[i]) / x_sigma[i]
    X = X_arr.tolist()
    return X


# calculate the final silhouette coefficient
def cal_silhouette_coefficient(dataset, assignments):
    final_clusters = defaultdict(list)
    for point, assignment in zip(dataset, assignments):
        final_clusters[assignment].append(point)

    s_o_list = []
    centers = final_clusters.keys()
    for center in centers:
        intra_dist_sum = 0
        inter_dist_sum = 0
        for o in final_clusters[center]:
            intra_dist_sum = sum(cal_distance(o, o_prime)
                                 for o_prime in final_clusters[center])
            a_o = intra_dist_sum / float(len(final_clusters[center])-1)

            inter_dist_list = []
            for another_center in centers:
                if another_center != center:
                    inter_dist_sum = sum(cal_distance(o, o_prime)
                                         for o_prime in final_clusters[another_center])
                    num_points_in_another_center_cluster = float(len(
                        final_clusters[another_center]))
                    inter_dist_avg = inter_dist_sum / num_points_in_another_center_cluster
                    inter_dist_list.append(inter_dist_avg)

            b_o = min(inter_dist_list)

            s_o = (b_o - a_o) / max(a_o, b_o)
            s_o_list.append(s_o)

    silhouette_coefficient = sum(s_o_list) / len(s_o_list)

    return silhouette_coefficient


# calculate the distance between two points
def cal_distance(a, b):
    dist_sum = 0
    for i, j in zip(a, b):
        dist_sum += (i - j)**2
    return math.sqrt(dist_sum)


# assign each point in the dataset to its nearest cluster
def assign_points_to_cluster(dataset, k_centers):
    assignments = []
    sum_squared_error = 0

    for point in dataset:
        shortest_distance = float("inf")
        shortest_center = 0
        for i in range(len(k_centers)):
            dist = cal_distance(point, k_centers[i])
            if dist < shortest_distance:
                shortest_center = i
                shortest_distance = dist

        assignments.append(shortest_center)
        sum_squared_error += shortest_distance**2
    return assignments, sum_squared_error


# calcualte the center of a clusters, given the points in this cluster
def points_average_centers(points):
    new_center = []

    for i in range(len(points[0])):
        attribute_sum = 0
        for p in points:
            attribute_sum += p[i]

        attribute_avg = attribute_sum / float(len(points))
        new_center.append(attribute_avg)

    return new_center


# update the center using the new assignments
def update_centers(dataset, assignments):
    new_centers = []
    clusters = defaultdict(list)
    # print("dataset: ", dataset)

    for point, assignment in zip(dataset, assignments):
        clusters[assignment].append(point)
    # print("clusters: ", clusters)

    for center in clusters:
        new_centers.append(points_average_centers(clusters[center]))
    return new_centers


# the main function for running kmeans
def run_kmeans(argv):
    input_dasaset = argv[0]
    k = int(argv[1])
    output_file = argv[2]

    start = time.time()
    dataset = parse_dataset(input_dasaset)

    # using different normalization methods
    # dataset = max_min_normalization(dataset)
    # dataset = z_score(dataset)

    initial_k_centers = random.sample(dataset, k)
    # print("initial_k_centers: ", initial_k_centers)
    sum_squared_error = 0
    silhouette_coefficient = 0

    old_assignments = None
    assignments, sum_squared_error = assign_points_to_cluster(
        dataset, initial_k_centers)
    # print("assignments: ", assignments)

    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments, sum_squared_error = assign_points_to_cluster(
            dataset, new_centers)
    # print("assignments: ", assignments)

    end = time.time()
    silhouette_coefficient = cal_silhouette_coefficient(dataset, assignments)
    write_to_output_file(assignments, sum_squared_error,
                         silhouette_coefficient, output_file)

    print("running time: ", end - start)

    # compare with sklearn
    sklearn_kmeans = KMeans(n_clusters=k).fit(dataset)
    labels = sklearn_kmeans.labels_
    print(labels)
    my_silhouette_coefficient = cal_silhouette_coefficient(dataset, labels)
    print("my silhouette_coefficient: ", my_silhouette_coefficient)

    print(sklearn_kmeans.inertia_, silhouette_score(
        dataset, labels, metric='euclidean'))


if __name__ == "__main__":
    assert len(
        sys.argv) == 4, "Please provide 3 parameters: dataset file, k, and output file."
    run_kmeans(sys.argv[1:])
