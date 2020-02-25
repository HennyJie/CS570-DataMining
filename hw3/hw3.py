'''
@Author: your name
@Date: 2020-02-24 14:40:35
@LastEditTime: 2020-02-24 23:02:28
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /CS570-DataMining/hw3/hw3.py
'''
import sys
import random
import numpy as np
import time
from collections import defaultdict


def parse_dataset(input_dasaset):
    dataset = []
    with open(input_dasaset, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            numerical_attributes = []
            all_attributes = [i for i in line.split(",")]

            for i in range(len(all_attributes)-1):
                numerical_attributes.append(float(all_attributes[i]))

            if numerical_attributes != []:
                dataset.append(numerical_attributes)

    return dataset


def write_to_output_file(assignments, sum_squared_error, silhouette_coefficient, output_file):
    with open(output_file, "w") as f:
        for i in range(len(assignments)):
            f.write('{} \n'.format(assignments[i]))
        f.write('sum squared error: {}, silhouette_coefficient: {} \n'.format(
            sum_squared_error, silhouette_coefficient))


def cal_silhouette_coefficient(dataset, assignments):
    final_clusters = defaultdict(list)
    for point, assignment in zip(dataset, assignments):
        final_clusters[assignment].append(point)
    a_o_list = []
    b_o_list = []
    centers = final_clusters.keys()
    for center in centers:
        intra_dist_sum = 0
        inter_dist_sum = 0
        for o in final_clusters[center]:
            intra_dist_sum = np.sum(cal_distance(o, o_prime)
                                    for o_prime in final_clusters[center])
            a_o = intra_dist_sum / float(len(final_clusters[center])-1)
            a_o_list.append(a_o)

            inter_dist_list = []
            for another_center in centers:
                if another_center != center:
                    inter_dist_sum = np.sum(cal_distance(o, o_prime)
                                            for o_prime in final_clusters[another_center])
                    num_points_in_another_center_cluster = len(
                        final_clusters[another_center])
                    inter_dist_avg = inter_dist_sum / num_points_in_another_center_cluster
                    inter_dist_list.append(inter_dist_avg)
                    b_o = min(inter_dist_list)
                    b_o_list.append(b_o)

    s_o_list = []
    for a_o, b_o in zip(a_o_list, b_o_list):
        s_o = (b_o - a_o) / max(a_o, b_o)
        s_o_list.append(s_o)

    silhouette_coefficient = sum(s_o_list) / len(s_o_list)
    return silhouette_coefficient


def cal_distance(a, b):
    dist_sum = 0
    for i, j in zip(a, b):
        dist_sum += (i - j)**2
    return dist_sum


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
        sum_squared_error += shortest_distance
    return assignments, sum_squared_error


def points_average_centers(points):
    new_center = []

    for i in range(len(points[0])):
        attribute_sum = 0
        for p in points:
            attribute_sum += p[i]

        attribute_avg = attribute_sum / float(len(points))
        new_center.append(attribute_avg)

    return new_center


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


def run_kmeans(argv):
    input_dasaset = argv[0]
    k = int(argv[1])
    output_file = argv[2]

    start = time.time()
    dataset = parse_dataset(input_dasaset)

    initial_k_centers = random.sample(dataset, k)
    print("initial_k_centers: ", initial_k_centers)

    assignments, sum_squared_error = assign_points_to_cluster(
        dataset, initial_k_centers)
    # print("assignments: ", assignments)

    old_assignments = None

    k_centers = initial_k_centers
    sum_squared_error = 0
    silhouette_coefficient = 0

    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments, sum_squared_error = assign_points_to_cluster(
            dataset, k_centers)
        print("assignments: ", assignments)

    end = time.time()
    silhouette_coefficient = cal_silhouette_coefficient(dataset, assignments)
    write_to_output_file(assignments, sum_squared_error,
                         silhouette_coefficient, output_file)

    print("running time: ", end - start)


if __name__ == "__main__":
    assert len(
        sys.argv) == 4, "Please provide 3 parameters: dataset file, k, and output file."
    run_kmeans(sys.argv[1:])
