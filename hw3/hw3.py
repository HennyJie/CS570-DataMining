'''
@Author: your name
@Date: 2020-02-24 14:40:35
@LastEditTime: 2020-02-24 19:32:43
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


def write_to_output_file(class_labels, output_file):
    for i in range(len(class_labels)):
        with open(output_file, "w") as f:
            f.write('{} \n'.format(class_labels[i]))


# def cal_SSE():

def cal_distance(a, b):
    distance = np.sum((i - j)**2 for i, j in zip(a, b))
    return np.sqrt(distance)


def assign_points_to_cluster(dataset, k_centers):
    assignments = []
    for point in dataset:
        shortest_distance = float("inf")
        shortest_center = 0
        for i in range(len(k_centers)):
            dist = cal_distance(point, k_centers[i])
            if dist < shortest_distance:
                shortest_center = i
        assignments.append(shortest_center)
    return assignments


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
    print("dataset: ", dataset)

    for point, assignment in zip(dataset, assignments):
        clusters[assignment].append(point)
    print("clusters: ", clusters)

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

    assignments = assign_points_to_cluster(dataset, initial_k_centers)
    old_assignments = None

    k_centers = initial_k_centers

    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points_to_cluster(dataset, k_centers)
        print("assignments: ", assignments)

    end = time.time()
    write_to_output_file(assignments, output_file)

    # print("running time: ", end - start)


if __name__ == "__main__":
    assert len(
        sys.argv) == 4, "Please provide 3 parameters: dataset file, k, and output file."
    run_kmeans(sys.argv[1:])
