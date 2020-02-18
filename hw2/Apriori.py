'''
@Author: Hejie Cui
@Date: 2020-02-17 18:54:57
@LastEditTime: 2020-02-17 23:44:24
@Description: In User Settings Edit
@FilePath: /CS570-DataMining/hw2/Apriori.py
'''
import time
import sys
import numpy as np


def parse_dataset(input_dataset_name):
    dataset = []
    with open(input_dataset_name, 'r') as input_dataset_file:
        for line in input_dataset_file.readlines():
            line = line.strip('\n')
            dataset.append([int(i) for i in line.split()])
    return dataset


def write_to_output_file(output_file_name, support_count):
    with open(output_file_name, 'w') as output_file:
        items = sorted(support_count.keys())
        for item in items:
            output_file.write('{item} ({support_count})\n'.format(
                item=item, support_count=support_count[item]))


def generate_C_1(dataset):
    C_1 = set()
    for transaction in dataset:
        for item in transaction:
            item_set = frozenset([item])
            C_1.add(item_set)
    return C_1


def apriori_gen(L_ksub1, k):
    C_k = set()
    list_L_ksub1 = list(L_ksub1)
    for i in range(len(L_ksub1)):
        for j in range(1, len(L_ksub1)):
            l1 = list(list_L_ksub1[i])
            l2 = list(list_L_ksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                C_k_item = list_L_ksub1[i] | list_L_ksub1[i]
                if has_infrequent_subset(C_k_item, L_ksub1):
                    break
                else:
                    C_k.add(C_k_item)
    return C_k


def has_infrequent_subset(C_k, L_ksub1):
    for item in C_k:
        C_ksub1 = C_k - frozenset(item)
        if C_ksub1 not in L_ksub1:
            return True
    return False


def generate_L_k_from_C_k(dataset, C_k, minimum_support_count_threshold, support_count):
    L_k = set()
    item_count = {}
    for transaction in dataset:
        for item in C_k:
            if item.issubset(transaction):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    for item in item_count:
        if item_count[item] > minimum_support_count_threshold:
            L_k.add(item)
            support_count[item] = item_count[item]

    return L_k


def run_apriori(argv):
    input_dataset_name = argv[0]
    minimum_support_count_threshold = argv[1]
    output_file_name = argv[2]

    start = time.time()
    dataset = parse_dataset(input_dataset_name)

    support_count = {}
    C_1 = generate_C_1(dataset)
    L_1 = generate_L_k_from_C_k(
        dataset, C_1, minimum_support_count_threshold, support_count)
    L_ksub1 = L_1.copy()

    current_L = L_1
    frequent_itemsets = []
    frequent_itemsets.append(L_ksub1)

    k = 2
    while(current_L != set()):
        C_i = apriori_gen(L_ksub1, k)
        L_i = generate_L_k_from_C_k(
            dataset, C_i, minimum_support_count_threshold, support_count)
        L_ksub1 = L_i.copy()
        frequent_itemsets.append(L_ksub1)
        k = k + 1
        current_L = L_i

    write_to_output_file(output_file_name, support_count)

    end = time.time()
    print('running time: ', end-start)


if __name__ == "__main__":
    assert len(sys.argv) == 4, "Please provide 3 parameters: input dataset file name, minimum support count, and output file name."
    run_apriori(sys.argv[1:])
