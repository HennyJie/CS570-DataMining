'''
@Author: Hejie Cui
@Date: 2020-02-17 18:54:57
@LastEditTime: 2020-02-19 16:32:24
@FilePath: /CS570-DataMining/hw2/Apriori.py
'''
import time
import sys
import itertools
from functools import cmp_to_key


# read the input dataset to a two-dimentional list
def parse_dataset(input_dataset_name):
    dataset = []
    with open(input_dataset_name, 'r') as input_dataset_file:
        for line in input_dataset_file.readlines():
            line = line.strip('\n')
            dataset.append([int(i) for i in line.split()])
    return dataset


# get the int item values from each transaction
def get_values(itemset):
    list_of_item_value = [int(x) for x in itemset]
    return list_of_item_value


# comparation function rank two frequent itemsets
def compare(a, b):
    a = get_values(a)
    a_len = len(a)
    b = get_values(b)
    b_len = len(b)

    if a_len == b_len:
        for i in range(a_len):
            if not a[i] == b[i]:
                return a[i] - b[i]
    else:
        for i in range(max(a_len, b_len)):
            if i >= a_len:
                return -1
            elif i >= b_len:
                return 1

            if not a[i] == b[i]:
                return a[i] - b[i]


# write the frequent_itemset_dict into file, as the same order as in "example-output.txt"
def write_to_output_file(output_file_name, frequent_itemset_dict):
    output = {}
    sorted_output_keys = sorted(
        frequent_itemset_dict.keys(), key=cmp_to_key(compare))
    sorted_output_keys_string = []

    for sorted_item_key in sorted_output_keys:
        sorted_item_key = [str(i) for i in sorted_item_key]
        sorted_item_key = ' '.join(sorted_item_key)
        sorted_output_keys_string.append(sorted_item_key)

    for frequent_itemset, support in frequent_itemset_dict.items():
        frequent_itemset = [str(i) for i in frequent_itemset]
        frequent_itemset_string = ' '.join(frequent_itemset)
        output[frequent_itemset_string] = support

    with open(output_file_name, 'w') as output_file:
        for item in sorted_output_keys_string:
            output_file.write(f'{item} ({output[item]})\n')


# generate C_1 and L_1
def generate_L_1(dataset, minimum_support_count_threshold, frequent_items_count):
    C_1 = {}
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if dataset[i][j] not in C_1:
                C_1[dataset[i][j]] = 1
            else:
                C_1[dataset[i][j]] += 1
    # print("len C_1: ", len(C_1))

    L_1 = []
    for item in C_1:
        if C_1[item] >= minimum_support_count_threshold:
            L_1.append(item)
            frequent_items_count[(item,)] = C_1[item]

    return L_1


# generate C_2 and L_2
def generate_L_2_with_hashtable(dataset, minimum_support_count_threshold, frequent_items_count):
    hash_table = {}
    for transaction in dataset:
        for itemset in sorted(itertools.combinations(transaction, 2)):
            if itemset not in hash_table:
                hash_table[itemset] = 1
            else:
                hash_table[itemset] += 1
    # print("len C_2: ", len(hash_table))

    L_2 = []
    for itemset in hash_table:
        if hash_table[itemset] >= minimum_support_count_threshold:
            L_2.append(itemset)
            frequent_items_count[tuple(itemset)] = hash_table[itemset]
    return L_2


# generate all the subset from a given set
def subset_generation(original_set, subset_size):
    return map(list, set(itertools.combinations(original_set, subset_size)))


# generate C_L, including joint and prune
def apriori_gen(L_ksub1, k):
    # joint process
    C_k = []
    for i in range(len(L_ksub1)):
        for j in range(i+1, len(L_ksub1)):
            l1 = list(L_ksub1[i])
            l2 = list(L_ksub1[j])
            if l1[:k-2] == l2[:k-2]:
                C_k.append(sorted(list(set(L_ksub1[i]) | set(L_ksub1[j]))))

    # prune process
    pruned_C_k = []
    L_ksub1 = set(L_ksub1)
    for itemset in C_k:
        all_subsets = list(subset_generation(set(itemset), k - 1))
        satisfied = True
        for i in range(len(all_subsets)):
            subset = sorted(all_subsets[i])
            if tuple(subset) not in L_ksub1:
                satisfied = False
                break
        if satisfied == True:
            pruned_C_k.append(tuple(itemset))

    # print("len C_{}: {}".format(k, len(pruned_C_k)))
    return pruned_C_k


# generate L_k from C_k
def generate_L_k_from_C_k(dataset, C_k, minimum_support_count_threshold, frequent_items_count):
    L_k = []
    itemset_count = {}
    C_k = [set(itemset) for itemset in C_k]

    for transaction in dataset:
        for itemset in C_k:
            if itemset.issubset(transaction):
                itemset = tuple(sorted(tuple(itemset)))
                if itemset not in itemset_count:
                    itemset_count[itemset] = 1
                else:
                    itemset_count[itemset] += 1

    for itemset in itemset_count:
        if itemset_count[itemset] >= minimum_support_count_threshold:
            L_k.append(itemset)
            frequent_items_count[itemset] = itemset_count[itemset]
    return L_k


# the main apriori algorithm for generate C_1 to C_k, L_1 to L_k and update frequent_items_count
def run_apriori(argv):
    input_dataset_name = argv[0]
    minimum_support_count_threshold = int(argv[1])
    output_file_name = argv[2]

    start = time.time()
    dataset = parse_dataset(input_dataset_name)

    L = []
    frequent_items_count = {}
    L_1 = generate_L_1(
        dataset, minimum_support_count_threshold, frequent_items_count)
    L.append(L_1)
    # print("len L_1: ", len(L_1))

    L_2 = generate_L_2_with_hashtable(
        dataset, minimum_support_count_threshold, frequent_items_count)
    L.append(L_2)
    # print("len L_2: ", len(L_2))

    current_L = sorted(L_2)
    k = 3
    dataset = [set(transaction) for transaction in dataset]
    while(len(current_L) > 0):
        C_k = apriori_gen(current_L, k)
        L_k = generate_L_k_from_C_k(
            dataset, C_k, minimum_support_count_threshold, frequent_items_count)
        # print("len L_{}: {}".format(k, len(L_k)))
        L.append(L_k)
        k = k + 1
        current_L = sorted(L_k)

    write_to_output_file(output_file_name, frequent_items_count)

    end = time.time()
    print('running time: {}s'.format(end-start))


if __name__ == "__main__":
    assert len(sys.argv) == 4, "Please provide 3 parameters: input dataset file name, minimum support count, and output file name."
    run_apriori(sys.argv[1:])
