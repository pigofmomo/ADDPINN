"""Graph optimization helpers for balanced and loss-aware subdomain division.
用于均衡与损失感知子区域划分的图优化工具。
"""

import networkx as nx
import copy
import numpy as np

# Identify nodes adjacent to another subdomain. / 判断哪些节点位于子区域边界。
def find_boundary_nodes(network, polygon_labels):
    polygon_num = len(polygon_labels)
    is_boundary_nodes = [False for i in range(polygon_num)]
    for node_idx in range(polygon_num):
        for neighbor in network.neighbors(node_idx):
            if polygon_labels[neighbor] != polygon_labels[node_idx]:  # Neighbor has a different label. / 邻居标签不同。
                is_boundary_nodes[node_idx] = True
                break
    return is_boundary_nodes

def labels_to_subgraph(polygon_labels, label_num):
    subgraphs = [set() for i in range(label_num)]
    for node_idx in range(polygon_labels):
        label = polygon_labels[node_idx]
        subgraphs[label].add(node_idx)

    return subgraphs

def check_connectivity(subgraphs, network):
    for graph in subgraphs:
        if not nx.is_connected(network.subgraph(graph)):
            return False

    return True

# Sum weights over one subgraph. / 计算子图的权重和。
def compute_subgraph_weights(subgraph, polygon_weights):
    sum_of_weights = 0.0
    for node in subgraph:
        sum_of_weights += polygon_weights[node][0]
    return sum_of_weights

def compute_subgraph_weights_labels(subgraph, polygon_weights, polygon_labels):
    sum_of_weights = 0.0
    for node in subgraph:
        label = polygon_labels[node]
        sum_of_weights += polygon_weights[node][label]
    return sum_of_weights

# Compute the subdomain balance penalty. / 计算子区域均衡误差。
def compute_balance_loss(subgraphs, polygon_weights):
    cost = []
    for i in range(len(subgraphs)):
        cost.append(compute_subgraph_weights(subgraphs[i], polygon_weights))

    avg_cost = np.mean(cost)
    balance_loss = np.sum(np.abs(np.array(cost) - avg_cost))
    return balance_loss

def compute_boundary_length(is_boundary_nodes):
    regularize_coeff = 0.001
    return regularize_coeff * sum(is_boundary_nodes)

def compute_min_loss(subgraphs, polygon_weights, polygon_labels):
    cost = []
    for i in range(len(subgraphs)):
        cost.append(compute_subgraph_weights_labels(subgraphs[i], polygon_weights, polygon_labels))

    total_cost = np.sum(cost)
    return total_cost


def search_new_graph(labels, searched_dict):

    pass

def divide_graph_balance(count_subdomain, polygon_weights, polygon_network, polygon_labels):

    # 把权重归一化到和为1
    polygon_weights = polygon_weights / np.sum(polygon_weights)

    subgraphs = labels_to_subgraph(polygon_labels, count_subdomain)
    is_boundary_nodes = find_boundary_nodes(polygon_network, polygon_labels)

    searched_dict = {}
    to_search_set = set()
    changed = True
    max_iter_times = 10
    top_k = 5
    num_of_new_graph = 100
    iter_times = 0
    while changed and iter_times < max_iter_times:
        changed = False

        # 把待搜索的搜索一遍
        for to_search_label in to_search_set:
            subgraphs = labels_to_subgraph(to_search_label, count_subdomain)
            connectivity = check_connectivity(subgraphs, polygon_weights)
            if not connectivity:
                searched_dict[to_search_label] = np.inf
            balance_loss = compute_balance_loss(subgraphs, polygon_weights)
            is_boundary_nodes = find_boundary_nodes(polygon_network, to_search_label)
            smooth_loss = compute_boundary_length(is_boundary_nodes)
            object_func_value = balance_loss + smooth_loss
            searched_dict[to_search_label] = object_func_value

        best_labels = [key for key, value in sorted(searched_dict.items(), key=lambda item: item[1])][:top_k]
        for label in best_labels:
            to_search_set.add(search_new_graph(label, searched_dict))
            is_boundary_nodes = find_boundary_nodes(polygon_network, label)





        iter_times += 1
        if iter_times >= max_iter_times:
            print("max iteration times reached")

    return polygon_labels

def divide_graph_min(count_subdomain, polygon_weights, polygon_network, polygon_labels):
    pass