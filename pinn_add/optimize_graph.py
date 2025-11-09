import networkx as nx
import copy
import numpy as np

# 判断哪些是边界节点
def find_boundary_nodes(network, polygon_labels):
    polygon_num = len(polygon_labels)
    is_boundary_nodes = [False for i in range(polygon_num)]
    for node_idx in range(polygon_num):
        for neighbor in network.neighbors(node_idx):
            if polygon_labels[neighbor] != polygon_labels[node_idx]: # 邻居的Label不同的
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

# 计算子图的权重和
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

# 计算平衡误差
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


# 计算平滑正则
# 1. 先把边界取出来，计算坐标差分。但是有些是树状的不是线状边界
# 2. 计算边界长度，也就是边界节点的数目
# 3. 计算边界节点周围的其他标签的节点数
# 4.
# def compute_boundary_smooth():
#     boundary_nodes = set()
#     boundary_subgraph = graph.subgraph(boundary_nodes)
#     pass

# 优化方式：但是得记录最优值
# 1. 贪婪优化，多次随机
# 2. 遗传算法？

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