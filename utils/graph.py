import torch
import torch.nn as nn
import networkx as nx
from graphviz import Digraph


node_attr = dict(style='filled', shape='box',
                 align='left', fontsize='12',
                 ranksep='0.1', height='0.2')
graph_attr = dict()


class Graph(object):
    def __init__(self, start_node, name):
        self.name = name
        self.node_dict = {}
        self.start_node = start_node
        self.add_node(start_node)
        return

    def add_node(self, node):
        self.node_dict[node.module_name] = node
        return

    def convert_to_networkx(self, G=None, head=None,
                            node_memo=None,
                            edge_memo=None):
        # Initialize networkx graph
        if G is None:
            G = nx.Graph()
            node_memo, edge_memo = set(), set()
            head = self.start_node
            G.add_node(str(head))
            node_memo.add(str(head))

        # Recursively add nodes
        for child in head.next_nodes:
            curr_node = str(child)
            curr_edge = (str(head), str(child))

            if curr_node not in node_memo:
                G.add_node(curr_node)
                node_memo.add(curr_node)

            if curr_edge not in edge_memo:
                G.add_edge(*curr_edge)
                edge_memo.add(curr_edge)

            G = self.convert_to_networkx(G, child,
                    node_memo, edge_memo)
        return G

    def convert_to_graphviz(self, G=None, head=None,
                            node_memo=None,
                            edge_memo=None):

        # Initialize graphviz Digraph
        if G is None:
            G = Digraph(self.name, format='png', node_attr=node_attr,
                        graph_attr=graph_attr)
            node_memo, edge_memo = set(), set()
            head = self.start_node
            G.node(str(head))
            node_memo.add(str(head))

        # Recursively add nodes
        for child in head.next_nodes:
            curr_node = str(child)
            curr_edge = (str(head), str(child))

            if curr_node not in node_memo:
                G.node(curr_node)
                node_memo.add(curr_node)

            if curr_edge not in edge_memo:
                G.edge(*curr_edge)
                edge_memo.add(curr_edge)

            G = self.convert_to_graphviz(G, child,
                    node_memo, edge_memo)
        return G


class Node(object):
    def __init__(self, module_name, module_type, info={}, graph=None):
        self.module_name = module_name
        self.module_type = module_type
        self.prev_nodes = []
        self.next_nodes = []
        self.info = info

        # Add this node to the given graph
        if graph is not None:
            graph.add_node(self)
        return

    def add_link_to(self, node):
        self.next_nodes.append(node)
        node.prev_nodes.append(self)
        return

    def __str__(self):
        string = 'type-> {}\nname-> {}'.format(self.module_type.upper(), self.module_name)
        return string


class AddNode(Node):

    counter = 0

    def __init__(self, graph):
        super(AddNode, self).__init__(str(AddNode.counter), 'add', graph=graph)
        AddNode.counter += 1
        return


class ConcatNode(Node):

    counter = 0

    def __init__(self, graph):
        super(ConcatNode, self).__init__(str(ConcatNode.counter), 'concat', graph=graph)
        ConcatNode.counter += 1
        return


def get_module_info(module):

    if isinstance(module, nn.Conv2d):
        info = {'inp': module.in_channels, 'oup': module.out_channels,
                'stride': module.stride, 'groups': module.groups,
                'kernel_size': module.kernel_size}

    elif isinstance(module, nn.BatchNorm2d):
        info = {'num_features': module.num_features}

    elif isinstance(module, nn.Linear):
        info = {'in_features': module.in_features,
                'out_features': module.out_features}

    else:
        raise ValueError('Unrecognized module: {}'.format(module))
    return info


class ClusterElement(object):
    def __init__(self, node):
        self.node = node
        self.parant = None
        return


def merge_clusters(element1, element2):
    # Find their heads
    head1 = element1
    while head1.parant is not None:
        head1 = head1.parant

    head2 = element2
    while head2.parant is not None:
        head2 = head2.parant

    # Merge the two clusters if they are not the same
    if head1 != head2:
        head2.parant = head1
    return


def handle_bn_node(bn_node, clusters):
    assert len(bn_node.prev_nodes) == 1, 'Handle only one prev_node for bn_node'
    prev_node = bn_node.prev_nodes[0]

    if prev_node.module_type == 'conv':
        merge_clusters(clusters[bn_node.module_name], clusters[prev_node.module_name])
    else:
        raise NotImplementedError('Not support {} node before bn_node'.format(prev_node.module_type))
    return


def handle_dw_conv_node(dw_conv_node, clusters):
    assert len(dw_conv_node.prev_nodes) == 1, 'Handle only one prev_node for dw_conv_node'
    prev_node = dw_conv_node.prev_nodes[0]

    if prev_node.module_type in ['conv', 'bn']:
        merge_clusters(clusters[dw_conv_node.module_name], clusters[prev_node.module_name])
    else:
        raise NotImplementedError('Not support {} node before dw_conv_node'.format(prev_node.module_type))
    return


def handle_shortcut_add_node(add_node, clusters, add_root=None):
    for prev_node in add_node.prev_nodes:
        if prev_node.module_type in ['conv', 'bn']:
            if add_root is None:
                add_root = prev_node
            else:
                merge_clusters(clusters[add_root.module_name], clusters[prev_node.module_name])
        elif prev_node.module_type == 'add':
            add_root = handle_shortcut_add_node(prev_node, clusters, add_root=add_root)
        else:
            raise NotImplementedError('Not support {} node before shorcut_add_node'.format(
                prev_node.module_type))
    return add_root


def handle_node_constraints(node, clusters):
    if node.module_type == 'bn':
        handle_bn_node(node, clusters)

    elif node.module_type == 'conv' and node.info['groups'] != 1:
        assert node.info['groups'] == node.info['inp']  # Consider depthwise conv only
        handle_dw_conv_node(node, clusters)

    elif node.module_type == 'add':
        handle_shortcut_add_node(node, clusters)

    elif node.module_type == 'concat':
        raise NotImplementedError('Concat node is not implemented yet!')

    elif node.module_type == 'conv' and node.info['groups'] == 1:
        pass  # No constraints for normal convolution layers

    elif node.module_type == 'fc':
        pass  # No constraints for normal fully connected layers

    else:
        print(node.module_type)
        raise NotImplementedError('Unexpected errors occur')

    return


def parse_clusters(clusters):
    cluster_dict = {}

    for _, cluster_element in clusters.items():
        head = cluster_element
        while head.parant is not None:
            head = head.parant

        if head not in cluster_dict:
            cluster_dict[head] = []
        cluster_dict[head].append(cluster_element)
    return {str(h.node): [e.node for e in c] for h, c in cluster_dict.items()}


def show_clusters(clusters):
    cluster_dict = parse_clusters(clusters)

    counter = 0
    for cluster_name, cluster_nodes in cluster_dict.items():
        print('=== Cluster {} ==='.format(counter))
        print(cluster_name)
        print('------------------')
        for node in cluster_nodes:
            print(node)
        print('==================')
        counter += 1
    return


def investigate_constraints(graph, show=False):

    # Each node becomes a cluter itself (except for AddNode and ConcatNode)
    clusters = {name: ClusterElement(node) for name, node in graph.node_dict.items()
                if not isinstance(node, AddNode) and not isinstance(node, ConcatNode)}

    # Iterate through all nodes and combine the constraints
    for _, node in graph.node_dict.items():
        handle_node_constraints(node, clusters)

    if show:
        show_clusters(clusters)
    return parse_clusters(clusters)
