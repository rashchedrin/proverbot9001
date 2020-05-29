import pytest
import random
from collections import defaultdict
from tree_traverses.algorithms import *
from tree_traverses.interfaces import *
from typing import (List, Optional, Dict, Any)

OUTPUT = ""


def lprint(s):
    global OUTPUT
    OUTPUT += s + '\n'
    # print(s)


class Tree(GraphInterface):
    def __init__(self, size=1, parents=None, children=None):
        self._n_nodes = size
        self._parents: Dict[int, Optional[int]] = parents if parents else {0: None}
        self._children: Dict[int, List[int]] = children if children else defaultdict(list)

    def add_child(self, parent):
        new_node_id = self._n_nodes
        self._n_nodes += 1
        self._parents[new_node_id] = parent
        self._children[parent].append(new_node_id)

    def children(self, node_id):
        return self._children[node_id]

    def parent(self, node_id):
        return self._parents[node_id]

    def root(self):
        return 0

    def size(self):
        return self._n_nodes

    def to_str_from(self, node_id):
        if len(self._children[node_id]) == 0:
            return f"{node_id}"
        s = f"{node_id}=("
        s += ", ".join([self.to_str_from(child) for child in self._children[node_id]])
        return s + ")"

    def __str__(self):
        return self.to_str_from(0)

    def __repr__(self):
        return f"Tree(size={self._n_nodes}, parents={self._parents}, children={self._children})"

    def edge_destination(self, edge):
        frm, to = edge
        lprint(f"Called destination of {edge}")
        return to

    def get_outgoing_edges(self, node) -> List:
        children = self.children(node)
        edges = list(map(lambda child: (node, child), children))
        lprint(f"Edges of {node} are {edges}")
        return edges


def mk_random_tree(n_nodes):
    tree = Tree()
    for _ in range(n_nodes):
        parent = random.randint(0, tree.size() - 1)
        tree.add_child(parent)
    return tree


class EventLoggingVisitor(TreeTraverseVisitor):
    def __init__(self):
        self._log = []

    def on_enter(self, graph: GraphInterface, entered_node) -> TraverseVisitorResult:
        self._log += [("on_enter", entered_node)]
        return super().on_enter(graph, entered_node)

    def on_traveling_edge(self, graph: GraphInterface, frm, edge) -> TraverseVisitorResult:
        self._log += [("on_traveling_edge", frm, edge)]
        return super().on_traveling_edge(graph, frm, edge)

    def on_discover(self, graph: GraphInterface, frm, discovered) -> TraverseVisitorResult:
        self._log += [("on_discover", frm, discovered)]
        return super().on_discover(graph, frm, discovered)

    def on_got_result(self, graph: GraphInterface, receiver_node, sender_node, result,
                      siblings_results) -> TraverseVisitorResult:
        self._log += [("on_got_result", receiver_node, sender_node, result,
                       siblings_results)]
        return super().on_got_result(graph, receiver_node, sender_node, result, siblings_results)

    def on_exit(self, graph: GraphInterface, node_left, all_children_results, stage,
                suggested_result) -> TraverseVisitorResult:
        self._log += [("on_exit", node_left, all_children_results, stage,
                       suggested_result)]
        return super().on_exit(graph, node_left, all_children_results, stage, suggested_result)

    def log(self):
        return self._log

def assert_lists_eq(first, second):

    all_ok = first == second
    if not all_ok:
        for i, (a, b) in enumerate(zip(first, second)):
            if a == b:
                print(f"OK {i}: {a}")
            else:
                print(f"FAIL {i}: {a} != {b}")
                all_ok = False
    assert all_ok


def check_equivalence(tree, impl_first, impl_second,
                      visitor_maker):
    visitor_first = visitor_maker()
    visitor_second = visitor_maker()
    res_first = impl_first(tree.root(), tree, visitor_first)
    res_second = impl_second(tree.root(), tree, visitor_second)
    assert res_first == res_second
    assert_lists_eq(visitor_first.log(), visitor_second.log())


def test_dfs_and_stack_dfs_equivalence():
    for size in range(20):
        print(size)
        for attempt in range(10):
            print(".", end='')
            tree = mk_random_tree(size)
            check_equivalence(tree, impl_first=dfs, impl_second=dfs_explicit,
                              visitor_maker=EventLoggingVisitor)
            # todo: check equivalence for flow controll

