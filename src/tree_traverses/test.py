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
        self._log = []

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
        self._log.append(f"Called destination of {edge}")
        return to

    def get_outgoing_edges(self, node) -> List:
        children = self.children(node)
        edges = list(map(lambda child: (node, child), children))
        self._log.append(f"Edges of {node} are {edges}")
        return edges

    def log(self):
        return self._log

    def clear_log(self):
        self._log = []

def mk_random_tree(n_nodes):
    tree = Tree()
    for _ in range(n_nodes):
        parent = random.randint(0, tree.size() - 1)
        tree.add_child(parent)
    return tree


class EventLoggingVisitor(TreeTraverseVisitor):
    def __init__(self, do_print=False):
        self._log = []
        self.print = print if do_print else lambda x: None

    def on_enter(self, graph: GraphInterface, entered_node) -> TraverseVisitorResult:
        logged = ("on_enter", entered_node)
        self.print(logged)
        self._log.append(logged)
        return super().on_enter(graph, entered_node)

    def on_traveling_edge(self, graph: GraphInterface, frm, edge) -> TraverseVisitorResult:
        logged = ("on_traveling_edge", frm, edge)
        self.print(logged)
        self._log.append(logged)
        return super().on_traveling_edge(graph, frm, edge)

    def on_discover(self, graph: GraphInterface, frm, discovered) -> TraverseVisitorResult:
        logged = ("on_discover", frm, discovered)
        self.print(logged)
        self._log.append(logged)
        return super().on_discover(graph, frm, discovered)

    def on_got_result(self, graph: GraphInterface, receiver_node, sender_node, result,
                      siblings_results) -> TraverseVisitorResult:
        logged = ("on_got_result", receiver_node, sender_node, result,
                  siblings_results.copy())
        self.print(logged)
        self._log.append(logged)
        return super().on_got_result(graph, receiver_node, sender_node, result, siblings_results)

    def on_exit(self, graph: GraphInterface, node_left, all_children_results, stage,
                suggested_result) -> TraverseVisitorResult:
        logged = ("on_exit", node_left, all_children_results.copy(), stage,
                  suggested_result)
        self.print(logged)
        self._log.append(logged)
        return super().on_exit(graph, node_left, all_children_results, stage, suggested_result)

    def log(self):
        return self._log


def hash_bit(seed: str, divisor=5):
    return hash(seed) % divisor == 0


def hashrandom_visitor_result(seed: str):
    do_return = hash_bit("0|" + seed)
    do_skip = hash_bit("1|" + seed)
    do_break = hash_bit("2|" + seed)
    what_return = seed + str((do_return, do_skip, do_break))
    return TraverseVisitorResult(do_return=do_return, what_return=what_return,
                                 do_skip=do_skip, do_break=do_break)


class LoggingDroppingVisitor(TreeTraverseVisitor):
    def __init__(self, do_print: bool = False, seed: str = "deadbeef"):
        self._log = []
        self._print = print if do_print else lambda x: None
        self._seed = seed

    def on_enter(self, graph: GraphInterface, entered_node) -> TraverseVisitorResult:
        logged = ("on_enter", entered_node)
        self._print(logged)
        self._log.append(logged)
        super().on_enter(graph, entered_node)
        return hashrandom_visitor_result(str(logged) + str(self._seed))

    def on_traveling_edge(self, graph: GraphInterface, frm, edge) -> TraverseVisitorResult:
        logged = ("on_traveling_edge", frm, edge)
        self._print(logged)
        self._log.append(logged)
        super().on_traveling_edge(graph, frm, edge)
        return hashrandom_visitor_result(str(logged) + str(self._seed))

    def on_discover(self, graph: GraphInterface, frm, discovered) -> TraverseVisitorResult:
        logged = ("on_discover", frm, discovered)
        self._print(logged)
        self._log.append(logged)
        super().on_discover(graph, frm, discovered)
        return hashrandom_visitor_result(str(logged) + str(self._seed))

    def on_got_result(self, graph: GraphInterface, receiver_node, sender_node, result,
                      siblings_results) -> TraverseVisitorResult:
        logged = ("on_got_result", receiver_node, sender_node, result,
                  siblings_results.copy())
        self._print(logged)
        self._log.append(logged)
        super().on_got_result(graph, receiver_node, sender_node, result, siblings_results)
        return hashrandom_visitor_result(str(logged) + str(self._seed))

    def on_exit(self, graph: GraphInterface, node_left, all_children_results, stage,
                suggested_result) -> TraverseVisitorResult:
        logged = ("on_exit", node_left, all_children_results.copy(), stage,
                  suggested_result)
        self._print(logged)
        self._log.append(logged)
        super().on_exit(graph, node_left, all_children_results, stage, suggested_result)
        return hashrandom_visitor_result(str(logged) + str(self._seed))

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
    tree.clear_log()
    res_first = impl_first(tree.root(), tree, visitor_first)
    tree_log_first = tree.log().copy()
    tree.clear_log()
    res_second = impl_second(tree.root(), tree, visitor_second)
    tree_log_second = tree.log().copy()
    tree.clear_log()
    assert res_first == res_second
    assert_lists_eq(visitor_first.log(), visitor_second.log())
    assert_lists_eq(tree_log_first, tree_log_second)



etalon = dfs
alternative = dfs_explicit


def test_dfs_and_stack_dfs_equivalence_no_flow():
    random.seed(54)
    for size in range(120):
        print(f"\n{size}", end='')
        for attempt in range(10):
            print(".", end='')
            tree = mk_random_tree(size)
            check_equivalence(tree, impl_first=etalon, impl_second=alternative,
                              visitor_maker=lambda: EventLoggingVisitor())
            # todo: check equivalence for flow controll


def test_dfs_and_stack_dfs_equivalence():
    random.seed(84)
    counter = 0
    for size in range(120):
        print(f"\n{size}", end='')
        for attempt in range(10):
            counter += 1
            print(".", end='')
            tree = mk_random_tree(size)
            seed_str = f"({counter})"
            visitor_maker = lambda: LoggingDroppingVisitor(seed=seed_str)
            check_equivalence(tree, impl_first=etalon, impl_second=alternative,
                              visitor_maker=visitor_maker)
