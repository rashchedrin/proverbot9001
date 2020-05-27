from typing import (List, Optional, Dict, Any, Set, Type)
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(init=True)
class TraverseVisitorResult:
    do_return: bool = False  # for any visitor
    what_return: Any = None  # for any visitor
    do_skip: bool = False  # for visitor_on_discover: skip this child
    do_break: bool = False  # for visitor_on_discover and visitor_on_got_result: stop children discovery
    do_overwrite_result = False  # for visitor_on_exit. Iff true, returned value will be overwritten by the new one


def nop_visitor(*args, **kwargs) -> TraverseVisitorResult:
    return TraverseVisitorResult()


class GraphInterface:
    def edge_destination(self, edge):
        raise NotImplementedError

    def get_outgoing_edges(self, node) -> List:
        raise NotImplementedError


class TreeTraverseVisitor:
    def on_enter(self, graph: GraphInterface,
                 entered_node) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_got_edge(self, graph: GraphInterface,
                    frm, edge) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_discover(self, graph: GraphInterface,
                    frm, discovered) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_got_result(self, graph: GraphInterface,
                      parent, node, result, all_children_results) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_exit(self, graph: GraphInterface,
                node_left, all_children_results, stage, suggested_result) -> TraverseVisitorResult:
        return TraverseVisitorResult()


class ExitStage(Enum):
    """Stage of DFS, on which exit was initiated"""
    ENTRANCE = auto()
    GOT_EDGE = auto()
    NODE_DISCOVERED = auto()
    EXIT = auto()
    GOT_RESULT = auto()


def dfs(initial_node,
        graph: GraphInterface,
        visitor: TreeTraverseVisitor,
        ):
    all_children_results = []

    def at_exit(stage: ExitStage, suggested_result):
        vis_res = visitor.on_exit(graph=graph, node_left=initial_node,
                                  all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        return vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT else suggested_result

    vis_res = visitor.on_enter(graph=graph, entered_node=initial_node)
    if vis_res.do_return:
        return at_exit(ExitStage.ENTRANCE, vis_res.what_return)
    for edge in graph.get_outgoing_edges(initial_node):
        vis_res = visitor.on_got_edge(graph=graph, frm=initial_node, edge=edge)
        if vis_res.do_return:
            return at_exit(ExitStage.GOT_EDGE, vis_res.what_return)
        if vis_res.do_break:
            break
        if vis_res.do_skip:
            continue

        child_node = graph.edge_destination(edge)
        vis_res = visitor.on_discover(graph=graph, frm=initial_node, discovered=child_node)
        if vis_res.do_return:
            return at_exit(ExitStage.NODE_DISCOVERED, vis_res.what_return)
        if vis_res.do_break:
            break
        if vis_res.do_skip:
            continue

        result = dfs(child_node, graph, visitor)
        all_children_results.append(result)
        vis_res = visitor.on_got_result(graph=graph, parent=initial_node, node=child_node, result=result,
                                        all_children_results=all_children_results)
        if vis_res.do_return:
            return at_exit(ExitStage.GOT_RESULT, vis_res.what_return)
        if vis_res.do_break:
            break
    return at_exit(ExitStage.EXIT, None)


@dataclass
class NodeExit:
    node: int


def dfs_explicit(initial_node,
                 get_outgoing_edges,
                 edge_destination,
                 visitor: TreeTraverseVisitor,
                 ):
    nodes_to_visit = [initial_node]
    node_results: Dict[int, Any] = {}
    children_of: Dict[int, List] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node: None}

    def mark_visited_recursive(node):
        if node not in node_results:
            node_results[node] = "IRRELEVANT"
        for child in children_of[node]:
            mark_visited_recursive(child)

    def close(node, stage, suggested_result):
        if node in node_results:
            return
        all_children_results = [node_results[child] for child in children_of[node] if child in node_results]
        vis_res = visitor.on_exit(node_left=node, all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == "EXIT" else suggested_result
        node_results[node] = node_value
        mark_visited_recursive(node)
        # manage

        if node != initial_node:
            vis_res = visitor.on_got_result(parent=parents[node], node=node, result=node_value,
                                            all_children_results=all_children_results)
            if vis_res.do_return:
                close(parents[node], "GOT_RESULT", vis_res.what_return)
                return
            if vis_res.do_break:
                close(parents[node], "EXIT", None)
                return

        # propagate
        parent = parents[node]
        if parent is not None:
            siblings = children_of[parent]
            if all([sibling in node_results for sibling in siblings]):
                close(parent, "EXIT", None)

    while nodes_to_visit:
        cur_node: int = nodes_to_visit.pop()
        if cur_node in node_results:
            continue
        vis_res = visitor.on_enter(entered_node=cur_node)
        if vis_res.do_return:
            close(cur_node, "ENTRANCE", vis_res.what_return)
            continue
        children: List[int] = list(map(edge_destination, get_outgoing_edges(cur_node)))
        for child in reversed(children):
            parents[child] = cur_node
            vis_res = visitor.on_discover(frm=cur_node, discovered=child)
            if vis_res.do_return:
                close(cur_node, "DISCOVERY", vis_res.what_return)
                break
            if vis_res.do_break:
                close(cur_node, "EXIT", None)
                break
            if vis_res.do_skip:
                continue
            children_of[cur_node].insert(0, child)
            nodes_to_visit.append(child)
        if len(children_of[cur_node]) == 0:
            close(cur_node, "EXIT", None)
            continue
    return node_results[initial_node]


def bfs(initial_node,
        get_outgoing_edges,
        edge_destination,
        visitor: TreeTraverseVisitor, ):
    queue = [initial_node]
    while queue:
        cur_node = queue.pop(0)
        visitor.on_enter(cur_node)
        for child in map(edge_destination, get_outgoing_edges(cur_node)):
            queue.append(child)
            visitor.on_discover(cur_node, child)
