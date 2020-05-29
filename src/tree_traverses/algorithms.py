"""
28.05.2020 23:31
"""

from typing import (List, Optional, Dict, Any, NamedTuple)
from collections import defaultdict

from tree_traverses.interfaces import GraphInterface, TreeTraverseVisitor, ExitStage


def dfs(initial_node,
        tree: GraphInterface,
        visitor: TreeTraverseVisitor,
        ):
    """
    Depth first search. Doesn't check if it already visited node, so only suitable for trees.
    """
    all_children_results: List[Any] = []

    def at_exit(stage: ExitStage, suggested_result):
        vis_res = visitor.on_exit(graph=tree, node_left=initial_node,
                                  all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        return vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT else suggested_result

    vis_res = visitor.on_enter(graph=tree, entered_node=initial_node)
    if vis_res.do_return:
        return at_exit(ExitStage.ENTRANCE, vis_res.what_return)
    edges = tree.get_outgoing_edges(initial_node)
    for edge in edges:
        vis_res = visitor.on_traveling_edge(graph=tree, frm=initial_node, edge=edge)
        if vis_res.do_return:
            return at_exit(ExitStage.TRAVELING_EDGE, vis_res.what_return)
        if vis_res.do_break:
            break
        if vis_res.do_skip:
            continue

        child_node = tree.edge_destination(edge)
        vis_res = visitor.on_discover(graph=tree, frm=initial_node, discovered=child_node)
        if vis_res.do_return:
            return at_exit(ExitStage.NODE_DISCOVERED, vis_res.what_return)
        if vis_res.do_break:
            break
        if vis_res.do_skip:
            continue

        result = dfs(child_node, tree, visitor)
        all_children_results.append(result)
        vis_res = visitor.on_got_result(graph=tree, receiver_node=initial_node, sender_node=child_node, result=result,
                                        siblings_results=all_children_results)
        if vis_res.do_return:
            return at_exit(ExitStage.GOT_RESULT, vis_res.what_return)
        if vis_res.do_break:
            break
    return at_exit(ExitStage.EXIT, None)


def dfs_non_recursive(initial_node,
                      tree: GraphInterface,
                      visitor: TreeTraverseVisitor, ):
    nodes_to_visit = [initial_node]
    node_results: Dict[int, Any] = {}  # todo: make it operate with unhashable nodes
    children_of: Dict[int, List] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node: None}
    is_opened: Dict[int, bool] = defaultdict(bool)

    edge_generators = {}

    def close(node, stage, suggested_result, nodes_to_visit):
        nodes_to_visit.remove(node)  # todo: make more efficient by remembering pos of node
        if node in node_results:
            return
        all_children_results = [node_results[child] for child in children_of[node] if child in node_results]
        vis_res = visitor.on_exit(tree, node_left=node, all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT else suggested_result
        node_results[node] = node_value

        if node != initial_node:  # send result to parent
            siblings_result = [node_results[child] for child in children_of[parents[node]] if child in node_results]
            vis_res = visitor.on_got_result(tree, receiver_node=parents[node], sender_node=node, result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(parents[node], ExitStage.GOT_RESULT, vis_res.what_return, nodes_to_visit)
                return
            if vis_res.do_break:
                close(parents[node], ExitStage.EXIT, None, nodes_to_visit)
                return

    def edge_getter(node_id):
        nonlocal edge_generators
        if node_id not in edge_generators:
            def generator():
                yield from tree.get_outgoing_edges(node_id)

            edge_generators[node_id] = generator()
        return edge_generators[node_id]

    while nodes_to_visit:
        cur_node: int = nodes_to_visit[-1]

        if not is_opened[cur_node]:
            vis_res = visitor.on_enter(tree, entered_node=cur_node)
            is_opened[cur_node] = True
            if vis_res.do_return:
                close(cur_node, ExitStage.ENTRANCE, vis_res.what_return, nodes_to_visit)
                continue

        # edges = tree.get_outgoing_edges(cur_node)
        try:
            generator = edge_getter(cur_node)
            edge = next(generator)
        except StopIteration:
            close(cur_node, ExitStage.EXIT, None, nodes_to_visit)
            continue

        vis_res = visitor.on_traveling_edge(graph=tree, frm=cur_node, edge=edge)
        if vis_res.do_return:
            close(cur_node, ExitStage.TRAVELING_EDGE, vis_res.what_return, nodes_to_visit)
            continue
        if vis_res.do_break:
            close(cur_node, ExitStage.EXIT, None, nodes_to_visit)
            continue
        if vis_res.do_skip:
            continue

        child = tree.edge_destination(edge)
        parents[child] = cur_node
        vis_res = visitor.on_discover(tree, frm=cur_node, discovered=child)
        if vis_res.do_return:
            close(cur_node, ExitStage.NODE_DISCOVERED, vis_res.what_return, nodes_to_visit)
            continue
        if vis_res.do_break:
            close(cur_node, ExitStage.EXIT, None, nodes_to_visit)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node].append(child)
        nodes_to_visit.append(child)
    return node_results[initial_node]


def dfs_non_recursive_no_hashes(initial_node,
                                tree: GraphInterface,
                                visitor: TreeTraverseVisitor, ):
    nodes: List[Any] = [initial_node]
    initial_node_id: int = 0
    ids_to_visit: List[int] = [0]
    node_results: Dict[int, Any] = {}
    children_of: Dict[int, List[int]] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node_id: None}
    is_opened: Dict[int, bool] = defaultdict(bool)

    edge_generators = {}

    def close(node_id: int, stage: ExitStage, suggested_result, nodes_to_visit: List[int]):
        nodes_to_visit.remove(node_id)  # todo: make more efficient by remembering pos of node
        if node_id in node_results:
            return
        all_children_results = [node_results[child] for child in children_of[node_id] if child in node_results]
        vis_res = visitor.on_exit(tree, node_left=nodes[node_id], all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT else suggested_result
        node_results[node_id] = node_value

        if node_id != initial_node_id:  # send result to parent
            siblings_result = [node_results[child_id] for child_id in children_of[parents[node_id]] if
                               child_id in node_results]
            vis_res = visitor.on_got_result(tree, receiver_node=nodes[parents[node_id]], sender_node=nodes[node_id],
                                            result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(parents[node_id], ExitStage.GOT_RESULT, vis_res.what_return, nodes_to_visit)
                return
            if vis_res.do_break:
                close(parents[node_id], ExitStage.EXIT, None, nodes_to_visit)
                return

    def edge_getter(node_id: int):
        nonlocal edge_generators
        if node_id not in edge_generators:
            def generator():
                yield from tree.get_outgoing_edges(nodes[node_id])

            edge_generators[node_id] = generator()
        return edge_generators[node_id]

    while ids_to_visit:
        cur_node_id: int = ids_to_visit[-1]

        if not is_opened[cur_node_id]:
            vis_res = visitor.on_enter(tree, entered_node=nodes[cur_node_id])
            is_opened[cur_node_id] = True
            if vis_res.do_return:
                close(cur_node_id, ExitStage.ENTRANCE, vis_res.what_return, ids_to_visit)
                continue

        # if has remaining edges, get one
        try:
            generator = edge_getter(cur_node_id)
            edge = next(generator)
        except StopIteration:
            close(cur_node_id, ExitStage.EXIT, None, ids_to_visit)
            continue

        vis_res = visitor.on_traveling_edge(graph=tree, frm=nodes[cur_node_id], edge=edge)
        if vis_res.do_return:
            close(cur_node_id, ExitStage.TRAVELING_EDGE, vis_res.what_return, ids_to_visit)
            continue
        if vis_res.do_break:
            close(cur_node_id, ExitStage.EXIT, None, ids_to_visit)
            continue
        if vis_res.do_skip:
            continue

        child = tree.edge_destination(edge)
        # assuming all children are different!
        child_id = len(nodes)
        nodes.append(child)
        parents[child_id] = cur_node_id
        vis_res = visitor.on_discover(tree, frm=nodes[cur_node_id], discovered=child)
        if vis_res.do_return:
            close(cur_node_id, ExitStage.NODE_DISCOVERED, vis_res.what_return, ids_to_visit)
            continue
        if vis_res.do_break:
            close(cur_node_id, ExitStage.EXIT, None, ids_to_visit)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node_id].append(child_id)
        ids_to_visit.append(child_id)
    return node_results[initial_node_id]


def bfs(initial_node,
        tree: GraphInterface,
        visitor: TreeTraverseVisitor, ):
    nodes_to_visit = [initial_node]
    node_results: Dict[int, Any] = {}  # todo: make it operate with unhashable nodes
    children_of: Dict[int, List] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node: None}
    is_opened: Dict[int, bool] = defaultdict(bool)

    edge_generators = {}

    def close(node, stage, suggested_result, nodes_to_visit):
        nodes_to_visit.remove(node)  # todo: make more efficient by remembering pos of node
        if node in node_results:
            return
        all_children_results = [node_results[child] for child in children_of[node] if child in node_results]
        vis_res = visitor.on_exit(tree, node_left=node, all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT else suggested_result
        node_results[node] = node_value

        if node != initial_node:  # send result to parent
            siblings_result = [node_results[child] for child in children_of[parents[node]] if child in node_results]
            vis_res = visitor.on_got_result(tree, receiver_node=parents[node], sender_node=node, result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(parents[node], ExitStage.GOT_RESULT, vis_res.what_return, nodes_to_visit)
                return
            if vis_res.do_break:  # todo: in BFS break and return differ significantly
                close(parents[node], ExitStage.EXIT, None, nodes_to_visit)
                return

    def edge_getter(node_id):
        nonlocal edge_generators
        if node_id not in edge_generators:
            def generator():
                yield from tree.get_outgoing_edges(node_id)

            edge_generators[node_id] = generator()
        return edge_generators[node_id]

    while nodes_to_visit:
        cur_node: int = nodes_to_visit[0]

        if not is_opened[cur_node]:
            vis_res = visitor.on_enter(tree, entered_node=cur_node)
            is_opened[cur_node] = True
            if vis_res.do_return:
                close(cur_node, ExitStage.ENTRANCE, vis_res.what_return, nodes_to_visit)
                continue

        # edges = tree.get_outgoing_edges(cur_node)
        try:
            generator = edge_getter(cur_node)
            edge = next(generator)
        except StopIteration:
            close(cur_node, ExitStage.EXIT, None, nodes_to_visit)
            continue

        vis_res = visitor.on_traveling_edge(graph=tree, frm=cur_node, edge=edge)
        if vis_res.do_return:
            close(cur_node, ExitStage.TRAVELING_EDGE, vis_res.what_return, nodes_to_visit)
            continue
        if vis_res.do_break:  # todo: in BFS break and return differ significantly
            close(cur_node, ExitStage.EXIT, None, nodes_to_visit)
            continue
        if vis_res.do_skip:
            continue

        child = tree.edge_destination(edge)
        parents[child] = cur_node
        vis_res = visitor.on_discover(tree, frm=cur_node, discovered=child)
        if vis_res.do_return:
            close(cur_node, ExitStage.NODE_DISCOVERED, vis_res.what_return, nodes_to_visit)
            continue
        if vis_res.do_break:  # todo: in BFS break and return differ significantly
            close(cur_node, ExitStage.EXIT, None, nodes_to_visit)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node].append(child)
        nodes_to_visit.append(child)
    return node_results[initial_node]
