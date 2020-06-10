"""
28.05.2020 23:31
"""

from typing import (List, Optional, Dict, Any, NamedTuple)
from collections import defaultdict
from enum import Enum
from functools import total_ordering

from tree_traverses.interfaces import GraphInterface, TreeTraverseVisitor, ExitStage, BestFirstSearchVisitor


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
        if vis_res.stop_discovering_edges:
            break
        if vis_res.do_skip:
            continue

        child_node = tree.edge_destination(edge)
        vis_res = visitor.on_discover(graph=tree, frm=initial_node, discovered=child_node)
        if vis_res.do_return:
            return at_exit(ExitStage.NODE_DISCOVERED, vis_res.what_return)
        if vis_res.stop_discovering_edges:
            break
        if vis_res.do_skip:
            continue

        result = dfs(child_node, tree, visitor)
        all_children_results.append(result)
        vis_res = visitor.on_got_result(graph=tree, receiver_node=initial_node, sender_node=child_node, result=result,
                                        siblings_results=all_children_results)
        if vis_res.do_return:
            return at_exit(ExitStage.GOT_RESULT, vis_res.what_return)
        if vis_res.stop_discovering_edges:
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
            if vis_res.stop_discovering_edges:
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
        if vis_res.stop_discovering_edges:
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
        if vis_res.stop_discovering_edges:
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
            if vis_res.stop_discovering_edges:
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
        if vis_res.stop_discovering_edges:
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
        if vis_res.stop_discovering_edges:
            close(cur_node_id, ExitStage.EXIT, None, ids_to_visit)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node_id].append(child_id)
        ids_to_visit.append(child_id)
    return node_results[initial_node_id]


@total_ordering
class BfsNodeState(Enum):
    """
    OPENED = visited by visitor.on_enter()
    EXPANDED = has no more edges to visit
    DELETED = node is not needed anymore, because parent is already closed
    CLOSED = has value
    """
    UNVISITED = 0
    OPENED = 1
    EXPANDED = 2
    DELETED = 3
    CLOSED = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        raise ValueError(f"Cant compare {self.__class__} with {other.__class__}")


def bdfs(initial_node,
         tree: GraphInterface,
         visitor: TreeTraverseVisitor, ):
    """
    Nodes can be:
        Opened: we've visited it, and called visitor.on_enter
        Expanded: there are no more edges to expand left. Now node is waiting for values of its children.
        Closed: Node has value.
    If visitor returns "do_break", it means that node becomes "expanded"
    If visitor returns "do_return", it means that node becomes closed, and all it's children are irrelevant now.
    BFS might:
        visit different vertices than DFS: for example if one of them returns early.
        return different value than DFS: for example if it returns set of visited vertices
    """
    nodes: List[Any] = [initial_node]
    initial_node_id: int = 0
    ids_to_visit: List[int] = [0]
    node_results: Dict[int, Any] = {}
    children_of: Dict[int, List[int]] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node_id: None}
    node_state: Dict[int, BfsNodeState] = defaultdict(lambda: BfsNodeState.UNVISITED)

    def promote(node_id: int, to_state: BfsNodeState):
        node_state[node_id] = max(node_state[node_id], to_state)

    edge_generators = {}

    def empty_generator():
        yield from []

    def mark_expanded(node_id: int):
        if node_state[node_id] == BfsNodeState.EXPANDED:
            return
        promote(node_id, BfsNodeState.EXPANDED)
        ids_to_visit.remove(node_id)  # todo: make more efficient by remembering pos of node
        edge_generators[node_id] = empty_generator()
        if all(node_state[child_id] == BfsNodeState.CLOSED for child_id in children_of[node_id]):
            close(node_id, ExitStage.EXIT, None)

    def mark_subtree_deleted(subtree_root_id: int):
        children = children_of[subtree_root_id]
        promote(subtree_root_id, BfsNodeState.DELETED)
        try:
            ids_to_visit.remove(subtree_root_id)
        except ValueError:
            pass
        for child_id in children:
            mark_subtree_deleted(child_id)

    def close(node_id: int, stage: ExitStage, suggested_result):
        """
        Set value. Mark closed.
        """
        if node_state[node_id] in [BfsNodeState.CLOSED, BfsNodeState.DELETED]:
            return

        if node_state[node_id] < BfsNodeState.EXPANDED:
            promote(node_id, BfsNodeState.EXPANDED)
            edge_generators[node_id] = empty_generator()

        all_children_results = [node_results[child] for child in children_of[node_id] if child in node_results]
        vis_res = visitor.on_exit(tree, node_left=nodes[node_id], all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT \
            else suggested_result
        node_results[node_id] = node_value
        mark_subtree_deleted(node_id)
        promote(node_id, BfsNodeState.CLOSED)

        if node_id != initial_node_id:  # send result to parent
            siblings_result = [node_results[child_id] for child_id in children_of[parents[node_id]] if
                               child_id in node_results]
            reciever_id = parents[node_id]
            vis_res = visitor.on_got_result(tree, receiver_node=nodes[reciever_id],
                                            sender_node=nodes[node_id],
                                            result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(reciever_id, ExitStage.GOT_RESULT, vis_res.what_return)
                return
            if vis_res.stop_discovering_edges:
                mark_expanded(reciever_id)
                return
            # propagate if can
            if node_state[reciever_id] == BfsNodeState.EXPANDED \
                    and all(node_state[child_id] == BfsNodeState.CLOSED for child_id in children_of[reciever_id]):
                close(reciever_id, ExitStage.EXIT, None)
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

        if node_state[cur_node_id] == BfsNodeState.UNVISITED:
            vis_res = visitor.on_enter(tree, entered_node=nodes[cur_node_id])
            promote(cur_node_id, BfsNodeState.OPENED)
            if vis_res.do_return:
                close(cur_node_id, ExitStage.ENTRANCE, vis_res.what_return)
                continue

        # if has remaining edges, get one
        try:
            generator = edge_getter(cur_node_id)
            edge = next(generator)
        except StopIteration:
            mark_expanded(cur_node_id)
            continue

        vis_res = visitor.on_traveling_edge(graph=tree, frm=nodes[cur_node_id], edge=edge)
        if vis_res.do_return:
            close(cur_node_id, ExitStage.TRAVELING_EDGE, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_node_id)
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
            close(cur_node_id, ExitStage.NODE_DISCOVERED, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_node_id)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node_id].append(child_id)
        ids_to_visit.append(child_id)
    return node_results[initial_node_id]


def bfs(initial_node,
        tree: GraphInterface,
        visitor: TreeTraverseVisitor, ):
    # todo: test
    """
    Nodes can be:
        Opened: we've visited it, and called visitor.on_enter
        Expanded: there are no more edges to expand left. Now node is waiting for values of its children.
        Closed: Node has value.
    If visitor returns "do_break", it means that node becomes "expanded"
    If visitor returns "do_return", it means that node becomes closed, and all it's children are irrelevant now.
    BFS might:
        visit different vertices than DFS: for example if one of them returns early.
        return different value than DFS: for example if it returns set of visited vertices
    """
    nodes: List[Any] = [initial_node]
    initial_node_id: int = 0
    ids_to_visit: List[int] = [0]
    node_results: Dict[int, Any] = {}
    children_of: Dict[int, List[int]] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node_id: None}
    node_state: Dict[int, BfsNodeState] = defaultdict(lambda: BfsNodeState.UNVISITED)

    def promote(node_id: int, to_state: BfsNodeState):
        node_state[node_id] = max(node_state[node_id], to_state)

    edge_generators = {}

    def empty_generator():
        yield from []

    def mark_expanded(node_id: int):
        if node_state[node_id] == BfsNodeState.EXPANDED:
            return
        promote(node_id, BfsNodeState.EXPANDED)
        ids_to_visit.remove(node_id)  # todo: make more efficient by remembering pos of node
        edge_generators[node_id] = empty_generator()
        if all(node_state[child_id] == BfsNodeState.CLOSED for child_id in children_of[node_id]):
            close(node_id, ExitStage.EXIT, None)

    def mark_subtree_deleted(subtree_root_id: int):
        children = children_of[subtree_root_id]
        promote(subtree_root_id, BfsNodeState.DELETED)
        try:
            ids_to_visit.remove(subtree_root_id)
        except ValueError:
            pass
        for child_id in children:
            mark_subtree_deleted(child_id)

    def close(node_id: int, stage: ExitStage, suggested_result):
        """
        Set value. Mark closed.
        """
        if node_state[node_id] in [BfsNodeState.CLOSED, BfsNodeState.DELETED]:
            return

        if node_state[node_id] < BfsNodeState.EXPANDED:
            promote(node_id, BfsNodeState.EXPANDED)
            edge_generators[node_id] = empty_generator()

        all_children_results = [node_results[child] for child in children_of[node_id] if child in node_results]
        vis_res = visitor.on_exit(tree, node_left=nodes[node_id], all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT \
            else suggested_result
        node_results[node_id] = node_value
        mark_subtree_deleted(node_id)
        promote(node_id, BfsNodeState.CLOSED)

        if node_id != initial_node_id:  # send result to parent
            siblings_result = [node_results[child_id] for child_id in children_of[parents[node_id]] if
                               child_id in node_results]
            reciever_id = parents[node_id]
            vis_res = visitor.on_got_result(tree, receiver_node=nodes[reciever_id],
                                            sender_node=nodes[node_id],
                                            result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(reciever_id, ExitStage.GOT_RESULT, vis_res.what_return)
                return
            if vis_res.stop_discovering_edges:
                mark_expanded(reciever_id)
                return
            # propagate if can
            if node_state[reciever_id] == BfsNodeState.EXPANDED \
                    and all(node_state[child_id] == BfsNodeState.CLOSED for child_id in children_of[reciever_id]):
                close(reciever_id, ExitStage.EXIT, None)
                return

    def edge_getter(node_id: int):
        nonlocal edge_generators
        if node_id not in edge_generators:
            def generator():
                yield from tree.get_outgoing_edges(nodes[node_id])

            edge_generators[node_id] = generator()
        return edge_generators[node_id]

    while ids_to_visit:
        cur_node_id: int = ids_to_visit[0]

        if node_state[cur_node_id] == BfsNodeState.UNVISITED:
            vis_res = visitor.on_enter(tree, entered_node=nodes[cur_node_id])
            promote(cur_node_id, BfsNodeState.OPENED)
            if vis_res.do_return:
                close(cur_node_id, ExitStage.ENTRANCE, vis_res.what_return)
                continue

        # if has remaining edges, get one
        try:
            generator = edge_getter(cur_node_id)
            edge = next(generator)
        except StopIteration:
            mark_expanded(cur_node_id)
            continue

        vis_res = visitor.on_traveling_edge(graph=tree, frm=nodes[cur_node_id], edge=edge)
        if vis_res.do_return:
            close(cur_node_id, ExitStage.TRAVELING_EDGE, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_node_id)
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
            close(cur_node_id, ExitStage.NODE_DISCOVERED, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_node_id)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node_id].append(child_id)
        ids_to_visit.append(child_id)
    return node_results[initial_node_id]


def best_first_search_old(initial_node,
                          tree: GraphInterface,
                          visitor: BestFirstSearchVisitor, ):
    # todo: test
    """
    Nodes can be:
        Opened: we've visited it, and called visitor.on_enter
        Expanded: there are no more edges to expand left. Now node is waiting for values of its children.
        Closed: Node has value.
    If visitor returns "do_break", it means that node becomes "expanded"
    If visitor returns "do_return", it means that node becomes closed, and all it's children are irrelevant now.
    BFS might:
        visit different vertices than DFS: for example if one of them returns early.
        return different value than DFS: for example if it returns set of visited vertices
    """
    nodes: List[Any] = [initial_node]
    initial_node_id: int = 0
    ids_to_visit: List[int] = [0]
    node_results: Dict[int, Any] = {}
    children_of: Dict[int, List[int]] = defaultdict(list)
    parents: Dict[int, Optional[int]] = {initial_node_id: None}
    node_state: Dict[int, BfsNodeState] = defaultdict(lambda: BfsNodeState.UNVISITED)

    def promote(node_id: int, to_state: BfsNodeState):
        node_state[node_id] = max(node_state[node_id], to_state)

    edge_generators = {}

    def empty_generator():
        yield from []

    def mark_expanded(node_id: int):
        if node_state[node_id] == BfsNodeState.EXPANDED:
            return
        promote(node_id, BfsNodeState.EXPANDED)
        ids_to_visit.remove(node_id)  # todo: make more efficient by remembering pos of node
        edge_generators[node_id] = empty_generator()
        if all(node_state[child_id] == BfsNodeState.CLOSED for child_id in children_of[node_id]):
            close(node_id, ExitStage.EXIT, None)

    def mark_subtree_deleted(subtree_root_id: int):
        children = children_of[subtree_root_id]
        promote(subtree_root_id, BfsNodeState.DELETED)
        try:
            ids_to_visit.remove(subtree_root_id)
        except ValueError:
            pass
        for child_id in children:
            mark_subtree_deleted(child_id)

    def close(node_id: int, stage: ExitStage, suggested_result):
        """
        Set value. Mark closed.
        """
        if node_state[node_id] in [BfsNodeState.CLOSED, BfsNodeState.DELETED]:
            return

        if node_state[node_id] < BfsNodeState.EXPANDED:
            promote(node_id, BfsNodeState.EXPANDED)
            edge_generators[node_id] = empty_generator()

        all_children_results = [node_results[child] for child in children_of[node_id] if child in node_results]
        vis_res = visitor.on_exit(tree, node_left=nodes[node_id], all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT \
            else suggested_result
        node_results[node_id] = node_value
        mark_subtree_deleted(node_id)
        promote(node_id, BfsNodeState.CLOSED)

        if node_id != initial_node_id:  # send result to parent
            siblings_result = [node_results[child_id] for child_id in children_of[parents[node_id]] if
                               child_id in node_results]
            reciever_id = parents[node_id]
            vis_res = visitor.on_got_result(tree, receiver_node=nodes[reciever_id],
                                            sender_node=nodes[node_id],
                                            result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(reciever_id, ExitStage.GOT_RESULT, vis_res.what_return)
                return
            if vis_res.stop_discovering_edges:
                mark_expanded(reciever_id)
                return
            # propagate if can
            if node_state[reciever_id] == BfsNodeState.EXPANDED \
                    and all(node_state[child_id] == BfsNodeState.CLOSED for child_id in children_of[reciever_id]):
                close(reciever_id, ExitStage.EXIT, None)
                return

    def edge_getter(node_id: int):
        nonlocal edge_generators
        if node_id not in edge_generators:
            def generator():
                yield from tree.get_outgoing_edges(nodes[node_id])

            edge_generators[node_id] = generator()
        return edge_generators[node_id]

    while ids_to_visit:
        leaf_index = visitor.leaf_picker(tree=tree,
                                         leaves=[nodes[id] for id in ids_to_visit])  # todo: can optimize here.
        cur_node_id: int = ids_to_visit[leaf_index]

        if node_state[cur_node_id] == BfsNodeState.UNVISITED:
            vis_res = visitor.on_enter(tree, entered_node=nodes[cur_node_id])
            promote(cur_node_id, BfsNodeState.OPENED)
            if vis_res.do_return:
                close(cur_node_id, ExitStage.ENTRANCE, vis_res.what_return)
                continue

        # if has remaining edges, get one
        try:
            generator = edge_getter(cur_node_id)
            edge = next(generator)
        except StopIteration:
            mark_expanded(cur_node_id)
            continue

        vis_res = visitor.on_traveling_edge(graph=tree, frm=nodes[cur_node_id], edge=edge)
        if vis_res.do_return:
            close(cur_node_id, ExitStage.TRAVELING_EDGE, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_node_id)
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
            close(cur_node_id, ExitStage.NODE_DISCOVERED, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_node_id)
            continue
        if vis_res.do_skip:
            continue
        children_of[cur_node_id].append(child_id)
        ids_to_visit.append(child_id)
    return node_results[initial_node_id]


class EdgeAndOrigin(NamedTuple):
    edge: Any
    origin_node_id: int


class BestFSNodeWrapper:
    def __init__(self, node: Any, parent_id: Optional[int]):
        self.__result: Any = None
        self.__has_result: bool = False
        self.state: BfsNodeState = BfsNodeState.UNVISITED
        self.parent_id: int = parent_id
        self._children_ids: List[int] = []
        self._ids_edges_from: List[int] = []
        self.node: Any = node

    def set_value(self, res):
        self.__result = res
        self.__has_result = True
        self.promote(BfsNodeState.CLOSED)

    def has_value(self):
        return self.__has_result

    def get_result(self):
        if self.__has_result:
            return self.__result
        raise RuntimeError("Node doesn't have result")

    def promote(self, new_state: BfsNodeState):
        assert new_state > self.state
        self.state = new_state

    def set_edges_ids(self, edges_ids):
        assert self.state == BfsNodeState.UNVISITED
        self._ids_edges_from = edges_ids.copy()
        self.promote(BfsNodeState.OPENED)

    def edges_ids(self):
        return self._ids_edges_from.copy()


def best_first_search(initial_node: Any,
                      tree: GraphInterface,
                      visitor: BestFirstSearchVisitor,
                      debug=False):
    """
    Nodes can be:
        Opened: we've visited it, and called visitor.on_enter
        Expanded: there are no more edges to expand left. Now node is waiting for values of its children.
        Closed: Node has value.
    If visitor returns "do_break", it means that node becomes "expanded"
    If visitor returns "do_return", it means that node becomes closed, and all it's children are irrelevant now.
    BFS might:
        visit different vertices than DFS: for example if one of them returns early.
        return different value than DFS: for example if it returns set of visited vertices
    """
    nodes_info: List[BestFSNodeWrapper] = [BestFSNodeWrapper(initial_node, None)]
    edges_and_origins: List[EdgeAndOrigin] = []
    initial_node_id: int = 0
    ids_edges_to_visit: List[int] = []

    UNVISITED = BfsNodeState.UNVISITED
    OPENED = BfsNodeState.OPENED
    EXPANDED = BfsNodeState.EXPANDED
    CLOSED = BfsNodeState.CLOSED
    DELETED = BfsNodeState.DELETED

    def get_edges_ids(node_id: int) -> List[Any]:
        nonlocal edges_and_origins
        if nodes_info[node_id].state == UNVISITED:
            vis_res = visitor.on_enter(tree, nodes_info[node_id].node)
            if vis_res.do_return:
                close(node_id, ExitStage.ENTRANCE, vis_res.what_return)
                return nodes_info[node_id].edges_ids()
            new_edges = tree.get_outgoing_edges(nodes_info[node_id].node)
            if not new_edges:
                mark_expanded(node_id)
                return nodes_info[node_id].edges_ids()
            ids_from = len(edges_and_origins)
            edges_and_origins += [EdgeAndOrigin(e, node_id) for e in new_edges]
            ids_to = len(edges_and_origins)
            nodes_info[node_id].set_edges_ids(list(range(ids_from, ids_to)))
        return nodes_info[node_id].edges_ids()

    def unqueue_edges_of(node_id: int):
        for edge_id in nodes_info[node_id].edges_ids():
            try:
                ids_edges_to_visit.remove(edge_id)
            except ValueError:
                pass

    def mark_expanded(node_id: int):
        if nodes_info[node_id].state >= EXPANDED:
            return
        unqueue_edges_of(node_id)
        nodes_info[node_id].promote(EXPANDED)
        attempt_close(node_id)

    def attempt_close(node_id: int):
        """close node, if all criteria met"""
        if nodes_info[node_id].state == EXPANDED and \
                all(nodes_info[child_id].state in [CLOSED, DELETED]
                    for child_id in nodes_info[node_id]._children_ids):
            close(node_id, ExitStage.EXIT, None)

    def discard_subtree(subtree_root_id: int):
        if nodes_info[subtree_root_id].state in [CLOSED, DELETED]:
            return
        children_ids = nodes_info[subtree_root_id]._children_ids
        unqueue_edges_of(subtree_root_id)
        nodes_info[subtree_root_id].promote(DELETED)
        for child_id in children_ids:
            discard_subtree(child_id)

    def assert_invariants():
        if not debug:
            return
        # 1. queue has only edges without destination
        # can't chack this
        # 2. children of deleted are deleted
        for node in nodes_info:
            if node.state == DELETED:
                for child_id in node._children_ids:
                    assert nodes_info[child_id].state in [DELETED, CLOSED]
        # 3. children of closed are deleted or closed
        for node in nodes_info:
            if node.state == CLOSED:
                for child_id in node._children_ids:
                    assert nodes_info[child_id].state in [DELETED, CLOSED]
        # 4. only closed and deleted can have values
        for node in nodes_info:
            if node.has_value():
                assert node.state in [CLOSED, DELETED]
        # 4.1. all closed have values
        for node in nodes_info:
            if node.state == CLOSED:
                assert node.has_value()
        # 5. Nodes with state >= expanded can't have edges in queue
        for node in nodes_info:
            if node.state >= EXPANDED:
                assert all([edge_id not in node.edges_ids() for edge_id in ids_edges_to_visit])

    def all_children_results(node_id):
        children_ids = nodes_info[node_id]._children_ids
        return [nodes_info[child].get_result() for child in children_ids if nodes_info[child].has_value()]

    def pull_value(node_id: int, stage: ExitStage, suggested_result):
        node = nodes_info[node_id].node
        vis_res = visitor.on_exit(tree, node_left=node,
                                  all_children_results=all_children_results(node_id),
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == ExitStage.EXIT \
            else suggested_result
        nodes_info[node_id].set_value(node_value)
        return node_value

    def close(node_id: int, stage: ExitStage, suggested_result):
        """
        Set value. Mark closed.
        """
        if nodes_info[node_id].state in [CLOSED, DELETED]:
            return

        if nodes_info[node_id].state < EXPANDED:
            unqueue_edges_of(node_id)
            nodes_info[node_id].promote(EXPANDED)

        node_value = pull_value(node_id, stage, suggested_result)
        for child_id in nodes_info[node_id]._children_ids:
            discard_subtree(child_id)

        if node_id != initial_node_id:  # send result to parent
            reciever_id = nodes_info[node_id].parent_id
            siblings_result = all_children_results(reciever_id)
            vis_res = visitor.on_got_result(tree, receiver_node=nodes_info[reciever_id].node,
                                            sender_node=nodes_info[node_id].node,
                                            result=node_value,
                                            siblings_results=siblings_result)
            if vis_res.do_return:
                close(reciever_id, ExitStage.GOT_RESULT, vis_res.what_return)
                return
            if vis_res.stop_discovering_edges:
                mark_expanded(reciever_id)
            # propagate if can
            attempt_close(reciever_id)

    def choose_edge():  # todo: can optimize here.
        return visitor.edge_picker(tree=tree, leaf_edges=[edges_and_origins[id].edge for id in ids_edges_to_visit])

    def attempt_mark_expanded(node_id):
        """
        if no edges in queue, mark expanded
        """
        for e in ids_edges_to_visit:
            if edges_and_origins[e].origin_node_id == node_id:
                return
        mark_expanded(node_id)

    ids_edges_to_visit = list(reversed(get_edges_ids(initial_node_id)))
    while ids_edges_to_visit:
        assert_invariants()
        cur_edge_id: int = ids_edges_to_visit.pop(choose_edge())
        cur_edge_and_origin = edges_and_origins[cur_edge_id]

        vis_res = visitor.on_traveling_edge(graph=tree,
                                            frm=nodes_info[cur_edge_and_origin.origin_node_id].node,
                                            edge=cur_edge_and_origin.edge)
        if vis_res.do_return:
            close(cur_edge_and_origin.origin_node_id, ExitStage.TRAVELING_EDGE, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_edge_and_origin.origin_node_id)
            continue
        if vis_res.do_skip:
            attempt_mark_expanded(cur_edge_and_origin.origin_node_id)
            continue
        child = tree.edge_destination(cur_edge_and_origin.edge)
        vis_res = visitor.on_discover(tree, frm=nodes_info[cur_edge_and_origin.origin_node_id].node, discovered=child)
        if vis_res.do_return:
            close(cur_edge_and_origin.origin_node_id, ExitStage.NODE_DISCOVERED, vis_res.what_return)
            continue
        if vis_res.stop_discovering_edges:
            mark_expanded(cur_edge_and_origin.origin_node_id)
            continue
        if vis_res.do_skip:
            attempt_mark_expanded(cur_edge_and_origin.origin_node_id)
            continue
        # assuming all children are different!
        child_id = len(nodes_info)
        nodes_info.append(BestFSNodeWrapper(child, cur_edge_and_origin.origin_node_id))
        nodes_info[cur_edge_and_origin.origin_node_id]._children_ids.append(child_id)
        child_edges_ids: List = get_edges_ids(child_id)
        # reverse, to allow converting it to DFS using `choose_edge = last`
        ids_edges_to_visit += list(reversed(child_edges_ids))
        attempt_mark_expanded(cur_edge_and_origin.origin_node_id)
    assert_invariants()
    if nodes_info[initial_node_id].state != CLOSED:
        print(f"initial node is {nodes_info[initial_node_id].state} but must be closed")
        # print(*[(ch, node_state[ch]) for ch in children_of[initial_node_id]])
        assert nodes_info[initial_node_id].state == CLOSED
    return nodes_info[initial_node_id].get_result()
