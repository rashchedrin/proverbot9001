from typing import (List, Optional, Dict, Any, Set)
from collections import defaultdict
from dataclasses import dataclass


@dataclass(init=True)
class DfsVisitorResult:
    do_return: bool = False  # for any visitor
    what_return: Any = None  # for any visitor
    do_skip: bool = False  # for visitor_on_discover: skip this child
    do_break: bool = False  # for visitor_on_discover and visitor_on_got_result: stop children discovery
    do_overwrite_result = False  # for visitor_on_exit. Iff true, returned value will be overwritten by the new one


def nop_visitor(*args, **kwargs) -> DfsVisitorResult:
    return DfsVisitorResult()


def dfs(initial_node,
        get_child_nodes,
        visitor_on_enter=nop_visitor,
        visitor_on_exit=nop_visitor,
        visitor_on_discover=nop_visitor,
        visitor_on_got_result=nop_visitor,
        ):
    all_children_results = []

    def at_exit(stage, suggested_result):
        vis_res = visitor_on_exit(node_left=initial_node, all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        return vis_res.what_return if vis_res.do_overwrite_result or stage == "EXIT" else suggested_result

    vis_res = visitor_on_enter(entered_node=initial_node)
    if vis_res.do_return:
        return at_exit("ENTRANCE", vis_res.what_return)
    for child_node in get_child_nodes(initial_node):
        vis_res = visitor_on_discover(frm=initial_node, discovered=child_node)
        if vis_res.do_return:
            return at_exit("DISCOVERY", vis_res.what_return)
        if vis_res.do_break:
            break
        if vis_res.do_skip:
            continue
        result = dfs(child_node, get_child_nodes,
                     visitor_on_enter,
                     visitor_on_exit,
                     visitor_on_discover,
                     visitor_on_got_result, )
        all_children_results.append(result)
        vis_res = visitor_on_got_result(parent=initial_node, node=child_node, result=result, all_children_results=all_children_results)
        if vis_res.do_return:
            return at_exit("GOT_RESULT", vis_res.what_return)
        if vis_res.do_break:
            break
    return at_exit("EXIT", None)


def dfs_explicit(initial_node,
                 get_child_nodes,
                 visitor_on_enter=nop_visitor,
                 visitor_on_exit=nop_visitor,
                 visitor_on_discover=nop_visitor, # this visitor is broken a bit
                 visitor_on_got_result=nop_visitor,
                 ):
    nodes_to_visit = [initial_node]
    node_results: Dict[int, Any] = {}
    children_of: Dict[int, List] = defaultdict(list)
    parents:Dict[int, Optional[int]] = {initial_node: None}

    def mark_visited_recursive(node):
        if node not in node_results:
            node_results[node] = "IRRELEVANT"
        for child in children_of[node]:
            mark_visited_recursive(child)

    def close(node, stage, suggested_result):
        if node in node_results:
            return
        all_children_results = [node_results[child] for child in children_of[node] if child in node_results]
        vis_res = visitor_on_exit(node_left=node, all_children_results=all_children_results,
                                  stage=stage, suggested_result=suggested_result)
        node_value = vis_res.what_return if vis_res.do_overwrite_result or stage == "EXIT" else suggested_result
        node_results[node] = node_value
        mark_visited_recursive(node)
        # manage

        if node != initial_node:
            vis_res = visitor_on_got_result(parent=parents[node], node=node, result=node_value,
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
        vis_res = visitor_on_enter(entered_node=cur_node)
        if vis_res.do_return:
            close(cur_node, "ENTRANCE", vis_res.what_return)
            continue
        children: List[int] = get_child_nodes(cur_node)
        for child in reversed(children):
            parents[child] = cur_node
            vis_res = visitor_on_discover(frm=cur_node, discovered=child)
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