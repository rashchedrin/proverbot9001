from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List


@dataclass(init=True)
class TraverseVisitorResult:
    do_return: bool = False  # for any visitor
    what_return: Any = None  # for any visitor
    do_skip: bool = False  # for visitor_on_discover: skip this child
    do_break: bool = False  # for visitor_on_discover and visitor_on_got_result: stop children discovery
    do_overwrite_result = False  # for visitor_on_exit. Iff true, returned value will be overwritten by the new one


class GraphInterface:
    def edge_destination(self, edge):
        raise NotImplementedError

    def get_outgoing_edges(self, node) -> List:
        raise NotImplementedError


class TreeTraverseVisitor:
    def on_enter(self, graph: GraphInterface,
                 entered_node) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_traveling_edge(self, graph: GraphInterface,
                          frm, edge) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_discover(self, graph: GraphInterface,
                    frm, discovered) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_got_result(self, graph: GraphInterface,
                      receiver_node, sender_node, result, siblings_results) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_exit(self, graph: GraphInterface,
                node_left, all_children_results, stage, suggested_result) -> TraverseVisitorResult:
        return TraverseVisitorResult()


class ExitStage(Enum):
    """Stage of DFS, on which exit was initiated"""
    ENTRANCE = auto()
    TRAVELING_EDGE = auto()
    NODE_DISCOVERED = auto()
    EXIT = auto()
    GOT_RESULT = auto()