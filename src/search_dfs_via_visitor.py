"""
DFS search strategy for Proverbot9001
"""
import argparse
import sys
from typing import (List, Optional, Dict, Any, Set, Tuple, NamedTuple)
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import search_file
import serapi_instance
from serapi_instance import ProofContext
from search_file import (SearchResult, SubSearchResult, SearchGraph, LabeledNode, TacticContext,
                         tryPrediction, completed_proof, contextInPath, contextIsBig, numNodesInTree,
                         SearchStatus, TqdmSpy)
from util import (eprint, escape_lemma_name,
                  mybarfmt)
from graph_traverses import dfs, TreeTraverseVisitor, GraphInterface, TraverseVisitorResult


def get_relevant_lemmas(args, coq):
    if args.relevant_lemmas == "local":
        return coq.local_lemmas[:-1]
    if args.relevant_lemmas == "hammer":
        return coq.get_hammer_premises()
    if args.relevant_lemmas == "searchabout":
        return coq.get_lemmas_about_head()
    raise RuntimeError(f"Unsupported relevant_lemmas type {args.relevant_lemmas}")


def cancel_last_statements(coq: serapi_instance.SerapiInstance,
                           num_stmts: int,
                           args: argparse.Namespace,
                           msg: Optional[str] = None):
    if msg:
        eprint(f"Cancelling {num_stmts} statements "
               f"because {msg}.", guard=args.verbose >= 2)
    for _ in range(num_stmts):
        coq.cancel_last()


def eprint_cancel(desired_state: int, args: argparse.Namespace, msg: Optional[str]):
    if msg:
        eprint(f"Cancelling until {desired_state} statements "
               f"because {msg}.", guard=args.verbose >= 2)


def cancel_until_state(coq: serapi_instance.SerapiInstance,
                       desired_state: int,
                       args: argparse.Namespace,
                       msg: Optional[str] = None):
    eprint_cancel(desired_state, args, msg)
    while coq.cur_state != desired_state:
        coq.cancel_last()


def goto_state_fake(coq: serapi_instance.SerapiInstance,
                    desired_state: int,
                    args: argparse.Namespace,
                    msg: Optional[str] = None):  # Todo: make real
    return cancel_until_state(coq, desired_state, args, msg)


def predict_k_tactics(coq: serapi_instance.SerapiInstance, args: argparse.Namespace, k: int):
    relevant_lemmas = get_relevant_lemmas(args, coq)
    tactic_context_before = TacticContext(relevant_lemmas, coq.prev_tactics, coq.hypotheses, coq.goals)
    predictions = [prediction.prediction for prediction in
                   search_file.predictor.predictKTactics(tactic_context_before, k)]
    if coq.use_hammer:
        predictions = add_hammer_commands(predictions)
    return predictions


def add_hammer_commands(predictions):
    return [prediction + "; try hammer." for prediction in predictions]


def update_distance_stack(extra_depth, subgoal_distance_stack, subgoals_closed, subgoals_opened):
    #### 1.
    if subgoal_distance_stack:
        new_distance_stack = (subgoal_distance_stack[:-1] +
                              [subgoal_distance_stack[-1] + 1])
    else:
        new_distance_stack = []
    #### 2.
    new_extra_depth = extra_depth
    for _ in range(subgoals_closed):
        closed_goal_distance = new_distance_stack.pop()
        new_extra_depth += closed_goal_distance
    #### 3.
    new_distance_stack += [0] * subgoals_opened
    return new_distance_stack, new_extra_depth


def manage_returned_result(sub_search_result, subgoals_closed, subgoals_opened):
    return_result = None
    if sub_search_result.solution or sub_search_result.solved_subgoals > subgoals_opened:
        new_subgoals_closed = sub_search_result.solved_subgoals + subgoals_closed - subgoals_opened  # what is it?
        return_result = SubSearchResult(sub_search_result.solution, new_subgoals_closed)
    elif subgoals_closed > 0:  # what does it mean?
        return_result = SubSearchResult(None, subgoals_closed)
    return return_result


class Edge(NamedTuple):
    frm: int
    tactic: str



class CoqGraphInterface(GraphInterface):
    """
    Interface to Coq as a graph
    """

    class ExtraNodeInfo(NamedTuple):
        """
        Additional node info, that CoqGraphInterface stores for each node
        """
        context_after: ProofContext
        subgoals_opened: int
        subgoals_closed: int
        is_proof_completed: bool

    def __init__(self,
                 coq: serapi_instance.SerapiInstance,
                 args: argparse.Namespace,
                 start_node
                 ):
        self._coq = coq
        self._args = args
        self._memoized_outgoing_edges: Dict[int, List[Edge]] = {}
        self._memoized_edge_destinations: Dict[int, Optional[LabeledNode]] = {}
        self._id2node: Dict[int, LabeledNode] = {}
        self._node_infos: Dict[int, CoqGraphInterface.ExtraNodeInfo] = {}

        # todo: figure out correct initialization
        root_node = LabeledNode("root_dummy_prediction", 0, coq.cur_state, coq.proof_context, start_node)
        self._id2node[root_node.node_id] = root_node
        self._node_infos[root_node.node_id] = CoqGraphInterface.ExtraNodeInfo(
            coq.proof_context, 0, 0, False
        )
        self._node_infos[start_node.node_id] = CoqGraphInterface.ExtraNodeInfo(
            coq.proof_context, 0, 0, False
        )

    def get_outgoing_edges(self, node: LabeledNode) -> List[Edge]:
        """
        Calls neural network to get predictions
        memoizes to self.memoized_outgoing_edges
        """
        if node.node_id in self._memoized_outgoing_edges:
            return self._memoized_outgoing_edges[node.node_id]
        goto_state_fake(self._coq, node.node_id, self._args)
        predictions = predict_k_tactics(self._coq, self._args, self._args.max_attempts)
        edges = [Edge(node.node_id, pred) for pred in predictions]
        self._memoized_outgoing_edges[node.node_id] = edges
        return edges

    def edge_destination(self, edge: Edge) -> Optional[LabeledNode]:
        """
        Creates new node: sends commands to Coq to get it
        Memoizes to self.memoized_edge_destinations
        adds new node to self.id2node
        adds values in
            self.contexts_after[new_state],
            self.subgoals_opened[new_state],
            self.subgoals_closed[new_state]
        returns None on Coq error
        """
        if edge in self._memoized_outgoing_edges:
            return self._memoized_outgoing_edges[edge]
        goto_state_fake(self._coq, edge.frm, self._args)
        parent_node = self._id2node[edge.frm].previous
        context_after, _, \
        subgoals_closed, subgoals_opened, \
        error, time_taken, new_state = \
            tryPrediction(self._args, self._coq, edge.tactic, parent_node)
        if error:
            return None
        context_before = self._node_infos[parent_node.node_id].context_after
        is_proof_completed = completed_proof(self._coq)
        extra_info = CoqGraphInterface.ExtraNodeInfo(context_after, subgoals_opened,
                                   subgoals_closed, is_proof_completed)
        self._node_infos[new_state] = extra_info
        new_node = LabeledNode(edge.tactic, time_taken, new_state, context_before, self._id2node[edge.frm])
        new_node.time_taken = time_taken
        self._id2node[new_state] = new_node
        self._memoized_outgoing_edges[edge] = new_node
        return new_node

    def context_after(self, node_id) -> ProofContext:
        return self._node_infos[node_id].context_after

    def subgoals_opened(self, node_id):
        return self._node_infos[node_id].subgoals_opened

    def subgoals_closed(self, node_id):
        return self._node_infos[node_id].subgoals_closed

    def is_proof_completed(self, node_id):
        return self._node_infos[node_id].is_proof_completed





class CoqVisitor(TreeTraverseVisitor):
    """
    Visitor for search in coq
    Makes search return SubSearchResult
    """

    class NodeInfo(NamedTuple):
        """
        Additional node info, that CoqVisitor stores for each node
        """
        extra_depth: int
        subgoal_distance_stack: List[int]
        subgoals_closed: int
        subgoals_opened: int
        path: List[LabeledNode]

    def __init__(self,
                 pbar: tqdm,
                 visualization_graph: SearchGraph,
                 args: argparse.Namespace,
                 initial_node_id: int,
                 initial_node_info: NodeInfo
                 ):
        self.pbar = pbar
        self.num_successful_predictions = defaultdict(int)
        self.visualization_graph = visualization_graph
        self.args = args
        self.nodes_info: Dict[int, CoqVisitor.NodeInfo] = {initial_node_id: initial_node_info}
        self.has_unexplored_node: bool = False

    def on_got_edge(self, graph: GraphInterface, frm, edge) -> TraverseVisitorResult:
        """limit search width"""
        if self.num_successful_predictions[frm.node_id] >= self.args.search_width:
            return TraverseVisitorResult(do_break=True)
        return TraverseVisitorResult()

    def on_discover(self, graph: CoqGraphInterface,
                    frm: LabeledNode, discovered: LabeledNode) -> TraverseVisitorResult:
        """
        skip erroneous nodes
        increment num_successful_predictions
        update visualizations
        calculate extra depth and distance_stack
        check:
            Qed --> return Qed
            context in current path --> cancel node
            context is too big --> cancel node
            depth limit --> return, has_unexplored = true

        """
        if discovered is None:  # coq error
            return TraverseVisitorResult(do_skip=True)
        self.num_successful_predictions[frm.node_id] += 1
        self.pbar.update(1)
        self.visualization_graph.mkNode(discovered.prediction,
                                        discovered.context_before,
                                        discovered.previous)

        # Handle stop conitions

        new_distance_stack, new_extra_depth = update_distance_stack(
            self.nodes_info[frm.node_id].extra_depth,
            self.nodes_info[frm.node_id].subgoal_distance_stack,
            self.nodes_info[frm.node_id].subgoals_closed,
            self.nodes_info[frm.node_id].subgoals_opened)

        subgoals_closed = graph.subgoals_closed(discovered.node_id)
        subgoals_opened = graph.subgoals_opened(discovered.node_id)
        context_after = graph.context_after(discovered.node_id)
        discovered_info = CoqVisitor.NodeInfo(new_extra_depth, new_distance_stack, subgoals_closed,
                                              subgoals_opened, self.nodes_info[frm.node_id].path + [discovered])
        self.nodes_info[discovered.node_id] = discovered_info

        depth_limit = self.args.search_depth + new_extra_depth
        if graph.is_proof_completed(discovered.node_id):
            solution = self.visualization_graph.mkQED(discovered)
            return TraverseVisitorResult(do_return=True,
                                         what_return=SubSearchResult(solution, subgoals_closed))
        if contextInPath(context_after, discovered_info.path[1:]):
            if not self.args.count_softfail_predictions:
                self.num_successful_predictions[frm.node_id] -= 1  # I don't like this +1 -1 logic
            self.visualization_graph.setNodeColor(discovered, "orange")
            eprint_cancel(frm.node_id, self.args, "resulting context is in current path")
            return TraverseVisitorResult(do_skip=True)
        if contextIsBig(context_after):
            self.visualization_graph.setNodeColor(discovered, "orange4")
            eprint_cancel(frm.node_id, self.args, "resulting context has too big a goal")
            return TraverseVisitorResult(do_skip=True)
        if len(discovered_info.path) >= depth_limit:
            self.has_unexplored_node = True
            eprint_cancel(frm.node_id, self.args, "we hit the depth limit")
            if subgoals_closed > 0:
                return TraverseVisitorResult(do_return=True,
                                             what_return=SubSearchResult(None, subgoals_closed))
            return TraverseVisitorResult(do_skip=True)

        return TraverseVisitorResult()

    def on_got_result(self, graph: CoqGraphInterface,
                      receiver_node: LabeledNode,
                      sender_node: LabeledNode,
                      result: SubSearchResult,
                      siblings_results: List[SubSearchResult]) -> TraverseVisitorResult:
        """
        if has solution, return it with new_subgoals_closed
        if has closed subgoal, return subgoals_closed
        I don't understand what it really does
        """
        subgoals_opened = graph.subgoals_opened(sender_node.node_id)
        subgoals_closed = graph.subgoals_closed(sender_node.node_id)
        if result.solution or \
                result.solved_subgoals > subgoals_opened:
            new_subgoals_closed = \
                subgoals_closed - subgoals_opened + \
                result.solved_subgoals
            return TraverseVisitorResult(do_return=True,
                                         what_return=SubSearchResult(result.solution,
                                                                     new_subgoals_closed))
        if subgoals_closed > 0:
            return TraverseVisitorResult(do_return=True,
                                         what_return=SubSearchResult(None, subgoals_closed))
        return TraverseVisitorResult()

    def on_exit(self, graph: GraphInterface,
                node_left, all_children_results, stage, suggested_result) -> TraverseVisitorResult:
        # All predictions made no progress
        return TraverseVisitorResult(what_return=SubSearchResult(None, 0), do_return=True)


def dfs_proof_search_with_graph_visitor(lemma_statement: str,
                                        module_name: Optional[str],
                                        coq: serapi_instance.SerapiInstance,
                                        args: argparse.Namespace,
                                        bar_idx: int) -> SearchResult:
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    g = SearchGraph(lemma_name)

    # Run search, and draw some interface
    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1

    with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
                 desc="Proof", disable=(not args.progress),
                 leave=False,
                 position=((bar_idx * 2) + 1),
                 dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        # visitor = CoqVisitor(pbar, [g.start_node], [], 0)
        visitor = CoqVisitor(pbar, g, args, coq.cur_state,
                             CoqVisitor.NodeInfo(0, [], 0, 0, [g.start_node]))
        graph_interface = CoqGraphInterface(coq, args, g.start_node)
        command_list, _ = dfs(LabeledNode("Idk dummy initial pred", 0, coq.cur_state, coq.proof_context, None),
                              graph_interface,
                              visitor)
        pbar.clear()
    module_prefix = escape_lemma_name(module_name)

    if lemma_name == "":
        search_file.unnamed_goal_number += 1
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}"
               f"{search_file.unnamed_goal_number}.svg")
    else:
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}.svg")

    if command_list:
        return SearchResult(SearchStatus.SUCCESS, command_list)
    if visitor.has_unexplored_node:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    return SearchResult(SearchStatus.FAILURE, None)
