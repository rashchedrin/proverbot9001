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


class CoqVisitor(TreeTraverseVisitor):
    """
    Visitor for search in coq
    Makes search return SubSearchResult
    """

    def __init__(self, max_successful_predictions: int, pbar: tqdm):
        self.max_successful_predictions = max_successful_predictions
        self.pbar = pbar
        self.num_successful_predictions = defaultdict(int)

    def on_enter(self, graph: GraphInterface,
                 entered_node) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_got_edge(self, graph: GraphInterface, frm, edge) -> TraverseVisitorResult:
        if self.num_successful_predictions[frm] >= self.max_successful_predictions:
            return TraverseVisitorResult(do_break=True)
        return TraverseVisitorResult()

    def on_discover(self, graph: GraphInterface,
                    frm: LabeledNode, discovered: LabeledNode) -> TraverseVisitorResult:
        if discovered is None:
            return TraverseVisitorResult(do_skip=True)
        self.num_successful_predictions += 1
        self.pbar.update(1)
        return TraverseVisitorResult()

    def on_got_result(self, graph: GraphInterface,
                      parent, node, result, all_children_results) -> TraverseVisitorResult:
        return TraverseVisitorResult()

    def on_exit(self, graph: GraphInterface,
                node_left, all_children_results, stage, suggested_result) -> TraverseVisitorResult:
        return TraverseVisitorResult()


class Edge(NamedTuple):
    frm: int
    tactic: str


class CoqGraphInterface(GraphInterface):
    """
    Interface to Coq as a graph
    """

    def __init__(self,
                 coq: serapi_instance.SerapiInstance,
                 args: argparse.Namespace,
                 ):
        self.coq = coq
        self.args = args
        self.contexts_after: Dict[int:ProofContext] = {}
        self.memoized_outgoing_edges: Dict[int: List[Edge]] = {}
        self.memoized_edge_destinations: Dict[int: Optional[LabeledNode]] = {}
        self.id2node: Dict[int: LabeledNode] = {}

    def get_outgoing_edges(self, node: LabeledNode) -> List[Edge]:
        """
        Calls neural network to get predictions
        memoizes to self.memoized_outgoing_edges
        """
        if node.node_id in self.memoized_outgoing_edges:
            return self.memoized_outgoing_edges[node.node_id]
        goto_state_fake(self.coq, node.node_id, self.args)
        predictions = predict_k_tactics(self.coq, self.args, self.args.max_attempts)
        edges = [Edge(node.node_id, pred) for pred in predictions]
        self.memoized_outgoing_edges[node.node_id] = edges
        return edges

    def edge_destination(self, edge: Edge) -> Optional[LabeledNode]:
        """
        Creates new node: sends commands to Coq to get it
        Memoizes to self.memoized_edge_destinations
        adds new node to self.id2node
        returns None on Coq error
        """
        if edge in self.memoized_outgoing_edges:
            return self.memoized_outgoing_edges[edge]
        goto_state_fake(self.coq, edge.frm, self.args)
        parent_node = self.id2node[edge.frm].previous
        context_after, _, \
        subgoals_closed, subgoals_opened, \
        error, time_taken, new_state = \
            tryPrediction(self.args, self.coq, edge.tactic, parent_node)
        if error:
            return None
        context_before = self.contexts_after[parent_node.node_id]
        # new_node = LabeledNode(edge.tactic, time_taken, new_state, context_before, self.id2node[edge.frm])
        new_node = g.mkNode()
        new_node.time_taken = time_taken
        self.id2node[new_state] = new_node
        self.memoized_outgoing_edges[edge] = new_node
        return new_node


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
        visitor = CoqVisitor(pbar, [g.start_node], [], 0)
        graph_interface = CoqGraphInterface(coq, args)
        command_list, _ = dfs(coq.cur_state,
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
