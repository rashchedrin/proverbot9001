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
from tree_traverses import dfs_non_recursive_no_hashes, bfs, dfs, best_first_search, \
    TreeTraverseVisitor, BestFirstSearchVisitor, GraphInterface, \
    TraverseVisitorResult
from models.tactic_predictor import Prediction


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
    frm = coq.cur_state
    eprint_cancel(desired_state, args, msg)
    while coq.cur_state != desired_state:
        if coq.cur_state <= 1:
            err = f"Attempt to cancel root, while trying to get from {frm} to {desired_state}, because {msg}"
            raise RuntimeError(err)
        coq.cancel_last()


def delete_duplicate_predictions(predictions_and_certainties: List[Prediction]):
    """
    deletes duplicates, preserves order
    based on
    source: https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    (https://www.peterbe.com/plog/uniqifiers-benchmark)
    """
    seen = set()
    seen_add = seen.add
    return [x for x in predictions_and_certainties
            if not (x.prediction in seen or seen_add(x.prediction))]


def predict_k_tactics(coq: serapi_instance.SerapiInstance, args: argparse.Namespace, k: int) -> List:
    relevant_lemmas = get_relevant_lemmas(args, coq)
    tactic_context_before = TacticContext(relevant_lemmas, coq.prev_tactics, coq.hypotheses, coq.goals)
    predictions_and_certainties = \
        delete_duplicate_predictions(search_file.predictor.predictKTactics(tactic_context_before, k))
    predictions = [prediction.prediction for prediction in predictions_and_certainties]
    certainties = [prediction.certainty for prediction in predictions_and_certainties]
    if coq.use_hammer:
        predictions = add_hammer_commands(predictions)
    # if len(without_duplicates) != len(predictions):
    #     print(predictions)
    return predictions, certainties


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


class Edge(NamedTuple):
    frm: int
    tactic: str
    certainty: float


class CoqGraphNode(NamedTuple):
    """
    Additional node info, that CoqGraphInterface stores for each node
    """
    context_after: ProofContext
    subgoals_opened: int
    subgoals_closed: int
    is_proof_completed: bool
    state_id: int
    vis_node: LabeledNode
    previous_state_id: Optional[int]


class CoqGraphInterface(GraphInterface):
    """
    Interface to Coq as a graph
    """

    def __init__(self,
                 coq: serapi_instance.SerapiInstance,
                 args: argparse.Namespace,
                 vis_graph: SearchGraph,
                 ):
        self._coq = coq
        self._args = args
        self._vis_graph = vis_graph
        self._memoized_outgoing_edges: Dict[int, List[Edge]] = {}
        self._memoized_edge_destinations: Dict[int, Optional[CoqGraphNode]] = {}
        self._state_id_to_node: Dict[int, CoqGraphNode] = {}

        # todo: figure out correct initialization
        root_node = CoqGraphNode(coq.proof_context, None, None, completed_proof(coq), coq.cur_state,
                                 self._vis_graph.start_node, None)
        self._state_id_to_node[root_node.state_id] = root_node
        self.root = root_node

    def _ids_on_path_to_root(self, state_id: int) -> List[int]:
        """
        Returns ids on path to root, in order state_id, ... , root_state_id
        """
        cur_node = self._state_id_to_node[state_id]
        path = [cur_node.state_id]
        while cur_node.previous_state_id is not None:
            path.append(cur_node.previous_state_id)
            cur_node = self._state_id_to_node[cur_node.previous_state_id]
        return path

    def _closest_common_ancestor(self, state_id_first: int, state_id_second: int) -> int:
        path_first: List[int] = list(reversed(self._ids_on_path_to_root(state_id_first)))
        path_second: List[int] = list(reversed(self._ids_on_path_to_root(state_id_second)))
        for i in range(min(len(path_first), len(path_second))):
            if path_first[i] != path_second[i]:
                return path_first[i - 1]
        return path_first[min(len(path_first), len(path_second)) - 1]

    def _commands_and_newtips_from_to(self, id_from: int, id_to: int) -> List[Tuple[str, int]]:
        """
        returns list of commands (from, .. to]
        """
        reversed_commands_and_tips = []
        cur_node = self._state_id_to_node[id_to]
        while cur_node.state_id != id_from:
            reversed_commands_and_tips.append((cur_node.vis_node.prediction, cur_node.state_id))
            cur_node = self._state_id_to_node[cur_node.previous_state_id]
        return list(reversed(reversed_commands_and_tips))

    def _redo_to_state(self, state_id):
        cmds_and_newtips = self._commands_and_newtips_from_to(self._coq.cur_state, state_id)
        for cmd_and_newtip in cmds_and_newtips:
            cmd, newtip = cmd_and_newtip
            # print(state_id, repr(cmd), newtip) # todo: remove
            prev_id = self._state_id_to_node[newtip].previous_state_id
            prev_labeled_node = self._state_id_to_node[prev_id].vis_node
            tryPrediction(self._args, self._coq, cmd, prev_labeled_node, newtip)
            # self._coq.run_stmt(cmd, newtip=newtip)
            assert self._coq.cur_state == newtip

    def _goto_state_fake(self,
                         desired_state_id: int,
                         msg: Optional[str] = None):  # Todo: make real
        cancel_until = self._closest_common_ancestor(self._coq.cur_state, desired_state_id)
        cancel_until_state(self._coq, cancel_until, self._args, msg)
        self._redo_to_state(desired_state_id)

    def get_outgoing_edges(self, node: CoqGraphNode) -> List[Edge]:
        """
        Calls neural network to get predictions
        memoizes to self.memoized_outgoing_edges
        """
        # print(f"Get edges of {node.state_id}")
        if node.state_id in self._memoized_outgoing_edges:
            # print(f"Edges recalled ({len(self._memoized_outgoing_edges[node.state_id])})")
            return self._memoized_outgoing_edges[node.state_id]
        self._goto_state_fake(node.state_id, "get outgoing edges")
        predictions, certainties = predict_k_tactics(self._coq, self._args, self._args.max_attempts)
        edges = [Edge(node.state_id, pred, certainty) for pred, certainty in zip(predictions, certainties)]
        self._memoized_outgoing_edges[node.state_id] = edges
        return edges

    def edge_destination(self, edge: Edge) -> Optional[CoqGraphNode]:
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
        if self._coq.cur_state != edge.frm:
            # print(f"mov {self._coq.cur_state} => {edge.frm} {edge.tactic}", end='')
            self._goto_state_fake(edge.frm, "goto edge source")
        context_before = self._coq.proof_context
        parent_vis_node = self._state_id_to_node[edge.frm].vis_node
        context_after, _, \
        subgoals_closed, subgoals_opened, \
        error, time_taken, new_state = \
            tryPrediction(self._args, self._coq, edge.tactic, parent_vis_node)
        if error:
            # print(f"failed {edge.frm} {edge.tactic}")
            return None
        # print(f"{edge.frm} -{edge.tactic}-> {new_state}")
        is_proof_completed = completed_proof(self._coq)
        new_vis_node = self._vis_graph.mkNode(edge.tactic, context_before, parent_vis_node)
        new_vis_node.time_taken = time_taken
        new_node = CoqGraphNode(context_after, subgoals_opened, subgoals_closed, is_proof_completed,
                                new_state, new_vis_node, edge.frm)
        self._state_id_to_node[new_state] = new_node
        self._memoized_outgoing_edges[edge] = new_node
        assert new_state != edge.frm
        return new_node

    def context_after(self, state_id) -> ProofContext:
        return self._state_id_to_node[state_id].context_after

    def subgoals_opened(self, state_id):
        return self._state_id_to_node[state_id].subgoals_opened

    def subgoals_closed(self, state_id):
        return self._state_id_to_node[state_id].subgoals_closed

    def is_proof_completed(self, state_id):
        return self._state_id_to_node[state_id].is_proof_completed


class CoqVisitor(BestFirstSearchVisitor):
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
        path: List[LabeledNode]

    def __init__(self,
                 pbar: tqdm,
                 vis_graph: SearchGraph,
                 args: argparse.Namespace,
                 initial_state_id: int,
                 ):
        self._pbar = pbar
        self._num_successful_predictions = defaultdict(int)
        self._vis_graph = vis_graph
        self._args = args
        self._nodes_info: Dict[int, CoqVisitor.NodeInfo] = \
            {initial_state_id: CoqVisitor.NodeInfo(0, [], [vis_graph.start_node])}
        self.has_unexplored_node: bool = False
        self._nodes_score: Dict[int, float] = {}

    def on_enter(self, graph: GraphInterface, entered_node) -> TraverseVisitorResult:
        # print(f"Launched from {entered_node.state_id}")
        return super().on_enter(graph, entered_node)

    def on_traveling_edge(self, graph: CoqGraphInterface, frm: CoqGraphNode, edge: Edge) -> TraverseVisitorResult:
        """limit search width"""
        if self._num_successful_predictions[frm.state_id] >= self._args.search_width:
            return TraverseVisitorResult(stop_discovering_edges=True)
        return TraverseVisitorResult()

    def on_discover(self, graph: CoqGraphInterface,
                    frm: CoqGraphNode, discovered: CoqGraphNode) -> TraverseVisitorResult:
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
        self._num_successful_predictions[frm.state_id] += 1
        self._pbar.update(1)

        # Handle stop conitions
        subgoals_opened = discovered.subgoals_opened
        subgoals_closed = discovered.subgoals_closed
        context_after = discovered.context_after

        new_distance_stack, new_extra_depth = update_distance_stack(
            self._nodes_info[frm.state_id].extra_depth,
            self._nodes_info[frm.state_id].subgoal_distance_stack,
            subgoals_closed,
            subgoals_opened)

        discovered_info = CoqVisitor.NodeInfo(new_extra_depth, new_distance_stack,
                                              self._nodes_info[frm.state_id].path + [discovered.vis_node])
        self._nodes_info[discovered.state_id] = discovered_info

        depth_limit = self._args.search_depth + new_extra_depth
        if discovered.is_proof_completed:
            solution = self._vis_graph.mkQED(discovered.vis_node)
            return TraverseVisitorResult(do_return=True,
                                         what_return=SubSearchResult(solution, subgoals_closed))
        if contextInPath(context_after, discovered_info.path[1:]):
            if not self._args.count_softfail_predictions:
                self._num_successful_predictions[frm.state_id] -= 1  # I don't like this +1 -1 logic
            self._vis_graph.setNodeColor(discovered.vis_node, "orange")
            eprint_cancel(frm.vis_node.node_id, self._args, "resulting context is in current path")
            return TraverseVisitorResult(do_skip=True)
        if contextIsBig(context_after):
            self._vis_graph.setNodeColor(discovered.vis_node, "orange4")
            eprint_cancel(frm.vis_node.node_id, self._args, "resulting context has too big a goal")
            return TraverseVisitorResult(do_skip=True)
        current_depth = len(discovered_info.path) - 1
        if current_depth >= depth_limit:
            self.has_unexplored_node = True
            eprint_cancel(frm.vis_node.node_id, self._args, "we hit the depth limit")
            if subgoals_closed > 0:
                return TraverseVisitorResult(do_return=True,
                                             what_return=SubSearchResult(None, subgoals_closed))
            return TraverseVisitorResult(do_skip=True)

        return TraverseVisitorResult()

    def on_got_result(self, graph: CoqGraphInterface,
                      receiver_node: CoqGraphNode,
                      sender_node: CoqGraphNode,
                      result: SubSearchResult,
                      siblings_results: List[SubSearchResult]) -> TraverseVisitorResult:
        """
        if has solution, return it with new_subgoals_closed
        if has closed subgoal, return subgoals_closed
        I don't understand what it really does
        """
        subgoals_opened = sender_node.subgoals_opened
        subgoals_closed = sender_node.subgoals_closed
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

    def on_exit(self, graph: CoqGraphInterface,
                node_left: CoqGraphNode, all_children_results, stage, suggested_result) -> TraverseVisitorResult:
        # All predictions made no progress
        # print(f"Exiting {node_left.state_id} at stage {stage}")
        return TraverseVisitorResult(what_return=SubSearchResult(None, 0), do_return=True)

    def _eval_edge(self, tree: CoqGraphInterface, edge: Edge):
        return edge.certainty

    def edge_picker(self, tree: CoqGraphInterface, leaf_edges: List[Edge]) -> int:
        return max(range(len(leaf_edges)), key=lambda i: self._eval_edge(tree, leaf_edges[i]))


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
        visitor = CoqVisitor(pbar, g, args, coq.cur_state)
        graph_interface = CoqGraphInterface(coq, args, g)
        command_list, _ = dfs(graph_interface.root,
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


def bfs_proof_search_with_graph_visitor(lemma_statement: str,
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
        visitor = CoqVisitor(pbar, g, args, coq.cur_state)
        graph_interface = CoqGraphInterface(coq, args, g)
        command_list, _ = best_first_search(graph_interface.root,
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
