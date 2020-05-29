"""
DFS search strategy for Proverbot9001
"""
import argparse
import sys
from typing import (List, Optional, Dict, Any, Set, Tuple, NamedTuple)
from dataclasses import dataclass
from tqdm import tqdm

import search_file
import serapi_instance
from search_file import (SearchResult, SubSearchResult, SearchGraph, LabeledNode, TacticContext,
                         tryPrediction, completed_proof, contextInPath, contextIsBig, numNodesInTree,
                         SearchStatus, TqdmSpy)
from util import (eprint, escape_lemma_name,
                  mybarfmt)
from tree_traverses import dfs, TreeTraverseVisitor


def get_relevant_lemmas(args, coq):
    if args.relevant_lemmas == "local":
        return coq.local_lemmas[:-1]
    if args.relevant_lemmas == "hammer":
        return coq.get_hammer_premises()
    if args.relevant_lemmas == "searchabout":
        return coq.get_lemmas_about_head()
    raise RuntimeError(f"Unsupported relevant_lemmas type {args.relevant_lemmas}")


# def dfs_proof_search_with_graph(lemma_statement: str,
#                                 module_name: Optional[str],
#                                 coq: serapi_instance.SerapiInstance,
#                                 args: argparse.Namespace,
#                                 bar_idx: int) \
#         -> SearchResult:
#     lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
#     g = SearchGraph(lemma_name)
#
#     def cleanupSearch(num_stmts: int, msg: Optional[str] = None):
#         if msg:
#             eprint(f"Cancelling {num_stmts} statements "
#                    f"because {msg}.", guard=args.verbose >= 2)
#         for _ in range(num_stmts):
#             coq.cancel_last()
#
#     hasUnexploredNode = False
#
#     def search(pbar: tqdm, current_path: List[LabeledNode],
#                subgoal_distance_stack: List[int],
#                extra_depth: int) -> SubSearchResult:
#         nonlocal hasUnexploredNode
#         if args.relevant_lemmas == "local":
#             relevant_lemmas = coq.local_lemmas[:-1]
#         elif args.relevant_lemmas == "hammer":
#             relevant_lemmas = coq.get_hammer_premises()
#         elif args.relevant_lemmas == "searchabout":
#             relevant_lemmas = coq.get_lemmas_about_head()
#         else:
#             assert False, args.relevant_lemmas
#         tactic_context_before = TacticContext(relevant_lemmas,
#                                               coq.prev_tactics,
#                                               coq.hypotheses,
#                                               coq.goals)
#         predictions = [prediction.prediction for prediction in
#                        search_file.predictor.predictKTactics(tactic_context_before, args.max_attempts)]
#         proof_context_before = coq.proof_context
#         if coq.use_hammer:
#             predictions = [prediction + "; try hammer." for prediction in predictions]
#         num_successful_predictions = 0
#         for prediction_idx, prediction in enumerate(predictions):
#             if num_successful_predictions >= args.search_width:
#                 break
#             try:
#                 context_after, num_stmts, \
#                 subgoals_closed, subgoals_opened, \
#                 error, time_taken = \
#                     tryPrediction(args, coq, prediction, current_path[-1])
#                 if error:
#                     if args.count_failing_predictions:
#                         num_successful_predictions += 1
#                     continue
#                 num_successful_predictions += 1
#                 pbar.update(1)
#                 assert pbar.n > 0
#
#                 predictionNode = g.mkNode(prediction, proof_context_before,
#                                           current_path[-1])
#                 predictionNode.time_taken = time_taken
#
#                 #### 1.
#                 if subgoal_distance_stack:
#                     new_distance_stack = (subgoal_distance_stack[:-1] +
#                                           [subgoal_distance_stack[-1] + 1])
#                 else:
#                     new_distance_stack = []
#
#                 #### 2.
#                 new_extra_depth = extra_depth
#                 for _ in range(subgoals_closed):
#                     closed_goal_distance = new_distance_stack.pop()
#                     new_extra_depth += closed_goal_distance
#
#                 #### 3.
#                 new_distance_stack += [0] * subgoals_opened
#
#                 #############
#                 if completed_proof(coq):
#                     solution = g.mkQED(predictionNode)
#                     return SubSearchResult(solution, subgoals_closed)
#                 elif contextInPath(context_after, current_path[1:] + [predictionNode]):
#                     if not args.count_softfail_predictions:
#                         num_successful_predictions -= 1
#                     g.setNodeColor(predictionNode, "orange")
#                     cleanupSearch(num_stmts, "resulting context is in current path")
#                 elif contextIsBig(context_after):
#                     g.setNodeColor(predictionNode, "orange4")
#                     cleanupSearch(num_stmts, "resulting context has too big a goal")
#                 elif len(current_path) < args.search_depth + new_extra_depth:
#                     sub_search_result = search(pbar, current_path + [predictionNode],
#                                                new_distance_stack, new_extra_depth)
#                     cleanupSearch(num_stmts, "we finished subsearch")
#                     if sub_search_result.solution or \
#                             sub_search_result.solved_subgoals > subgoals_opened:
#                         new_subgoals_closed = \
#                             subgoals_closed + \
#                             sub_search_result.solved_subgoals - \
#                             subgoals_opened
#                         return SubSearchResult(sub_search_result.solution,
#                                                new_subgoals_closed)
#                     if subgoals_closed > 0:
#                         return SubSearchResult(None, subgoals_closed)
#                 else:
#                     hasUnexploredNode = True
#                     cleanupSearch(num_stmts, "we hit the depth limit")
#                     if subgoals_closed > 0:
#                         depth = (args.search_depth + new_extra_depth + 1) \
#                                 - len(current_path)
#                         return SubSearchResult(None, subgoals_closed)
#             except (serapi_instance.CoqExn, serapi_instance.TimeoutError,
#                     serapi_instance.OverflowError, serapi_instance.ParseError,
#                     serapi_instance.UnrecognizedError):
#                 continue
#             except serapi_instance.NoSuchGoalError:
#                 raise
#         return SubSearchResult(None, 0)
#
#     total_nodes = numNodesInTree(args.search_width,
#                                  args.search_depth + 2) - 1
#     with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
#                  desc="Proof", disable=(not args.progress),
#                  leave=False,
#                  position=((bar_idx * 2) + 1),
#                  dynamic_ncols=True, bar_format=mybarfmt) as pbar:
#         command_list, _ = search(pbar, [g.start_node], [], 0)
#         pbar.clear()
#     module_prefix = escape_lemma_name(module_name)
#     if lemma_name == "":
#         search_file.unnamed_goal_number += 1
#         g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}"
#                f"{search_file.unnamed_goal_number}.svg")
#     else:
#         g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}.svg")
#     if command_list:
#         return SearchResult(SearchStatus.SUCCESS, command_list)
#     elif hasUnexploredNode:
#         return SearchResult(SearchStatus.INCOMPLETE, None)
#     else:
#         return SearchResult(SearchStatus.FAILURE, None)


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
                    msg: Optional[str] = None):
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


def dfs_proof_search_with_graph(lemma_statement: str,
                                module_name: Optional[str],
                                coq: serapi_instance.SerapiInstance,
                                args: argparse.Namespace,
                                bar_idx: int) -> SearchResult:
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    g = SearchGraph(lemma_name)
    hasUnexploredNode = False

    def search(pbar: tqdm,
               current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int
               ) -> SubSearchResult:

        nonlocal hasUnexploredNode
        predictions = predict_k_tactics(coq, args, args.max_attempts)
        proof_context_before = coq.proof_context
        num_successful_predictions = 0
        for prediction in predictions:
            if num_successful_predictions >= args.search_width:
                break
            try:
                context_after, num_stmts, \
                subgoals_closed, subgoals_opened, \
                error, time_taken = \
                    tryPrediction(args, coq, prediction, current_path[-1])
                if error:
                    if args.count_failing_predictions:
                        num_successful_predictions += 1
                    continue
                num_successful_predictions += 1
                pbar.update(1)

                prediction_node = g.mkNode(prediction, proof_context_before, current_path[-1])
                prediction_node.time_taken = time_taken

                # Handle stop conitions
                new_distance_stack, new_extra_depth = update_distance_stack(extra_depth, subgoal_distance_stack,
                                                                            subgoals_closed, subgoals_opened)
                depth_limit = args.search_depth + new_extra_depth
                if completed_proof(coq):
                    solution = g.mkQED(prediction_node)
                    return SubSearchResult(solution, subgoals_closed)
                if contextInPath(context_after, current_path[1:] + [prediction_node]):
                    if not args.count_softfail_predictions:
                        num_successful_predictions -= 1
                    g.setNodeColor(prediction_node, "orange")
                    cancel_last_statements(coq, num_stmts, args, "resulting context is in current path")
                    continue
                if contextIsBig(context_after):
                    g.setNodeColor(prediction_node, "orange4")
                    cancel_last_statements(coq, num_stmts, args, "resulting context has too big a goal")
                    continue
                if len(current_path) >= depth_limit:
                    hasUnexploredNode = True
                    cancel_last_statements(coq, num_stmts, args, "we hit the depth limit")
                    if subgoals_closed > 0:
                        return SubSearchResult(None, subgoals_closed)
                    continue

                # Run recursion
                sub_search_result = search(pbar, current_path + [prediction_node],
                                           new_distance_stack, new_extra_depth)
                cancel_last_statements(coq, num_stmts, args, "we finished subsearch")
                if sub_search_result.solution or \
                        sub_search_result.solved_subgoals > subgoals_opened:
                    new_subgoals_closed = \
                        subgoals_closed + \
                        sub_search_result.solved_subgoals - \
                        subgoals_opened
                    return SubSearchResult(sub_search_result.solution,
                                           new_subgoals_closed)
                if subgoals_closed > 0:
                    return SubSearchResult(None, subgoals_closed)

            except (serapi_instance.CoqExn, serapi_instance.TimeoutError,
                    serapi_instance.OverflowError, serapi_instance.ParseError,
                    serapi_instance.UnrecognizedError):
                continue
            except serapi_instance.NoSuchGoalError:
                raise
        return SubSearchResult(None, 0)  # ran out of predictions.

    # Run search, and draw some interface
    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1

    with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
                 desc="Proof", disable=(not args.progress),
                 leave=False,
                 position=((bar_idx * 2) + 1),
                 dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        command_list, _ = search(pbar, [g.start_node], [], 0)
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
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    return SearchResult(SearchStatus.FAILURE, None)


def manage_returned_result(sub_search_result, subgoals_closed, subgoals_opened):
    return_result = None
    if sub_search_result.solution or sub_search_result.solved_subgoals > subgoals_opened:
        new_subgoals_closed = sub_search_result.solved_subgoals + subgoals_closed - subgoals_opened  # what is it?
        return_result = SubSearchResult(sub_search_result.solution, new_subgoals_closed)
    elif subgoals_closed > 0:  # what does it mean?
        return_result = SubSearchResult(None, subgoals_closed)
    return return_result


def dfs_proof_search_with_graph_refactored(lemma_statement: str,
                                           module_name: Optional[str],
                                           coq: serapi_instance.SerapiInstance,
                                           args: argparse.Namespace,
                                           bar_idx: int) -> SearchResult:
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    g = SearchGraph(lemma_name)
    hasUnexploredNode = False
    parent_states = {}

    def search(pbar: tqdm,
               current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int,
               search_origin_state: int,
               ) -> SubSearchResult:

        nonlocal hasUnexploredNode
        goto_state_fake(coq, search_origin_state, args)
        predictions = predict_k_tactics(coq, args, args.max_attempts)
        proof_context_before = coq.proof_context
        num_successful_predictions = 0
        for prediction in predictions:
            if num_successful_predictions >= args.search_width:
                break
            try:
                goto_state_fake(coq, search_origin_state, args)
                context_after, _, \
                subgoals_closed, subgoals_opened, \
                error, time_taken, new_state = \
                    tryPrediction(args, coq, prediction, current_path[-1])
                if error:
                    if args.count_failing_predictions:
                        num_successful_predictions += 1
                    continue  # try next prediction
                parent_states[new_state] = search_origin_state
                prediction_node = g.mkNode(prediction, proof_context_before, current_path[-1])
                prediction_node.time_taken = time_taken
                num_successful_predictions += 1
                pbar.update(1)
                # Handle stop conditions
                new_distance_stack, new_extra_depth = update_distance_stack(extra_depth, subgoal_distance_stack,
                                                                            subgoals_closed, subgoals_opened)
                cancel_reason, num_successful_predictions, return_result = \
                    manage_stop_conditions(context_after, coq, current_path, new_extra_depth,
                                           num_successful_predictions, prediction_node, subgoals_closed)
                if cancel_reason is not None:
                    eprint_cancel(search_origin_state, args, cancel_reason)
                if return_result is not None:
                    return return_result  # Qed, or depth limit => (None, subgoals_closed), else: continue
                if cancel_reason is not None:
                    continue
                # Run recursion
                sub_search_result = search(pbar,
                                           current_path=current_path + [prediction_node],
                                           subgoal_distance_stack=new_distance_stack,
                                           extra_depth=new_extra_depth,
                                           search_origin_state=new_state)
                # manage returned result
                return_result = manage_returned_result(sub_search_result, subgoals_closed, subgoals_opened)
                if return_result is not None:
                    return return_result  # solution, or (None, subgoals_closed > 0), else contunue
            except (serapi_instance.CoqExn, serapi_instance.TimeoutError,
                    serapi_instance.OverflowError, serapi_instance.ParseError,
                    serapi_instance.UnrecognizedError):
                continue  # try next prediction
        return SubSearchResult(None, 0)  # ran out of predictions.

    def manage_stop_conditions(context_after, coq, current_path, new_extra_depth, num_successful_predictions,
                               prediction_node, subgoals_closed):
        nonlocal hasUnexploredNode
        cancel_reason = None
        return_result = None
        depth_limit = args.search_depth + new_extra_depth
        if completed_proof(coq):
            solution = g.mkQED(prediction_node)
            return_result = SubSearchResult(solution, subgoals_closed)
        elif contextInPath(context_after, current_path[1:] + [prediction_node]):
            if not args.count_softfail_predictions:
                num_successful_predictions -= 1
            g.setNodeColor(prediction_node, "orange")
            cancel_reason = "resulting context is in current path"
        elif contextIsBig(context_after):
            g.setNodeColor(prediction_node, "orange4")
            cancel_reason = "resulting context has too big a goal"
        elif len(current_path) >= depth_limit:
            hasUnexploredNode = True
            cancel_reason = "we hit the depth limit"
            if subgoals_closed > 0:
                return_result = SubSearchResult(None, subgoals_closed)
        return cancel_reason, num_successful_predictions, return_result

    # Run search, and draw some interface
    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1

    with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
                 desc="Proof", disable=(not args.progress),
                 leave=False,
                 position=((bar_idx * 2) + 1),
                 dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        command_list, _ = search(pbar, [g.start_node], [], 0, coq.cur_state)
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
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    return SearchResult(SearchStatus.FAILURE, None)
