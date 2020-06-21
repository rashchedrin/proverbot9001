#!/usr/bin/env python3
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################
import argparse
import os
import sys
import re
import datetime
import time
import csv
import traceback
import pygraphviz as pgv
from typing import (List, Tuple, NamedTuple, Optional, Dict,
                    Union, Iterator, Callable, Iterable)

from models.tactic_predictor import TacticPredictor
from predict_tactic import (static_predictors, loadPredictorByFile,
                            loadPredictorByName)
import serapi_instance
from serapi_instance import ProofContext, Obligation, SerapiInstance

import linearize_semicolons
# import syntax
from format import TacticContext
from util import (unwrap, eprint, escape_filename, escape_lemma_name,
                  mybarfmt, split_by_char_outside_matching, nostderr)
import itertools
from dataclasses import dataclass
from enum import Enum, auto

from tqdm import tqdm
from yattag import Doc
from pathlib_revised import Path2
import logger

Tag = Callable[..., Doc.Tag]
Text = Callable[..., None]
Line = Callable[..., None]

details_css = "details.css"
details_javascript = "search-details.js"


class ReportStats(NamedTuple):
    filename: str
    num_proofs: int
    num_proofs_failed: int
    num_proofs_completed: int


class SearchStatus(Enum):
    SUCCESS = auto()
    INCOMPLETE = auto()
    FAILURE = auto()


class VernacBlock(NamedTuple):
    commands: List[str]


class TacticInteraction(NamedTuple):
    tactic: str
    context_before: ProofContext


class ProofBlock(NamedTuple):
    lemma_statement: str
    module: Optional[str]
    status: SearchStatus
    predicted_tactics: List[TacticInteraction]
    original_tactics: List[TacticInteraction]


class ArgsMismatchException(Exception):
    pass


class SourceChangedException(Exception):
    pass

@dataclass(init=True)
class LabeledNode:
    prediction : str
    time_taken : Optional[float]
    node_id : int
    context_before : ProofContext
    previous : Optional["LabeledNode"]
class SearchGraph:
    __graph : pgv.AGraph
    __next_node_id : int
    start_node : LabeledNode
    def __init__(self, lemma_name : str) -> None:
        self.__graph = pgv.AGraph(directed=True)
        self.__next_node_id = 0
        self.start_node = self.mkNode(lemma_name, ProofContext([],[],[],[]), None)
        self.start_node.time_taken = 0.0
        pass
    def addPredictions(self, src : LabeledNode, context_before : ProofContext,
                       predictions : List[str]) -> List[LabeledNode]:
        return [self.mkNode(pred, context_before, src) for pred in predictions]

    def mkNode(self, prediction : str, context_before : ProofContext,
               previous_node : Optional[LabeledNode],
               **kwargs) -> LabeledNode:
        self.__graph.add_node(self.__next_node_id, label=prediction, **kwargs)
        self.__next_node_id += 1
        newNode = LabeledNode(prediction, None, self.__next_node_id-1,
                              context_before, previous_node)
        if previous_node:
            self.__graph.add_edge(previous_node.node_id, newNode.node_id, **kwargs)
        return newNode

    def mkQED(self, predictionNode : LabeledNode):
        qedNode = self.mkNode("QED", ProofContext([],[],[],[]),
                              predictionNode,
                              fillcolor="green", style="filled")
        cur_node = predictionNode
        cur_path = []
        while cur_node != self.start_node:
            self.setNodeColor(cur_node, "palegreen1")
            cur_path.append(cur_node)
            assert cur_node.previous
            cur_node = cur_node.previous
        return [TacticInteraction(n.prediction, n.context_before)
                for n in reversed(cur_path)]
        pass
    def setNodeColor(self, node : LabeledNode, color : str) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"
    def draw(self, filename : str) -> None:
        with nostderr():
            self.__graph.draw(filename, prog="dot")
class SubSearchResult (NamedTuple):
    solution : Optional[List[TacticInteraction]]
    solved_subgoals : int


DocumentBlock = Union[VernacBlock, ProofBlock]

predictor: TacticPredictor
unnamed_goal_number: int


def main(arg_list: List[str], bar_idx: int) -> None:
    sys.setrecursionlimit(4500)
    global predictor

    args, parser = parse_arguments(arg_list)
    predictor = get_predictor(parser, args)
    base = Path2(os.path.dirname(os.path.abspath(__file__)))
    coqargs = ["sertop", "--implicit"]

    try:
        with open(args.prelude + "/_CoqProject", 'r') as includesfile:
            includes = includesfile.read()
    except FileNotFoundError:
        eprint("Didn't find a _CoqProject file in prelude dir")
        includes = ""
    if not args.output_dir.exists():
        args.output_dir.makedirs()

    for filename in [details_css, details_javascript]:
        destpath = args.output_dir / filename
        if not destpath.exists():
            srcpath = base.parent / 'reports' / filename
            srcpath.copyfile(destpath)
    search_file(args, coqargs, includes, predictor, bar_idx)


def parse_arguments(args_list: List[str]) -> Tuple[argparse.Namespace,
                                                   argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description=
        "Produce an html report from attempting "
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".")
    parser.add_argument("--output", "-o", dest="output_dir",
                        help="output data folder name",
                        default="search-report",
                        type=Path2)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--read-progress", "-p",
                        help="show progress of reading the file",
                        action='store_true')
    parser.add_argument("--hardfail", "-f",
                        help="fail when hitting a coq anomaly",
                        action='store_true')
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default=None)
    parser.add_argument('--weightsfile', default=None, type=Path2)
    parser.add_argument('--predictor', choices=list(static_predictors.keys()),
                        default=None)
    parser.add_argument("--no-truncate_semicolons", dest="truncate_semicolons",
                        action='store_false')
    parser.add_argument("--search-width", dest="search_width", type=int,
                        default=5)
    parser.add_argument("--max-attempts", dest="max_attempts", type=int,
                        default=10)
    parser.add_argument("--search-depth", dest="search_depth", type=int,
                        default=6)
    parser.add_argument("--no-resume", dest="resume", action='store_false')
    parser.add_argument("--overwrite-mismatch", dest="overwrite_mismatch",
                        action='store_true')
    parser.add_argument("--max-print-term", dest="max_print_term", type=int,
                        default=None)
    parser.add_argument("--max-print-hyps", dest="max_print_hyps", type=int,
                        default=None)
    parser.add_argument("--max-print-subgoals", dest="max_print_subgoals",
                        type=int, default=2)
    parser.add_argument("--max-proof-time", dest="max_proof_time",
                        type=float, default=300)
    parser.add_argument("--max-lemma-proof-search-time", dest="max_lemma_proof_search_time",
                        help="Time limit for lemma proof search. Doesn't work for OldDFS",
                        type=float, default=9999999999)
    parser.add_argument("--max-tactic-time", type=float, default=2)
    parser.add_argument("--linearize", action='store_true')
    parser.add_argument("--proof-times", default=None, type=Path2)
    parser.add_argument('filename', help="proof file name (*.v)", type=Path2)
    parser.add_argument("--use-hammer",
                        help="Use Hammer tactic after every predicted tactic",
                        action='store_const', const=True, default=False)
    parser.add_argument('--no-check-consistent', action='store_false',
                        dest='check_consistent')
    parser.add_argument('--count-failing-predictions', action='store_true',
                        dest="count_failing_predictions")
    parser.add_argument('--count-softfail-predictions', action='store_true',
                        dest="count_softfail_predictions")
    parser.add_argument("--relevant-lemmas", dest="relevant_lemmas",
                        choices=['local', 'hammer', 'searchabout'],
                        default='local')
    parser.add_argument("--command-limit", type=int, default=None)
    parser.add_argument("--proof", default=None)
    parser.add_argument("--log-anomalies", type=Path2, default=None)
    parser.add_argument('--traverse-method', choices=["BestFS", "DFS", "OldDFS", "BFS"],
                        default="BestFS", dest='traverse_method')
    parser.add_argument('--bestfs-edge-scoring-fun', choices=["certainty",
                                                              "product_certainty",
                                                              "dfs_then_product_certainty",
                                                              "BestFSyDFS"],
                        default="certainty", dest='bestfs_edge_scoring_fun')
    parser.add_argument("--bestfs-edge-scoring-temperature", dest="bestfs_edge_scoring_temperature",
                        type=float, default=1.0)
    parser.add_argument('--dont-skip-visited', dest='skip_visited', action='store_false')
    parser.set_defaults(skip_visited=True)
    parser.add_argument('--experiment-tag',
                        default="V_01", dest='experiment_tag')
    parser.add_argument('--cur-node-bonus', dest='cur_node_bonus', type=int, default=1.1)
    parser.add_argument('--cur-node-bonus-type', choices=["additive", "multiplicative", "none"],
                        default="none", dest='cur_node_bonus_type')
    parser.add_argument("--certainty-bias", type=float, dest='certainty_bias', default='0.0')
    parser.add_argument("--coef-extra-depth-bonus", type=float, dest='coef_extra_depth_bonus', default='0.0')
    known_args, unknown_args = parser.parse_known_args(args_list)
    return known_args, parser


def get_predictor(parser: argparse.ArgumentParser,
                  args: argparse.Namespace) -> TacticPredictor:
    predictor: TacticPredictor
    if args.weightsfile:
        predictor = loadPredictorByFile(args.weightsfile)
    elif args.predictor:
        predictor = loadPredictorByName(args.predictor)
    else:
        print("You must specify either --weightsfile or --predictor!")
        parser.print_help()
        sys.exit(1)
    return predictor


def reset_times(args: argparse.Namespace):
    if args.proof_times:
        with args.proof_times.open('w'):
            pass


def append_time(args: argparse.Namespace, action: str, seconds: float):
    if args.proof_times:
        with args.proof_times.open('a') as f:
            f.write(f"{action}: {datetime.timedelta(seconds=seconds)}\n")


def search_file(args: argparse.Namespace, coqargs: List[str],
                includes: str, predictor: TacticPredictor,
                bar_idx: int) -> None:
    global unnamed_goal_number
    unnamed_goal_number = 0
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    commands_run: List[str] = []
    blocks_out: List[DocumentBlock] = []
    commands_caught_up = 0
    lemmas_to_skip: List[str] = []
    lemmas_to_proof = []
    if args.resume:
        try:
            check_csv_args(args, args.filename)
            with tqdm(total=1, unit="cmd", file=sys.stdout,
                      desc=args.filename.name + " (Resumed)",
                      disable=(not args.progress),
                      leave=True,
                      position=(bar_idx * 2),
                      dynamic_ncols=True, bar_format=mybarfmt) as pbar:
                pbar.update(1)
            if not args.progress:
                print(f"Resumed {str(args.filename)} from existing state")
            return
        except FileNotFoundError:
            reset_times(args)
            pass
        except ArgsMismatchException as e:
            if not args.progress:
                eprint(f"Arguments in csv for {str(args.filename)} "
                       f"didn't match current arguments! {e} "
                       f"Overwriting (interrupt to cancel).")

    if args.linearize:
        commands_in = linearize_semicolons.get_linearized(
            args, coqargs, includes, bar_idx, str(args.filename))
    else:
        commands_in = serapi_instance.load_commands_preserve(
            args, bar_idx, args.prelude / args.filename)
    num_commands_total = len(commands_in)
    lemma_statement = ""

    # Run vernacular until the next proof (or end of file)
    def run_to_next_proof(coq: serapi_instance.SerapiInstance, pbar: tqdm) \
            -> str:
        nonlocal commands_run
        nonlocal commands_in
        nonlocal blocks_out
        vernacs: List[str] = []
        assert not coq.proof_context
        starttime = time.time()
        while not coq.proof_context and len(commands_in) > 0:
            next_in_command = commands_in.pop(0)
            # Longer timeout for vernac stuff (especially requires)
            coq.run_stmt(next_in_command, timeout=60)
            if not coq.proof_context:
                vernacs.append(next_in_command)
                pbar.update(1)
        append_time(args, "vernac", time.time() - starttime)
        if len(vernacs) > 0:
            blocks_out.append(VernacBlock(vernacs))
            commands_run += vernacs
            append_to_solution_vfile(args.output_dir, args.filename, vernacs)
        return next_in_command

    def run_to_next_vernac(coq: serapi_instance.SerapiInstance,
                           pbar: tqdm,
                           initial_full_context: ProofContext,
                           lemma_statement: str) -> List[TacticInteraction]:
        nonlocal commands_run
        nonlocal commands_in
        coq.run_stmt(lemma_statement)
        original_tactics: List[TacticInteraction] = []
        lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
        try:
            starttime = time.time()
            while coq.proof_context is not None:
                next_in_command = commands_in.pop(0)
                original_tactics.append(
                    TacticInteraction(next_in_command,
                                      coq.proof_context
                                      or ProofContext([], [], [], [])))
                coq.run_stmt(next_in_command)
                pbar.update(1)
            body_tactics = [t.tactic for t in original_tactics]
            if next_in_command.strip() == "Defined.":
                append_to_solution_vfile(args.output_dir, args.filename,
                                         [f"Reset {lemma_name}.",
                                          lemma_statement] + body_tactics)
            commands_run.append(lemma_statement)
            commands_run += body_tactics
            append_time(args, "Orig: " + lemma_name, time.time() - starttime)
        except serapi_instance.SerapiException:
            commands_in = [lemma_statement] + \
                [t.tactic for t in original_tactics] \
                + commands_in
            raise
        return original_tactics

    def add_proof_block(coq: serapi_instance.SerapiInstance,
                        status: SearchStatus,
                        solution: Optional[List[TacticInteraction]],
                        initial_full_context: ProofContext,
                        original_tactics: List[TacticInteraction]) -> None:
        nonlocal num_proofs_failed
        nonlocal num_proofs_completed
        nonlocal blocks_out
        empty_context = ProofContext([], [], [], [])
        # Append the proof data
        if solution:
            num_proofs_completed += 1
            blocks_out.append(ProofBlock(
                lemma_statement, coq.module_prefix, status,
                [TacticInteraction("Proof.",
                                   initial_full_context)] +
                solution +
                [TacticInteraction("Qed.", empty_context)],
                original_tactics))
        else:
            blocks_out.append(ProofBlock(
                lemma_statement, coq.module_prefix, status,
                [TacticInteraction("Proof.",
                                   initial_full_context),
                 TacticInteraction("Admitted.",
                                   initial_full_context)],
                original_tactics))

    if not args.progress:
        print("Loaded {} commands for file {}".format(len(commands_in),
                                                      args.filename))
    with tqdm(total=num_commands_total, unit="cmd", file=sys.stdout,
              desc=args.filename.name,
              disable=(not args.progress),
              leave=True,
              position=(bar_idx * 2),
              dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        while len(commands_in) > 0:
            try:
                # print("Starting a coq instance...")
                with serapi_instance.SerapiContext(
                        coqargs,
                        serapi_instance.get_module_from_filename(
                            args.filename),
                        includes, args.prelude, use_hammer=args.use_hammer
                ) as coq:
                    coq.verbose = args.verbose
                    try_run_prelude(args, coq)
                    if args.progress:
                        pbar.reset()
                    for command in commands_run:
                        pbar.update(1)
                        coq.run_stmt(command)
                    if args.resume and len(commands_run) == 0:
                        model_name = dict(predictor.getOptions())["predictor"]
                        try:
                            commands_run, commands_in, blocks_out, \
                               num_proofs, num_proofs_failed, \
                               num_proofs_completed, \
                               num_original_commands_run = \
                                   replay_solution_vfile(args, coq, model_name,
                                                         args.filename,
                                                         commands_in,
                                                         bar_idx)
                            pbar.update(num_original_commands_run)
                        except FileNotFoundError:
                            make_new_solution_vfile(args, model_name,
                                                    args.filename)
                            pass
                        except (ArgsMismatchException,
                                SourceChangedException) as e:
                            eprint(f"Arguments in solution vfile for {str(args.filename)} "
                                   f"didn't match current arguments, or sources mismatch! "
                                   f"{e}")
                            if args.overwrite_mismatch:
                                eprint("Overwriting.")
                                make_new_solution_vfile(args, model_name,
                                                        args.filename)
                                raise serapi_instance.CoqAnomaly("Replaying")
                            else:
                                raise SourceChangedException

                    if len(commands_run) > 0 and args.verbose:
                        eprint("Caught up with commands:\n{}\n...\n{}"
                               .format(commands_run[0].strip(),
                                       commands_run[-1].strip()))
                    while len(commands_in) > 0:
                        lemma_statement = run_to_next_proof(coq, pbar)
                        if len(commands_in) == 0:
                            break
                        if "Fixpoint" in lemma_statement:
                            coq.cancel_last()
                            original_tactics = run_to_next_vernac(coq, pbar, coq.proof_context,
                                                                  lemma_statement)
                            original_commands = [lemma_statement] + [tac.tactic for tac in original_tactics]
                            append_to_solution_vfile(args.output_dir, args.filename,
                                                     [lemma_statement, "Proof.", "Admitted.",
                                                      "Reset " + serapi_instance.lemma_name_from_statement(lemma_statement)
                                                      + "."])
                            append_to_solution_vfile(args.output_dir, args.filename,
                                                     original_commands)
                            blocks_out.append(VernacBlock(original_commands))
                            continue
                        # Get beginning of next proof
                        num_proofs += 1
                        initial_context = coq.proof_context
                        # Try to search
                        if lemma_statement in lemmas_to_skip or \
                           (args.proof and
                            serapi_instance.lemma_name_from_statement(lemma_statement)
                            != args.proof) \
                                or (lemmas_to_proof and (serapi_instance.lemma_name_from_statement(lemma_statement) not in lemmas_to_proof)):
                            print("skipping " + serapi_instance.lemma_name_from_statement(lemma_statement))
                            search_status = SearchStatus.FAILURE
                            tactic_solution : Optional[List[TacticInteraction]] = []
                        else:
                            starttime = time.time()
                            search_status, tactic_solution = \
                                attempt_search(args, lemma_statement,
                                               coq.module_prefix,
                                               coq, bar_idx)
                            append_time(args,
                                        serapi_instance.
                                        lemma_name_from_statement(lemma_statement),
                                        time.time() - starttime)
                        lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
                        if coq.proof_context:
                            coq.run_stmt(f"Admitted.")
                        if lemma_name:
                            coq.run_stmt(f"Reset {lemma_name}.")
                        else:
                            coq.run_stmt(f"Back.")
                        if tactic_solution:
                            append_to_solution_vfile(args.output_dir, args.filename,
                                                     [lemma_statement, "Proof."] +
                                                     [tac.tactic for tac in tactic_solution]
                                                     + ["Qed."])
                        else:
                            let_match = re.match("\s*Let\s*(.*)\.$",
                                                 lemma_statement,
                                                 flags=re.DOTALL)
                            if let_match and not ":=" in lemma_statement:
                                split = split_by_char_outside_matching("\(", "\)", ":=",
                                                                       let_match.group(1))
                                assert not split
                                name_and_type = let_match.group(1)
                                if search_status == SearchStatus.FAILURE:
                                    postfix = "(*FAILURE*)"
                                else:
                                    postfix = "(*INCOMPLETE*)"
                                admitted_defn = f"Hypothesis {name_and_type} {postfix}."
                                admitted_cmds = [admitted_defn]
                            else:
                                if search_status == SearchStatus.FAILURE:
                                    num_proofs_failed += 1
                                    admitted = "Admitted (*FAILURE*)."
                                else:
                                    admitted = "Admitted (*INCOMPLETE*)."
                                admitted_cmds = [lemma_statement, "Proof.\n", admitted]
                            append_to_solution_vfile(args.output_dir, args.filename,
                                                     admitted_cmds)
                        # Run the original proof
                        original_tactics = run_to_next_vernac(coq, pbar, initial_context,
                                                              lemma_statement)
                        add_proof_block(coq,
                                        search_status, tactic_solution,
                                        initial_context, original_tactics)
            except serapi_instance.CoqAnomaly as e:
                if args.log_anomalies:
                    with args.log_anomalies.open('a') as f:
                        traceback.print_exc(file=f)
                if lemma_statement:
                    commands_in.insert(0, lemma_statement)
                if commands_caught_up == len(commands_run):
                    eprint(f"Hit the same anomaly twice!")
                    if lemma_statement in lemmas_to_skip:
                        raise e
                    else:
                        lemmas_to_skip.append(lemma_statement)
                commands_caught_up = len(commands_run)
                if args.hardfail:
                    raise e
                if args.verbose:
                    eprint(f"Hit a coq anomaly {e.msg}! Restarting coq instance.")
            except Exception as e:
                eprint(f"FAILED: in file {str(args.filename)}, {repr(e)}")
                raise
    write_html(args, args.output_dir, args.filename, blocks_out)
    write_csv(args, args.filename, blocks_out)

def html_header(tag : Tag, doc : Doc, text : Text, css : List[str],
                javascript : List[str], title : str) -> None:
    with tag('head'):
        for filename in css:
            doc.stag('link', href=filename, rel='stylesheet')
        for filename in javascript:
            with tag('script', type='text/javascript',
                     src=filename):
                pass
        with tag('title'):
            text(title)

def write_csv(args : argparse.Namespace, filename : str, doc_blocks : List[DocumentBlock]):
    with open("{}/{}.csv".format(args.output_dir, escape_filename(str(filename))),
              'w', newline='') as csvfile:
        for k, v in vars(args).items():
            csvfile.write("# {}: {}\n".format(k, v))

        rowwriter = csv.writer(csvfile, lineterminator=os.linesep)
        for block in doc_blocks:
            if isinstance(block, ProofBlock):
                rowwriter.writerow([block.lemma_statement.strip(),
                                    block.status,
                                    len(block.original_tactics)])

def read_csv_options(f : Iterable[str]) -> Tuple[argparse.Namespace, Iterable[str]]:
    params : Dict[str, str] = {}
    f_iter = iter(f)
    final_line = ""
    for line in f_iter:
        param_match = re.match("# (.*): (.*)", line)
        if param_match:
            params[param_match.group(1)] = param_match.group(2)
        else:
            final_line = line
            break
    rest_iter : Iterable[str]
    if final_line == "":
        rest_iter = iter([])
    else:
        rest_iter = itertools.chain([final_line], f_iter)
    return argparse.Namespace(**params), rest_iter

important_args = ["prelude", "context_filter", "weightsfile", "predictor", "search_width", "search_depth"]

def check_csv_args(args : argparse.Namespace, vfilename : Path2) -> None:
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    with open(args.output_dir / (escape_filename(str(vfilename)) + ".csv"),
              'r', newline='') as csvfile:
        if args.check_consistent:
            saved_args, rest_iter = read_csv_options(csvfile)
            for arg in important_args:
                try:
                    oldval = str(vars(saved_args)[arg])
                    newval = str(vars(args)[arg])
                    if oldval != newval:
                        raise ArgsMismatchException(f"Old value of {arg} is {oldval}, "
                                                    f"new value is {newval}")
                except KeyError:
                    raise ArgsMismatchException(f"No old value for arg {arg} found.")

def write_html(args : argparse.Namespace,
               output_dir : str, filename : Path2,
               doc_blocks : List[DocumentBlock]) -> None:
    global unnamed_goal_number
    unnamed_goal_number = 0
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        html_header(tag, doc, text, [details_css], [details_javascript],
                    "Proverbot Detailed Report for {}".format(str(filename)))
        with tag('body', onload='init()'), tag('pre'):
            for block_idx, block in enumerate(doc_blocks):
                if isinstance(block, VernacBlock):
                    write_commands(block.commands, tag, text, doc)
                else:
                    assert isinstance(block, ProofBlock)
                    status_klass = classFromSearchStatus(block.status)
                    write_lemma_button(block.lemma_statement, block.module,
                                       status_klass, tag, text)
                    with tag('div', klass='region'):
                        with tag('div', klass='predicted'):
                            write_tactics(args, block.predicted_tactics, block_idx,
                                          tag, text, doc)
                        with tag('div', klass='original'):
                            write_tactics(args, block.original_tactics, block_idx,
                                          tag, text, doc)
    with open("{}/{}.html".format(output_dir, escape_filename(str(filename))), 'w') as fout:
        # fout.write(syntax.syntax_highlight(doc.getvalue()))
        fout.write(doc.getvalue())

def write_lemma_button(lemma_statement : str, module : Optional[str],
                       status_klass : str, tag : Tag, text : Text):
    global unnamed_goal_number
    lemma_name = \
        serapi_instance.lemma_name_from_statement(lemma_statement)
    module_prefix = escape_lemma_name(module)
    if lemma_name == "":
        unnamed_goal_number += 1
        fullname = module_prefix + lemma_name + str(unnamed_goal_number)
    else:
        fullname = module_prefix + lemma_name
    with tag('button', klass='collapsible {}'.format(status_klass),
             onmouseover="hoverLemma(\"{}\")".format(fullname),
             onmouseout="unhoverLemma(\"{}\")".format(fullname)):
        with tag('code', klass='buttontext'):
            text(lemma_statement.strip())
def write_commands(commands : List[str], tag : Tag, text : Text, doc : Doc):
    for cmd in commands:
        with tag('code', klass='plaincommand'):
            text(cmd.strip("\n"))
        doc.stag('br')

def escape_quotes(term : str):
    return re.sub("\"", "\\\"", term)

def subgoal_to_string(args : argparse.Namespace, sg : Obligation) -> str:
    return "(\"" + escape_quotes(sg.goal[:args.max_print_term]) + "\", (\"" + \
        "\",\"".join([escape_quotes(hyp[:args.max_print_term]) for hyp in
                      sg.hypotheses[:args.max_print_hyps]]) + "\"))"

def write_tactics(args : argparse.Namespace,
                  tactics : List[TacticInteraction],
                  region_idx : int,
                  tag : Tag, text : Text, doc : Doc):
    for t_idx, t in enumerate(tactics):
        idStr = '{}-{}'.format(region_idx, t_idx)
        subgoals_str = "(" + ",".join([subgoal_to_string(args, subgoal)
                                       for subgoal in
                                       t.context_before.all_goals[:args.max_print_subgoals]]) + ")"
        with tag('span',
                 ('data-subgoals', subgoals_str),
                 id='command-{}'.format(idStr),
                 onmouseover='hoverTactic("{}")'.format(idStr),
                 onmouseout='unhoverTactic()'):
            with tag('code', klass='plaincommand'):
                text(t.tactic.strip())
            doc.stag('br')

def classFromSearchStatus(status : SearchStatus) -> str:
    if status == SearchStatus.SUCCESS:
        return 'good'
    elif status == SearchStatus.INCOMPLETE:
        return 'okay'
    else:
        return 'bad'

def make_new_solution_vfile(args : argparse.Namespace, model_name : str,
                            filename : Path2) -> None:
    solution_vfile_path = (args.output_dir / escape_filename(str(filename)))\
        .with_suffix(".v")
    with solution_vfile_path.open(mode='w') as f:
        for k, v in [("search-width", args.search_width),
                     ("search-depth", args.search_depth),
                     ("model", model_name)]:
            print(f"(* {k}: {v} *)", file=f)

def append_to_solution_vfile(outdir : Path2, filename : Path2,
                             lines : List[str]) -> None:
    solution_vfile_path = (outdir / escape_filename(str(filename))).with_suffix(".v")
    with solution_vfile_path.open(mode='a') as f:
        for line in lines:
            print(line.strip(), file=f, flush=True)

def check_solution_vfile_args(args : argparse.Namespace, model_name : str,
                              f_iter : Iterator[str]) -> Iterable[str]:
    next_line = next(f_iter)
    argline_match = re.match("\(\* (\S*): (\S*) \*\)", next_line)
    checked_args = {"search-width":args.search_width,
                    "search-depth":args.search_depth,
                    "model": model_name}
    while argline_match:
        k, v = argline_match.group(1,2)
        if not str(checked_args[k]) == v:
            raise ArgsMismatchException(f"Arg mistmatch: {k} is {checked_args[k]} "
                                        f"in cur report, {v} in file")
        try:
            next_line = next(f_iter)
        except:
            return f_iter
        argline_match = re.match("\(\* (\S*): (\S*) \*\)", next_line)
    return itertools.chain([next_line], f_iter)

def replay_solution_vfile(args : argparse.Namespace, coq : serapi_instance.SerapiInstance,
                          model_name : str, filename : Path2, commands_in : List[str],
                          bar_idx : int) \
                          -> Tuple[List[str], List[str], List[DocumentBlock],
                                   int, int, int, int]:
    blocks_out : List[DocumentBlock] = []
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    num_original_commands_run = 0
    in_proof = False
    skip_sync_next_lemma = False
    curLemma = ""
    curProofInters : List[TacticInteraction] = []
    curVernacCmds : List[str] = []
    commands_run = []
    with open(f"{args.output_dir}/{escape_filename(str(filename))}.v", 'r') as f:
        print(f"Resuming from file {args.filename}")
        f_iter = check_solution_vfile_args(args, model_name,
                                           iter(f))
        svfile_commands = serapi_instance.read_commands_preserve(args, bar_idx,
                                                                 "".join(f_iter))
        commands_in_iter = iter(commands_in)
        def peek_loaded():
            nonlocal commands_in_iter
            item = next(commands_in_iter)
            commands_in_iter = itertools.chain([item], commands_in_iter)
            return item
        with tqdm(total=len(svfile_commands),
                  unit="cmd", file=sys.stdout,
                  desc="Replaying", disable=(not args.progress),
                  leave=False,position=(bar_idx*2),
                  dynamic_ncols=True, bar_format=mybarfmt) as pbar:
            while svfile_commands:
                saved_command = svfile_commands.pop(0)
                pbar.update(1)
                context_before = coq.proof_context or ProofContext([],[],[],[])
                if re.match("\s*Hypothesis", saved_command) and \
                   re.match("\s*Let", peek_loaded()):
                    status_marker = unwrap(re.search("(\(\*(?:INCOMPLETE|FAILURE)\*\))",
                                                     saved_command)).group(1)

                    svfile_commands = [peek_loaded(), "Proof.",
                                       f"Admitted {status_marker}."] + \
                                       svfile_commands
                    continue

                if not(coq.proof_context != None and len(context_before.all_goals) == 0 and
                       not (serapi_instance.ending_proof(saved_command) or
                            re.match("\s*[}]", saved_command))):
                    coq.run_stmt(saved_command)
                if coq.proof_context == None:
                    loaded_command = next(commands_in_iter)
                    commands_run.append(loaded_command)
                    if in_proof:
                        if not skip_sync_next_lemma:
                            curProofInters.append(TacticInteraction(saved_command, context_before))
                            num_proofs += 1
                            if re.match("Qed\.", saved_command):
                                search_status = SearchStatus.SUCCESS
                                num_proofs_completed += 1
                            elif re.match("Admitted \(\*FAILURE\*\)\.", saved_command):
                                search_status = SearchStatus.FAILURE
                                num_proofs_failed += 1
                            else:
                                search_status = SearchStatus.INCOMPLETE
                            lemma_name = serapi_instance.lemma_name_from_statement(curLemma)
                            if lemma_name:
                                coq.run_stmt(f"Reset {lemma_name}.")
                            else:
                                coq.run_stmt(f"Back.")

                            origProofInters = []
                    else:
                        if re.match("Reset .*\.", saved_command):
                            commands_in_iter = itertools.chain([loaded_command],
                                                               commands_in_iter)
                            commands_run = commands_run[:-1]
                            skip_sync_next_lemma = True
                            continue
                        def normalize_command(cmd : str) -> str:
                            return re.sub("\s*\(\*(?:INCOMPLETE|FAILURE)\*\)", "",
                                          re.sub("Let", "Hypothesis",
                                                 re.sub("\s+", " ", cmd.strip())))
                        if not normalize_command(loaded_command) == \
                           normalize_command(saved_command):
                            raise SourceChangedException(
                                f"Loaded command {normalize_command(loaded_command)} doesn't match saved command {normalize_command(saved_command)}")
                        curVernacCmds.append(loaded_command)
                    if in_proof:
                        in_proof = False
                        if not skip_sync_next_lemma:
                            proof_cmds = list(serapi_instance.next_proof(
                                itertools.chain([loaded_command], commands_in_iter)))
                            coq.run_stmt(loaded_command)
                            num_original_commands_run += len(proof_cmds)
                            commands_run += proof_cmds[1:]
                            for proof_cmd in tqdm(proof_cmds[1:], unit="tac", file=sys.stdout,
                                                  desc="Running original proof",
                                                  disable=(not args.progress),
                                                  leave=False, position=(bar_idx * 2) + 1,
                                                  dynamic_ncols=True, bar_format=mybarfmt):
                                context_before_orig = coq.proof_context
                                coq.run_stmt(proof_cmd)
                                origProofInters.append(
                                    TacticInteraction(proof_cmd, context_before_orig))
                            blocks_out.append(ProofBlock(curLemma,
                                                         coq.module_prefix,
                                                         search_status,
                                                         curProofInters, origProofInters))
                            curProofInters = []
                        else:
                            commands_in_iter = itertools.chain([loaded_command],
                                                               commands_in_iter)
                            commands_run = commands_run[:-1]
                            skip_sync_next_lemma = False

                else:
                    if not in_proof:
                        in_proof = True
                        curLemma = saved_command
                        blocks_out.append(VernacBlock(curVernacCmds))
                        curVernacCmds = []
                    else:
                        curProofInters.append(TacticInteraction(saved_command, context_before))
        assert not in_proof
        if curVernacCmds:
            blocks_out.append(VernacBlock(curVernacCmds))
        return commands_run, list(commands_in_iter), blocks_out,\
            num_proofs, num_proofs_failed, num_proofs_completed, num_original_commands_run

def try_run_prelude(args: argparse.Namespace, coq : SerapiInstance):
    if not args.weightsfile:
        eprint("No weightsfile")
        return
    prelude_path = args.weightsfile.with_suffix(".prelude.v")
    if not prelude_path.exists():
        eprint(f"Couldn't find prelude at {prelude_path}")
        return
    eprint("Running prelude:", guard=args.verbose>=2)
    prelude_commands = serapi_instance.load_commands_preserve(args, 0, prelude_path)
    for command in prelude_commands:
        eprint(f"Found command {command}", guard=args.verbose>=2)
        coq.run_stmt(command)
    eprint("Finished with prelude", guard=args.verbose>=2)

# The core of the search report

class SearchResult(NamedTuple):
    status : SearchStatus
    commands : Optional[List[TacticInteraction]]

def metrics_from_search_result(result:SearchResult) -> Dict:
    answer_status = str(result.status).split('.')[1]
    proof = "\n".join([cmd.tactic for cmd in result.commands]) if result.commands else "No proof"
    proof_len = len(result.commands) if result.commands else 0
    return {
        "answer_status": answer_status,
        "proof": proof,
        "proof_len": proof_len,
    }

# This method attempts to complete proofs using search.

def tryPrediction(args : argparse.Namespace,
                  coq : serapi_instance.SerapiInstance,
                  prediction : str,
                  previousNode : LabeledNode) -> Tuple[ProofContext, int, int, int, Optional[Exception], float]:
    coq.quiet = True
    time_left = max(args.max_proof_time - time_on_path(previousNode), 0)
    start_time = time.time()
    time_per_command = 60 if coq.use_hammer else args.max_tactic_time
    try:
        coq.run_stmt(prediction, timeout=min(time_left, time_per_command))
        error = None
    except (serapi_instance.TimeoutError, serapi_instance.ParseError,
            serapi_instance.CoqExn, serapi_instance.OverflowError,
            serapi_instance.UnrecognizedError) as e:
        return coq.proof_context, 0, 0, 0, e, time.time() - start_time

    time_taken = time.time() - start_time
    num_stmts = 1
    subgoals_closed = 0
    if len(unwrap(coq.proof_context).fg_goals) == 0 and \
       len(unwrap(coq.proof_context).shelved_goals) > 0:
        coq.run_stmt("Unshelve.")
        num_stmts += 1
    while len(unwrap(coq.proof_context).fg_goals) == 0 and not completed_proof(coq):
        coq.run_stmt("}")
        subgoals_closed += 1
        num_stmts += 1
    if coq.count_fg_goals() > 1 or \
       (coq.count_fg_goals() > 0 and subgoals_closed > 0):
        subgoals_opened = 1
        coq.run_stmt("{")
        num_stmts += 1
    else:
        subgoals_opened = 0
    context_after = coq.proof_context
    assert context_after
    return context_after, num_stmts, subgoals_closed, subgoals_opened, error, time_taken

def completed_proof(coq : serapi_instance.SerapiInstance) -> bool:
    if coq.proof_context:
        return len(coq.proof_context.all_goals) == 0 and \
            coq.tactic_history.curDepth() == 0
    else:
        return False


def contextInPath(full_context : ProofContext, path : List[LabeledNode]):
    return any([contextSurjective(full_context, n.context_before)
                for n in path])

goalBignessLimit = 3000
def contextIsBig(context : ProofContext):
    for obligation in context.all_goals:
        for hypothesis in obligation.hypotheses:
            if len(hypothesis) > goalBignessLimit:
                return True
        if len(obligation.goal) > goalBignessLimit:
            return True
    return False

def numNodesInTree(branching_factor : int, depth : int):
    if depth*branching_factor > 100000:
        return 9999999 # too big anyway
    assert depth > 0, f"depth is {depth}"
    result = int((branching_factor ** depth - 1) / \
                 (branching_factor - 1))
    assert result >= 1, f"result is {result}"
    return result

class TqdmSpy(tqdm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time = time.time
    @property
    def n(self):
        return self.__n
    @n.setter
    def n(self, value):
        self.__n = value
    def update(self, value):
        self.n = self.n + value
        super().update(value);

from search_dfs_via_visitor import proof_search_with_graph_visitor, \
    CoqVisitorCertaintyWithCurNodeBonus, CoqVisitorProductCertaintyWithCurNodeBonus, CoqVisitorDfs,\
    CoqVisitorDfsThenProductCertainty
# from search_dfs import dfs_proof_search_with_graph
from tree_traverses import best_first_search, dfs, bfs

def old_dfs_proof_search_with_graph(lemma_statement : str,
                                module_name : Optional[str],
                                coq : serapi_instance.SerapiInstance,
                                args : argparse.Namespace,
                                bar_idx : int) \
                                -> SearchResult:
    global unnamed_goal_number
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    g = SearchGraph(lemma_name)
    def cleanupSearch(num_stmts : int, msg : Optional[str] = None):
        if msg:
            eprint(f"Cancelling {num_stmts} statements "
                   f"because {msg}.", guard=args.verbose >= 2)
        for _ in range(num_stmts):
            coq.cancel_last()
    hasUnexploredNode = False
    def search(pbar : tqdm, current_path : List[LabeledNode],
               subgoal_distance_stack : List[int],
               extra_depth : int) -> SubSearchResult:
        nonlocal hasUnexploredNode
        if args.relevant_lemmas == "local":
            relevant_lemmas = coq.local_lemmas[:-1]
        elif args.relevant_lemmas == "hammer":
            relevant_lemmas = coq.get_hammer_premises()
        elif args.relevant_lemmas == "searchabout":
            relevant_lemmas = coq.get_lemmas_about_head()
        else:
            assert False, args.relevant_lemmas
        tactic_context_before = TacticContext(relevant_lemmas,
                                              coq.prev_tactics,
                                              coq.hypotheses,
                                              coq.goals)
        predictions = [prediction.prediction for prediction in
                       predictor.predictKTactics(tactic_context_before, args.max_attempts)]
        proof_context_before = coq.proof_context
        if coq.use_hammer:
            predictions = [prediction + "; try hammer." for prediction in predictions]
        num_successful_predictions = 0
        for prediction_idx, prediction in enumerate(predictions):
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
                assert pbar.n > 0

                predictionNode = g.mkNode(prediction, proof_context_before,
                                          current_path[-1])
                predictionNode.time_taken = time_taken

                #### 1.
                if subgoal_distance_stack:
                    new_distance_stack = (subgoal_distance_stack[:-1] +
                                          [subgoal_distance_stack[-1]+1])
                else:
                    new_distance_stack = []

                #### 2.
                new_extra_depth = extra_depth
                for _ in range(subgoals_closed):
                    closed_goal_distance = new_distance_stack.pop()
                    new_extra_depth += closed_goal_distance

                #### 3.
                new_distance_stack += [0] * subgoals_opened

                #############
                if completed_proof(coq):
                    solution = g.mkQED(predictionNode)
                    return SubSearchResult(solution, subgoals_closed)
                elif contextInPath(context_after, current_path[1:] + [predictionNode]):
                    if not args.count_softfail_predictions:
                        num_successful_predictions -= 1
                    g.setNodeColor(predictionNode, "orange")
                    cleanupSearch(num_stmts, "resulting context is in current path")
                elif contextIsBig(context_after):
                    g.setNodeColor(predictionNode, "orange4")
                    cleanupSearch(num_stmts, "resulting context has too big a goal")
                elif len(current_path) < args.search_depth + new_extra_depth:
                    sub_search_result = search(pbar, current_path + [predictionNode],
                                               new_distance_stack, new_extra_depth)
                    cleanupSearch(num_stmts, "we finished subsearch")
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
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        depth = (args.search_depth + new_extra_depth + 1) \
                            - len(current_path)
                        return SubSearchResult(None, subgoals_closed)
            except (serapi_instance.CoqExn, serapi_instance.TimeoutError,
                    serapi_instance.OverflowError, serapi_instance.ParseError,
                    serapi_instance.UnrecognizedError):
                continue
            except serapi_instance.NoSuchGoalError:
                raise
        return SubSearchResult(None, 0)
    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1
    with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
                 desc="Proof", disable=(not args.progress),
                 leave=False,
                 position=((bar_idx*2)+1),
                 dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        command_list, _ = search(pbar, [g.start_node], [], 0)
        pbar.clear()
    module_prefix = escape_lemma_name(module_name)
    if lemma_name == "":
        unnamed_goal_number += 1
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}"
               f"{unnamed_goal_number}.svg")
    else:
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}.svg")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, command_list)
    elif hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    else:
        return SearchResult(SearchStatus.FAILURE, None)


def attempt_search(args: argparse.Namespace,
                   lemma_statement: str,
                   module_name: Optional[str],
                   coq: serapi_instance.SerapiInstance,
                   bar_idx: int) \
        -> SearchResult:
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    logger.set_experiment_as_lemma(module_name, lemma_name, args)
    log_filename = str(args.output_dir) + "/" + logger.run_id() + escape_lemma_name(module_name) + lemma_name+".apkl"
    if args.traverse_method == "OldDFS":
        start = time.time()
        result = old_dfs_proof_search_with_graph(lemma_statement, module_name, coq, args, bar_idx)
        time_spent = time.time() - start
        logger.log_metric(log_filename, "time_spent", time_spent)
        logger.log_metrics(log_filename, metrics_from_search_result(result))
        return result
    if args.traverse_method == "DFS":
        traverse_function = dfs
    elif args.traverse_method == "BestFS":
        traverse_function = best_first_search
    elif args.traverse_method == "BFS":
        traverse_function = bfs
    else:
        raise NotImplementedError(f"Unknown traverse method {args.traverse_method}")

    if args.bestfs_edge_scoring_fun == "certainty":
        visitor_class = CoqVisitorCertaintyWithCurNodeBonus
    elif args.bestfs_edge_scoring_fun == "product_certainty":
        visitor_class = CoqVisitorProductCertaintyWithCurNodeBonus
    elif args.bestfs_edge_scoring_fun == "dfs_then_product_certainty":
        visitor_class = CoqVisitorDfsThenProductCertainty
    elif args.bestfs_edge_scoring_fun == "BestFSyDFS":
        visitor_class = CoqVisitorDfs
    else:
        raise NotImplementedError(f"Unknown bestfs-edge-scoring-fun {args.bestfs_edge_scoring_fun}")
    result, metrics = proof_search_with_graph_visitor(lemma_statement, module_name, coq, args, bar_idx,
                                                      traverse_function=traverse_function,
                                                      visitor_class=visitor_class,
                                                      temperature=args.bestfs_edge_scoring_temperature,
                                                      certainty_bias=args.certainty_bias)
    logger.log_metrics(log_filename, metrics)
    return result

# This implementation is here for reference/documentation
# def dfs_proof_search(lemma_statement : str, coq : serapi_instance.SerapiInstance,
#                      args : argparse.Namespace) -> Optional[List[str]]:
#     def get_context() -> TacticContext:
#         return TacticContext(coq.prev_tactics, coq.hypotheses,
#                              coq.goals)
#     def predictions() -> List[str]:
#         return [pred.prediction for pred in
#                 predictor.predictKTactics(get_context(), args.search_width)]
#     def search(current_path : List[str]) -> Optional[List[str]]:
#         for prediction in predictions():
#             try:
#                 coq.quiet = True
#                 coq.run_stmt(prediction)
#                 if completed_proof(coq):
#                     return current_path + [prediction]
#                 elif len(current_path) + 1 < args.search_depth:
#                     sub_search_result = search(current_path + [prediction])
#                     if sub_search_result:
#                         return sub_search_result
#                 coq.cancel_last()
#             except (serapi_instance.CoqExn, serapi_instance.TimeoutError):
#                 continue
#         return None
#     return search([])


# from graphviz import Digraph

def subgoalSurjective(newsub : Obligation,
                      oldsub : Obligation) -> bool:
    oldhyp_terms = [serapi_instance.get_hyp_type(hyp) for hyp in oldsub.hypotheses]
    for newhyp_term in [serapi_instance.get_hyp_type(hyp)
                        for hyp in newsub.hypotheses]:
        if newhyp_term not in oldhyp_terms:
            return False
    return newsub.goal == oldsub.goal
def contextSurjective(newcontext : ProofContext, oldcontext : ProofContext):
    for oldsub in oldcontext.all_goals:
        if not any([subgoalSurjective(newsub, oldsub)
                    for newsub in newcontext.all_goals]):
            return False
    return len(newcontext.all_goals) >= len(oldcontext.all_goals)


def time_on_path(node : LabeledNode) -> float:
    if node.previous == None:
        return node.time_taken
    else:
        return time_on_path(node.previous) + node.time_taken


if __name__ == "__main__":
    main(sys.argv[1:], 0)
