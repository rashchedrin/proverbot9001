#!/usr/bin/env python3.7
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
import datetime
import os
import os.path
import queue
import re
import subprocess
import sys
import threading

# This dependency is in pip, the python package manager
from sexpdata import *
from traceback import *
from util import *
from compcert_linearizer_failures import compcert_failures

import serapi_instance
from serapi_instance import (AckError, CompletedError, CoqExn,
                             BadResponse, TimeoutError, ParseError, NoSuchGoalError)

from typing import (Optional, List, Iterator, Iterable, Any, Match,
                    Tuple, Pattern, Union)

from itertools import islice

# exception for when things go bad, but not necessarily because of the linearizer
class LinearizerCouldNotLinearize(Exception):
    pass

# exception for when the linearizer trips onto itself
class LinearizerThisShouldNotHappen(Exception):
    pass

from tqdm import tqdm
def linearize_commands(args : argparse.Namespace, file_idx : int,
                       commands_sequence: Iterable[str],
                       coq : serapi_instance.SerapiInstance,
                       filename : str, relative_filename : str,
                       skip_nochange_tac:bool):
    commands_iter = iter(commands_sequence)
    command = next(commands_iter, None)
    assert command, "Got an empty sequence!"
    while command:
        # Run up to the next proof
        while coq.count_fg_goals() == 0:
            coq.run_stmt(command)
            if coq.count_fg_goals() == 0:
                yield command
                command = next(commands_iter, None)
                if not command:
                    return

        # Cancel the proof starting command so that we're right before the proof
        coq.cancel_last()

        # Pull the entire proof from the lifter into command_batch
        command_batch = []
        while command and not serapi_instance.ending_proof(command):
            command_batch.append(command)
            command = next(commands_iter, None)
        # Get the QED on there too.
        if command:
            command_batch.append(command)

        # Now command_batch contains everything through the next
        # Qed/Defined.
        theorem_statement = serapi_instance.kill_comments(command_batch.pop(0))
        theorem_name = theorem_statement.split(":")[0].strip()
        coq.run_stmt(theorem_statement)
        yield theorem_statement
        if [relative_filename, theorem_name] in compcert_failures:
            eprint("Skipping {}".format(theorem_name))
            for command in command_batch:
                coq.run_stmt(command)
                yield command
            command = next(commands_iter, None)
            continue

        # This might not be super robust?
        match = re.fullmatch("\s*Proof with (.*)\.", command_batch[0])
        if match and match.group(1):
            with_tactic = match.group(1)
        else:
            with_tactic = ""

        orig = command_batch[:]
        command_batch = list(prelinear_desugar_tacs(command_batch))
        try:
            batch_handled = list(handle_with(command_batch, with_tactic))
            linearized_commands = list(linearize_proof(coq, theorem_name, batch_handled,
                                                       args.debug, skip_nochange_tac))
            yield from linearized_commands
        except (BadResponse, CoqExn, LinearizerCouldNotLinearize, ParseError, TimeoutError, NoSuchGoalError) as e:
            eprint("Aborting current proof linearization!")
            eprint("Proof of:\n{}\nin file {}".format(theorem_name, filename))
            eprint()
            if args.hardfail:
                raise e
            coq.run_stmt("Abort.")
            coq.run_stmt(theorem_statement)
            for command in orig:
                if command:
                    coq.run_stmt(command)
                    yield command

        command = next(commands_iter, None)

def split_to_next_matching(openpat : str, closepat : str, target : str) \
    -> Tuple[str, str]:
    counter = 1
    openp = re.compile(openpat)
    closep = re.compile(closepat)
    firstmatch = openp.search(target)
    assert firstmatch, "Coudn't find an opening pattern!"
    curpos = firstmatch.end()
    while counter > 0:
        nextopenmatch = openp.search(target, curpos)
        nextopen = nextopenmatch.end() if nextopenmatch else len(target)

        nextclosematch = closep.search(target, curpos)
        nextclose = nextclosematch.end() if nextclosematch else len(target)
        if nextopen < nextclose:
            counter += 1
            assert nextopen + 1 > curpos, (target, curpos, nextopen)
            curpos = nextopen
        else:
            counter -= 1
            assert nextclose + 1 > curpos
            curpos = nextclose
    return target[:curpos], target[curpos:]

def multisplit_matching(openpat : str, closepat : str,
                        splitpat : str, target : str) \
                        -> List[str]:
    splits = []
    nextsplit = split_by_char_outside_matching(openpat, closepat, splitpat, target)
    while nextsplit:
        before, rest = nextsplit
        splits.append(before)
        nextsplit = split_by_char_outside_matching(openpat, closepat, splitpat, rest[1:])
    splits.append(rest[1:])
    return splits

def split_by_char_outside_matching(openpat : str, closepat : str,
                                   splitpat : str, target : str) \
    -> Optional[Tuple[str, str]]:
    counter = 0
    curpos = 0
    openp = re.compile(openpat)
    closep = re.compile(closepat)
    splitp = re.compile(splitpat)
    def search_pat(pat : Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(target, curpos)
        return match, match.end() if match else len(target) + 1

    while curpos < len(target) + 1:
        _, nextopenpos = search_pat(openp)
        _, nextclosepos = search_pat(closep)
        nextsplitchar, nextsplitpos = search_pat(splitp)

        if nextopenpos < nextclosepos and nextopenpos < nextsplitpos:
            counter += 1
            assert nextopenpos > curpos
            curpos = nextopenpos
        elif nextclosepos < nextopenpos and \
             (nextclosepos < nextsplitpos or
              (nextclosepos == nextsplitpos and counter > 0)):
            counter -= 1
            assert nextclosepos > curpos
            curpos = nextclosepos
        else:
            if counter <= 0:
                if nextsplitpos > len(target):
                    return None
                assert nextsplitchar
                return target[:nextsplitchar.start()], target[nextsplitchar.start():]
            else:
                assert nextsplitpos > curpos
                curpos = nextsplitpos
    return None

def linearize_proof(coq : serapi_instance.SerapiInstance,
                    theorem_name : str,
                    command_batch : List[str],
                    debug:bool=False,
                    skip_nochange_tac:bool=False) -> Iterable[str]:
    pending_commands_stack : List[Union[str, List[str], None]] = []
    while command_batch:
        while coq.count_fg_goals() == 0:
            indentation = "  " * (len(pending_commands_stack))
            if len(pending_commands_stack) == 0:
                while command_batch:
                    command = command_batch.pop(0)
                    if "Transparent" in command or \
                       serapi_instance.ending_proof(command):
                        coq.run_stmt(command)
                        yield command
                return
            coq.run_stmt("}")
            yield indentation + "}"
            if coq.count_fg_goals() > 0:
                coq.run_stmt("{")
                yield indentation + "{"
                pending_commands = pending_commands_stack[-1]
                if isinstance(pending_commands, list):
                    next_cmd, *rest_cmd = pending_commands
                    dotdotmatch = re.match("(.*)<\.\.>", next_cmd)
                    if (not rest_cmd) and dotdotmatch:
                        pending_commands_stack[-1] = [next_cmd]
                        command_batch.insert(0, dotdotmatch.group(1))
                    else:
                        command_batch.insert(0, next_cmd)
                    pending_commands_stack[-1] = rest_cmd if rest_cmd else None
                    pass
                elif pending_commands:
                    command_batch.insert(0, pending_commands)
            else:
                popped = pending_commands_stack.pop()
                if isinstance(popped, list) and len(popped) > 0 and len(pending_commands_stack) > 1:
                    if pending_commands_stack[-1] is None:
                        pending_commands_stack[-1] = popped
                    elif isinstance(pending_commands_stack[-1], list):
                        pending_commands_stack[-1] = popped + pending_commands_stack[-1]
        command = command_batch.pop(0)
        assert serapi_instance.isValidCommand(command), \
            f"command is \"{command}\", command_batch is {command_batch}"
        comment_before_command = ""
        command_proper = command
        while "(*" in command_proper:
            next_comment, command_proper = \
                split_to_next_matching("\(\*", "\*\)", command_proper)
            command_proper = command_proper[1:]
            comment_before_command += next_comment
        if comment_before_command:
            yield comment_before_command
        if re.match("\s*[*+-]+\s*|\s*[{}]\s*", command):
            continue

        command = command_proper
        if debug:
            eprint(f"Linearizing command \"{command}\"")

        goal_selector_match = re.match(r"\s*(\d*)\s*:\s*(.*)\.\s*", command)
        if goal_selector_match:
            goal_num = int(goal_selector_match.group(1))
            rest = goal_selector_match.group(2)
            assert goal_num >= 2
            if pending_commands_stack[-1] is None:
                pending_commands_stack[-1] = ["idtac."]*(goal_num - 2) + [rest + "."]
            elif isinstance(pending_commands_stack[-1], str):
                pending_cmd = pending_commands_stack[-1]
                pending_commands_stack[-1] = [pending_cmd] * (goal_num - 2) + \
                    [rest + " ; " + pending_cmd] + [pending_cmd + "<..>"]
            else:
                assert isinstance(pending_commands_stack[-1], list)
            continue

        if split_by_char_outside_matching("\(", "\)", "\|\||&&", command):
            coq.run_stmt(command)
            yield command
            continue

        if re.match("\(", command.strip()):
            inside_parens, after_parens = split_to_next_matching('\(', '\)', command)
            command = inside_parens.strip()[1:-1] + after_parens

        # Extend this to include "by \(" as an opener if you don't
        # desugar all the "by"s
        semi_match = split_by_char_outside_matching("try \(|\(|\{\|", "\)|\|\}",
                                                    "\s*;\s*", command)
        if semi_match:
            base_command, rest = semi_match
            rest = rest.lstrip()[1:]
            coq.run_stmt(base_command + ".")
            indentation = "  " * (len(pending_commands_stack) + 1)
            yield indentation + base_command.strip() + "."

            if re.match("\(", rest) and not \
               split_by_char_outside_matching("\(", "\)", "\|\|", rest):
                inside_parens, after_parens = split_to_next_matching('\(', '\)', rest)
                rest = inside_parens[1:-1] + after_parens
            bracket_match = re.match("\[", rest.strip())
            if bracket_match:
                bracket_command, rest_after_bracket = \
                    split_to_next_matching('\[', '\]', rest)
                rest_after_bracket = rest_after_bracket.lstrip()[1:]
                clauses = multisplit_matching("\[", "\]", "(?<!\|)\|(?!\|)",
                                              bracket_command.strip()[1:-1])
                commands_list = [cmd.strip() if cmd.strip().strip(".") != ""
                                 else "idtac" + cmd.strip() for cmd in
                                 clauses]
                dotdotpat = re.compile(r"(.*)\.\.($|\W)")
                ending_dotdot_match = dotdotpat.match(commands_list[-1])
                if ending_dotdot_match:
                    commands_list = commands_list[:-1] + \
                        ([ending_dotdot_match.group(1)] *
                         (coq.count_fg_goals() -
                          len(commands_list) + 1))
                else:
                    starting_dotdot_match = dotdotpat.match(commands_list[0])
                    if starting_dotdot_match:
                        starting_tac = starting_dotdot_match.group(1)
                        commands_list = [starting_tac] * (coq.count_fg_goals() -
                                                          len(commands_list) + 1)\
                                                          + commands_list[1:]
                    else:
                        for idx, command_case in enumerate(commands_list[1:-1]):
                            middle_dotdot_match = dotdotpat.match(command_case)
                            if middle_dotdot_match:
                                commands_list = \
                                    commands_list[:idx] + \
                                    [command_case] * (coq.count_fg_goals() -
                                                      len(commands_list) + 1) + \
                                                      commands_list[idx+1:]
                                break
                if rest_after_bracket.strip():
                    command_remainders = [cmd + ";" + rest_after_bracket
                                          for cmd in commands_list]
                else:
                    command_remainders = [cmd + "." for cmd in commands_list]
                command_batch.insert(0, command_remainders[0])
                if coq.count_fg_goals() > 1:
                    pending_commands_stack.append(command_remainders[1:])
                    coq.run_stmt("{")
                    yield indentation + "{"
            else:
                if coq.count_fg_goals() > 0:
                    command_batch.insert(0, rest)
                if coq.count_fg_goals() > 1:
                    pending_commands_stack.append(rest)
                    coq.run_stmt("{")
                    yield indentation + "{"
        elif coq.count_fg_goals() > 0:
            coq.run_stmt(command)
            indentation = "  " * (len(pending_commands_stack) + 1) if command.strip() != "Proof." else ""
            yield indentation + command.strip()
            if coq.count_fg_goals() > 1:
                pending_commands_stack.append(None)
                coq.run_stmt("{")
                yield indentation + "{"
    pass

def handle_with(command_batch : Iterable[str],
                with_tactic : str) -> Iterable[str]:
    if not with_tactic:
        for command in command_batch:
            yield re.sub("(.*)\s*\.\.\.", r"\1.", command)
    else:
        yield "Proof."
        for command in islice(command_batch, 1, None):
            newcommand = re.sub("(.*)\s*\.\.\.", rf"\1 ; {with_tactic}.", command)
            yield newcommand

def split_commas(command : str) -> str:
    rewrite_match = re.match("(.*)(?:\s|^)(rewrite\s+)([^;,]*?,\s*.*)", command,
                             flags=re.DOTALL)
    unfold_match = re.match("(.*)(unfold\s+)([^;,]*?),\s*(.*)", command,
                            flags=re.DOTALL)
    if rewrite_match:

        prefix, rewrite_command, id_and_rest = rewrite_match.group(1, 2, 3)
        split = split_by_char_outside_matching("\(", "\)", ",", id_and_rest)
        if not split:
            return command
        first_id, rest = split
        rest = rest[1:]
        split = split_by_char_outside_matching("\(", "\)", ";|\.", rest)
        assert split
        rewrite_rest, command_rest = split
        by_match = re.match("(.*)(\sby\s.*)", rewrite_rest)
        in_match = re.match("(.*)(\sin\s.*)", rewrite_rest)
        postfix = ""
        if by_match:
            if " by " not in first_id:
                postfix += by_match.group(2)
        if in_match:
            if " in " not in first_id:
                postfix += in_match.group(2)
        first_command = "(" + rewrite_command + first_id + " " + postfix + ")"
        result = prefix + first_command + ";" + split_commas(rewrite_command + rest)
        return result
    elif unfold_match:
        prefix, unfold_command, first_id, rest = unfold_match.group(1, 2, 3, 4)
        if re.search("\sin\s", unfold_command + first_id):
            return command
        split = split_by_char_outside_matching("\(", "\)", ";|\.", rest)
        assert split
        unfold_rest, command_rest = split
        in_match = re.match("(.*)(\sin\s.*)", unfold_rest)
        postfix = ""
        if in_match:
            if "in" not in first_id:
                postfix += in_match.group(2)
        first_command = unfold_command + first_id + " " + postfix
        return prefix + first_command + ";" + split_commas(unfold_command + rest)
    else:
        return command
def postlinear_desugar_tacs(commands :  Iterable[str]) -> Iterable[str]:
    yield from commands
def desugar_rewrite_by(cmd : str) -> str:
    rewrite_by_match = re.search(r"\b(rewrite\s*(?:!|<-)?.*?\s+)by\s+", cmd)
    if rewrite_by_match:
        prefix = cmd[:rewrite_by_match.start()]
        after_match = cmd[rewrite_by_match.end():]
        split = split_by_char_outside_matching("[\[(]", "[\])]",
                                               r"\.\W|\.$|[|]|\)|;", after_match)
        assert split
        body, postfix = split
        postfix = desugar_rewrite_by(postfix)
        return f"{prefix}{rewrite_by_match.group(1)} ; [|{body} ..] {postfix}"
    else:
        return cmd
def desugar_assert_by(cmd : str) -> str:
    assert_by_match = re.search(r"\b(assert.*)\s+by\s+", cmd)
    if assert_by_match:
        prefix = cmd[:assert_by_match.start()]
        after_match = cmd[assert_by_match.end():]
        split = split_by_char_outside_matching("[[(]", "[\])]",
                                               r"\.\W|\.$|[|]",
                                               after_match)
        assert split
        body, postfix = split
        return f"{prefix}{assert_by_match.group(1)} ; [{body}..|] {postfix}"
    else:
        return cmd
def desugar_now(command : str) -> str:
    now_match = re.search(r"\bnow\s+", command)
    while(now_match):
        prefix = command[:now_match.start()]
        after_match = command[now_match.end():]
        split = split_by_char_outside_matching("[[(]", "[\])]",
                                               r"\.\W|\.$|]|\||\)",
                                               after_match)
        assert split
        body, postfix = split
        command = f"{prefix}({body} ; easy){postfix}"
        now_match = re.search(r"\bnow\s+", command)
    return command
desugar_passes = [split_commas, desugar_now, desugar_rewrite_by, desugar_assert_by]
def prelinear_desugar_tacs(commands : Iterable[str]) -> Iterable[str]:
    for command in commands:
        comment_before_command = ""
        command_proper = command
        while "(*" in command_proper:
            next_comment, command_proper = \
                split_to_next_matching("\(\*", "\*\)", command_proper)
            comment_before_command += next_comment
        for f in desugar_passes:
            newcommand = f(command_proper)
            assert serapi_instance.isValidCommand(newcommand), (command_proper, newcommand)
            command_proper = newcommand
        full_command = comment_before_command + command_proper
        yield full_command

def lifted_vernac(command : str) -> Optional[Match[Any]]:
    return re.match("Ltac\s", serapi_instance.kill_comments(command).strip())

def generate_lifted(commands : List[str], coq : serapi_instance.SerapiInstance,
                    pbar : tqdm) \
    -> Iterator[str]:
    lemma_stack = [] # type: List[List[str]]
    for command in commands:
        if serapi_instance.possibly_starting_proof(command):
            coq.run_stmt(command)
            if coq.proof_context:
                lemma_stack.append([])
            coq.cancel_last()
        if len(lemma_stack) > 0 and not lifted_vernac(command):
            lemma_stack[-1].append(command)
        else:
            pbar.update(1)
            yield command
        if serapi_instance.ending_proof(command):
            pending_commands = lemma_stack.pop()
            pbar.update(len(pending_commands))
            yield from pending_commands
    assert(len(lemma_stack) == 0)

def preprocess_file_commands(args : argparse.Namespace, file_idx : int,
                             commands : List[str], coqargs : List[str], includes : str,
                             filename : str, relative_filename : str,
                             skip_nochange_tac : bool) -> List[str]:
    try:
        with serapi_instance.SerapiContext(coqargs, includes, args.prelude) as coq:
            coq.debug = args.debug
            with tqdm(file=sys.stdout,
                      disable=not args.progress,
                      position=(file_idx * 2),
                      desc="Linearizing", leave=False,
                      total=len(commands),
                      dynamic_ncols=True,
                      bar_format=mybarfmt) as pbar:
                result = list(
                    postlinear_desugar_tacs(
                        linearize_commands(
                            args, file_idx,
                            generate_lifted(commands, coq, pbar),
                            coq, filename, relative_filename,
                        skip_nochange_tac)))
        return result
    except (CoqExn, BadResponse, AckError, CompletedError):
        eprint("In file {}".format(filename))
        raise
    except serapi_instance.TimeoutError:
        eprint("Timed out while lifting commands! Skipping linearization...")
        return commands
def get_linearized(args : argparse.Namespace, coqargs : List[str], includes : str,
                   bar_idx : int, filename : str) -> List[str]:
    local_filename = args.prelude + "/" + filename
    loaded_commands = serapi_instance.try_load_lin(args, bar_idx, local_filename)
    if loaded_commands is None:
        original_commands = \
            serapi_instance.load_commands_preserve(args, bar_idx,
                                                   args.prelude + "/" + filename)
        fresh_commands = preprocess_file_commands(
            args, bar_idx,
            original_commands,
            coqargs, includes,
            local_filename, filename, False)
        serapi_instance.save_lin(fresh_commands, local_filename)
        return fresh_commands
    else:
        return loaded_commands


def main():
    parser = argparse.ArgumentParser(description=
                                     "linearize a set of files")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', default=False, const=True, action='store_const')
    parser.add_argument('--hardfail', default=False, const=True, action='store_const')
    parser.add_argument('--verbose', default=False, const=True, action='store_const')
    parser.add_argument('--skip-nochange-tac', default=False, const=True, action='store_const',
                        dest='skip_nochange_tac')
    parser.add_argument("--progress",
                        action='store_const', const=True, default=False)
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    arg_values = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__)) + "/.."
    includes = subprocess.Popen(['make', '-C', arg_values.prelude, 'print-includes'],
                                stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
    coqargs = ["sertop"]

    for filename in arg_values.filenames:
        if arg_values.verbose:
            eprint("Linearizing {}".format(filename))
        local_filename = arg_values.prelude + "/" + filename
        original_commands = serapi_instance.load_commands_preserve(
            arg_values, 0, arg_values.prelude + "/" + filename)
        fresh_commands = preprocess_file_commands(arg_values, 0,
                                                  original_commands,
                                                  coqargs, includes,
                                                  local_filename, filename, False)
        serapi_instance.save_lin(fresh_commands, local_filename)

if __name__ == "__main__":
    main()
