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
import re
import random
import sys

import serapi_instance
import dataloader
import tokenizer
from models import tactic_predictor
import predict_tactic
from util import maybe_cuda, eprint
from models.components import WordFeaturesEncoder, DNNScorer

from dataclasses import dataclass
from typing import List, Tuple, Iterator, TypeVar, Dict
from format import TacticContext
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torch import optim
from pathlib_revised import Path2

def main(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="A module for exploring deep Q learning with proverbot9001")

    parser.add_argument("scrape_file")

    parser.add_argument("environment_file", type=Path2)
    parser.add_argument("--proof", default=None)

    parser.add_argument("--prelude", default=".", type=Path2)

    parser.add_argument("--predictor-weights", default=Path2("data/polyarg-weights.dat"),
                        type=Path2)
    parser.add_argument("--num-predictions", default=16, type=int)

    parser.add_argument("--buffer-size", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=256)
    parser.add_argument("--episode-length", default=16)

    parser.add_argument("--learning-rate", default=0.5)

    args = parser.parse_args()

    reinforce(args)

def reinforce(args : argparse.Namespace) -> None:

    # Load the scraped (demonstrated) samples, the proof environment
    # commands, and the predictor
    replay_memory = assign_rewards(
        dataloader.tactic_transitions_from_file(args.scrape_file,
                                                args.buffer_size))
    env_commands = serapi_instance.load_commands_preserve(args, 0, args.prelude / args.environment_file)
    predictor = predict_tactic.loadPredictorByFile(args.predictor_weights)

    q_estimator = FeaturesQEstimator(args.learning_rate)
    epsilon = 0.3
    gamma = 0.9

    with serapi_instance.SerapiContext(
            ["sertop", "--implicit"],
            serapi_instance.get_module_from_filename(args.environment_file),
            str(args.prelude)) as coq:
        ## Get us to the correct proof context
        rest_commands, run_commands = coq.run_into_next_proof(env_commands)
        lemma_statement = run_commands[-1]
        if args.proof != None:
            while coq.cur_lemma_name != args.proof:
                if not rest_commands:
                    eprint("Couldn't find lemma {args.proof}! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, _ = coq.run_into_next_proof(env_commands)
        else:
            # Don't use lemmas without names (e.g. "Obligation")
            while coq.cur_lemma_name == "":
                if not rest_commands:
                    eprint("Couldn't find usable lemma! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, _ = coq.run_into_next_proof(env_commands)

        lemma_name = coq.cur_lemma_name

        for episode in range(args.num_episodes):
            for t in range(args.episode_length):
                context_before = coq.tactic_context(coq.local_lemmas[:-1])
                predictions = predictor.predictKTactics(context_before, args.num_predictions)
                if random.random() < epsilon:
                    ordered_actions = [p.prediction for p in
                                       random.sample(predictions, len(predictions))]
                    action = random.choice(predictions).prediction
                else:
                    q_choices = [(q_estimator(context_before, prediction.prediction),
                                  prediction.prediction)
                                 for prediction in predictions]
                    ordered_actions = [p[1] for p in
                                       sorted(q_choices, key=lambda q: q[0], reverse=True)]

                for try_action in ordered_actions:
                    try:
                        coq.run_stmt(try_action)
                        action = try_action
                        break
                    except (serapi_instance.ParseError, serapi_instance.CoqExn):
                        pass

                context_after = coq.tactic_context(coq.local_lemmas[:-1])

                replay_memory.append(assign_reward(context_before, context_after, action))
                transition_samples = sample_batch(replay_memory, args.batch_size)
                training_samples = assign_scores(transition_samples,
                                                 q_estimator, predictor,
                                                 args.num_predictions,
                                                 gamma)

                q_estimator.train(training_samples)
                pass

            # Clean up episode
            coq.run_stmt("Admitted.")
            coq.run_stmt(f"Reset {lemma_name}.")
            coq.run_stmt(lemma_statement)

@dataclass
class LabeledTransition:
    before : dataloader.ProofContext
    after : dataloader.ProofContext
    action : str
    reward : float

def sample_batch(transitions: List[LabeledTransition], k: int) -> List[LabeledTransition]:
    return random.sample(transitions, k)

def assign_reward(before: TacticContext, after: TacticContext, tactic: str) -> LabeledTransition:
    if after.goal == "":
        reward = 1000.0
    else:
        goal_size_reward = len(tokenizer.get_words(before.goal)) - \
            len(tokenizer.get_words(after.goal))
        num_hyps_reward = len(before.hypotheses) - len(after.hypotheses)
        reward = goal_size_reward * 3 + num_hyps_reward
    return LabeledTransition(before, after, tactic, reward)


def assign_rewards(transitions : List[dataloader.ScrapedTransition]) -> \
    List[LabeledTransition]:
    def generate() -> Iterator[LabeledTransition]:
        for transition in transitions:
            yield assign_reward(context_r2py(transition.before),
                                context_r2py(transition.after),
                                transition.tactic)

    return list(generate())

class QEstimator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, state: TacticContext, action: str) -> float:
        pass
    @abstractmethod
    def train(self, samples: List[Tuple[TacticContext, str, float]]) -> None:
        pass

def assign_scores(transitions: List[LabeledTransition],
                  q_estimator: QEstimator,
                  predictor: tactic_predictor.TacticPredictor,
                  num_predictions: int,
                  discount : float) -> List[Tuple[TacticContext, str, float]]:
    def generate() -> Iterator[Tuple[dataloader.ProofContext, str, float]]:
        for transition in transitions:
            ctxt = transition.after
            predictions = predictor.predictKTactics(ctxt, num_predictions)
            new_q = transition.reward + \
                discount * max([q_estimator(ctxt, prediction.prediction)
                                for prediction in predictions])
            yield transition.before, transition.action, new_q
    return list(generate())

def context_r2py(r_context : dataloader.ProofContext) -> TacticContext:
    return TacticContext(r_context.lemmas, r_context.tactics,
                         r_context.hyps, r_context.goal)

class FeaturesQEstimator(QEstimator):
    def __init__(self, learning_rate: float) -> None:
        self.model = FeaturesQModel(32, 128,
                                    2, 128, 3)
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate)
        self.criterion = nn.MSELoss()
        self.tactic_map = {}
        self.token_map = {}
        pass
    def __call__(self, state: TacticContext, action: str) -> float:
        state_word_features, vec_features = self._features(state)
        encoded_action = self._encode_action(state, action)
        all_word_features = list(encoded_action) + state_word_features
        output = self.model(torch.LongTensor([all_word_features]),
                            torch.FloatTensor([vec_features]))
        return output[0].item()
    def train(self, samples: List[Tuple[TacticContext, str, float]]) -> None:
        self.optimizer.zero_grad()
        state_word_features, vec_features = zip(*[self._features(state) for state, _, _ in samples])
        encoded_actions = [self._encode_action(state, action) for state, action, _ in samples]
        all_word_features = [list(ea) + swf for ea, swf in zip(encoded_actions, state_word_features)]
        outputs = self.model(torch.LongTensor(all_word_features),
                             torch.FloatTensor(vec_features))
        expected_outputs = maybe_cuda(torch.FloatTensor([output for _, _, output in samples]))
        loss = self.criterion(outputs, expected_outputs)
        loss.backward()
        self.optimizer.step()
    def _features(self, context: TacticContext) -> Tuple[List[int], List[float]]:
        if len(context.prev_tactics) > 0:
            prev_tactic = serapi_instance.get_stem(context.prev_tactics[-1])
            prev_tactic_index = emap_lookup(self.tactic_map, 32, prev_tactic)
        else:
            prev_tactic_index = 0
        if context.goal != "":
            goal_head_index = emap_lookup(self.token_map, 128, tokenizer.get_words(context.goal)[0])
        else:
            goal_head_index = 0
        goal_length_feature = min(len(tokenizer.get_words(context.goal)), 100) / 100
        num_hyps_feature = min(len(context.hypotheses), 30) / 30
        return [prev_tactic_index, goal_head_index], [goal_length_feature, num_hyps_feature]
    def _encode_action(self, context: TacticContext, action: str) -> Tuple[int, int]:
        stem, argument = serapi_instance.split_tactic(action)
        stem_idx = emap_lookup(self.tactic_map, 32, stem)
        all_premises = context.hypotheses + context.relevant_lemmas
        stripped_arg = argument.strip(".").strip()
        if stripped_arg == "":
            arg_idx = 0
        else:
            index_hyp_vars = dict(serapi_instance.get_indexed_vars_in_hyps(all_premises))
            if stripped_arg in index_hyp_vars:
                hyp_varw, _, rest = all_premises[index_hyp_vars[stripped_arg]].partition(":")
                arg_idx = emap_lookup(self.token_map, 128, tokenizer.get_words(rest)[0]) + 2
            else:
                goal_symbols = tokenizer.get_symbols(context.goal)
                if stripped_arg in goal_symbols:
                    arg_idx = emap_lookup(self.token_map, 128, stripped_arg) + 128 + 2
                else:
                    arg_idx = 1
        return stem_idx, arg_idx

T = TypeVar('T')

def emap_lookup(emap: Dict[T, int], size: int, item: T):
    if item in emap:
        return emap[item]
    elif len(emap) < size - 1:
        emap[item] = len(emap) + 1
        return emap[item]
    else:
        return 0

class FeaturesQModel(nn.Module):
    def __init__(self,
                 num_tactics : int,
                 num_tokens : int,
                 vec_features_size : int,
                 hidden_size : int,
                 num_layers : int) -> None:
        super().__init__()
        # Consider making the word embedding the same for all token-type inputs, also for tactic-type inputs
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder([num_tactics, num_tokens * 2 + 2,
                                 num_tactics, num_tokens],
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNScorer(hidden_size + vec_features_size,
                      hidden_size, num_layers))
    def forward(self,
                word_features_batch : torch.LongTensor,
                vec_features_batch : torch.FloatTensor) -> torch.FloatTensor:
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        scores = self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)), dim=1))\
        .view(vec_features_batch.size()[0])
        return scores

if __name__ == "__main__":
    main(sys.argv[1:])