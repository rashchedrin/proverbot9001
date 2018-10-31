#!/usr/bin/env python3

import random
import itertools
# from graphviz import Digraph
import inspect
import re
import functools

from typing import List, Iterable, Tuple, Dict, Optional

from util import *
import multiprocessing

eps = 0.001
class HiddenMarkovModel:
    def __init__(self, num_states : int, num_emissions : int) -> None:
        self.num_states = num_states
        self.num_emissions = num_emissions
        self.randomly_initialize_weights()
        pass
    def predictStates(self, length : int) -> List[int]:
        back_pointers : List[List[int]] = []
        state_probabilities : List[float] = self.initial_probabilities
        for i in range(length):
            seq_probs_and_back_pointers = \
                [max([(state_probabilities[j] * self.transition_probabilities[(j, i)], j)
                      for j in range(self.num_states+1)], key=lambda p: p[0])
                 for i in range(self.num_states+1)]
            back_pointers.append([back_pointer for prob, back_pointer
                                  in seq_probs_and_back_pointers])
            state_probabilities = [prob for prob, back_pointer in
                                   seq_probs_and_back_pointers]
        backwards_best_states : List[int] = [self.num_states]
        for pointer_row in reversed(back_pointers):
            backwards_best_states.append(pointer_row[backwards_best_states[-1]])

        return list(reversed(backwards_best_states[1:]))

    def predictSequence(self, length : int) -> List[int]:
        best_states = self.predictStates(length)

        return [max(range(self.num_emissions+1),
                    key=lambda e: self.emission_probabilities[(s, e)])
                for s in best_states]
    def sampleStates(self, length : int) -> List[int]:
        states = [sample_distribution(self.initial_probabilities)]
        for t in range(1, length):
            new_state = \
                sample_distribution(
                    [self.transition_probabilities[(states[-1], s)]
                     for s in range(self.num_states+1)])
            if new_state == self.num_states:
                break
            states.append(new_state)
        return states

    def sampleSequence(self, length : int) -> List[int]:
        sampled_states = self.sampleStates(length)
        return [sample_distribution([self.emission_probabilities[(s, e)] for
                                     e in range(self.num_emissions + 1)])
                for s in sampled_states]
    def backwardLikelyhoods(self, seq : List[int]) -> List[List[float]]:
        reversed_probabilities : List[List[float]] = []

        # Set up the first (last) row of probabilities
        reversed_probabilities.append([1.0]*(self.num_states+1))

        for emission in reversed(seq[1:]):
            reversed_probabilities.append([])
            for state_num in range(self.num_states+1):
                PrS = 0.0
                for next_state_num, nextPrS in enumerate(reversed_probabilities[-2]):
                    PrLeavingToNext = (self.transition_probabilities[(state_num,
                                                                      next_state_num)]
                                       * nextPrS
                                       * self.emission_probabilities[(next_state_num,
                                                                      emission)])
                    PrS += PrLeavingToNext
                reversed_probabilities[-1].append(PrS)

        return list(reversed(reversed_probabilities))
    def forwardLikelyhoods(self, seq : List[int]) -> List[List[float]]:
        probabilities : List[List[float]] = []

        # Set up the first row of probabilities
        probabilities.append([])
        for state_num, PrS in enumerate(self.initial_probabilities):
            # Probability of the emission 'seq[0]' in state 'state_num'
            PrEinS = self.emission_probabilities[(state_num, seq[0])] * PrS
            probabilities[0].append(PrEinS)

        # Get the rows after that
        for t, emission in enumerate(seq[1:], 1):
            # Set up the next row of probabilities
            probabilities.append([])

            for state_num in range(self.num_states+1):
                # Probability of being in state 'state_num' at time
                # 't', and emitting 'emission'
                PrS = sum([self.transition_probabilities[(prev_state_num,
                                                          state_num)] * prevPrS
                           for prev_state_num, prevPrS in enumerate(probabilities[t-1])]) \
                               * self.emission_probabilities[(state_num, emission)]
                probabilities[t].append(PrS)
        return probabilities
    def forwardLikelyhood(self, seq : List[int]) -> float:
        return sum(self.forwardLikelyhoods(seq)[-1])
    def individualStateLikelyhoods(self, seq : List[int]) -> List[List[float]]:
        forwardProbabilities = self.forwardLikelyhoods(seq)
        backwardProbabilities = self.backwardLikelyhoods(seq)
        probabilities : List[List[float]] = []
        for t, (emission, forward_t, backward_t) in \
            enumerate(zip(seq, forwardProbabilities, backwardProbabilities)):
            unnormalized_probabilities = [forward_prob * backward_prob
                                           for forward_prob, backward_prob
                                           in zip(forward_t, backward_t)]

            total_prob = sum(unnormalized_probabilities)
            assert total_prob > 0, "Can't run on a sequence with zero probability in existing model!"
            probabilities.append([unnormalized_prob / total_prob if total_prob > 0 else 0
                                  for unnormalized_prob in unnormalized_probabilities])

        return probabilities

    def expectedTransitionLikelyhoods(self, seq : List[int]) -> \
        List[Dict[Tuple[int, int], float]]:
        forwardProbabilities = self.forwardLikelyhoods(seq)
        backwardProbabilities = self.backwardLikelyhoods(seq)
        probabilities : List[Dict[Tuple[int, int], float]] = []
        for t, (emission_t_plus_1, forward_t, backward_t_plus_1) in \
            enumerate(zip(seq[1:], forwardProbabilities[:-1], backwardProbabilities[1:])):

            unnormalized_probabilities : Dict[Tuple[int, int], float] = {}
            for state_num_i, forward_prob in enumerate(forward_t):
                for state_num_j, backward_prob in enumerate(backward_t_plus_1):
                    unnormalized_probabilities[(state_num_i, state_num_j)] = \
                        forward_prob * self.transition_probabilities[(state_num_i, state_num_j)] * \
                        self.emission_probabilities[(state_num_j, emission_t_plus_1)] * backward_prob
            total_prob = sum(unnormalized_probabilities.values())
            probabilities.append({key : (value / total_prob if total_prob > 0 else 0)
                                  for (key, value) in unnormalized_probabilities.items()})

        return probabilities
    def reestimate(self, sequences : List[List[int]], num_threads : int) -> \
        Tuple[List[float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        with multiprocessing.Pool(num_threads) as pool:
            sequenceStateLikelyhoods = \
                list(itertools.chain.from_iterable(pool.imap_unordered(
                    functools.partial(listmap, self.individualStateLikelyhoods),
                    chunks(sequences, 100))))
            sequenceTransitionLikelyhoods = \
                list(itertools.chain.from_iterable(pool.imap_unordered(
                    functools.partial(listmap, self.expectedTransitionLikelyhoods),
                    chunks(sequences, 10))))
        num_states_visited_total = sum([len(seq) for seq in sequences])
        new_initial = [sum(likelyhoods) / len(likelyhoods) for likelyhoods in
                       zip(*[stateLikelyhood[0] for stateLikelyhood in
                             sequenceStateLikelyhoods])]
        with multiprocessing.Pool(num_threads) as pool:
            transition_probabilities = \
                list(itertools.chain.from_iterable(pool.map(
                    functools.partial(state_chunk_transition_probabilities,
                                      num_states_visited_total,
                                      self.num_states,
                                      sequenceStateLikelyhoods,
                                      sequenceTransitionLikelyhoods),
                    chunks(range(self.num_states+1), math.ceil(self.num_states / 10)))))
        new_transitions = {(state_num_i, state_num_j) :
                           transition_probabilities[state_num_i][state_num_j]
                           for (state_num_i, state_num_j) in self.transition_probabilities}
        with multiprocessing.Pool(num_threads) as pool:
            emission_probabilities = \
                list(itertools.chain.from_iterable(pool.map(
                    functools.partial(state_chunk_emission_probabilities,
                                      num_states_visited_total,
                                      self.num_emissions,
                                      sequenceStateLikelyhoods,
                                      sequences),
                    chunks(range(self.num_states+1), math.ceil(self.num_states / 10)))))

        new_emission = {(state_num, emission) :
                        emission_probabilities[state_num][emission]
                        for (state_num, emission) in self.emission_probabilities}

        return new_initial, new_transitions, new_emission

    def randomly_initialize_weights(self) -> None:
        self.initial_probabilities : List[float] = \
            random_distribution(self.num_states) + [0]
        self.transition_probabilities = {**{(i, j) : f
                                            for j, f in
                                            enumerate(random_distribution(
                                                self.num_states+1))
                                            for i in range(self.num_states)},
                                         **{(self.num_states, self.num_states) : 0},
                                         **{(self.num_states, j) : 0
                                            for j in range(self.num_states)}}
        self.emission_probabilities : Dict[Tuple[int, int], float]= {}
        for s in range(self.num_states):
            emission_distribution = random_distribution(self.num_emissions)
            for emission, probability in enumerate(emission_distribution):
                self.emission_probabilities[(s, emission)] = probability
            self.emission_probabilities[(s, self.num_emissions)] = 0
        for emission in range(self.num_emissions):
            self.emission_probabilities[(self.num_states, emission)] = 0
        self.emission_probabilities[(self.num_states, self.num_emissions)] = 1
    # def view(self, path : str) -> None:
    #     def state_str(s : int):
    #         if s == self.num_states:
    #             return "end"
    #         else:
    #             return str(s)
    #     def em_str(e : int):
    #         return chr(e+65)
    #     graph = Digraph(filename=path)
    #     for (a, b), probability in self.transition_probabilities.items():
    #         if probability > eps:
    #             graph.edge(state_str(a), state_str(b),
    #                        label="{:.1f}%".format(probability * 100),
    #                        penwidth="{:.1f}".format(4 * probability + 1))
    #     for (s, e), probability in self.emission_probabilities.items():
    #         if probability > eps and s != self.num_states:
    #             graph.edge(state_str(s), em_str(e),
    #                        label="{:.1f}%".format(probability * 100),
    #                        color="blue",
    #                        penwidth="{:.1f}".format(4 * probability + 1))
    #     for s, probability in enumerate(self.initial_probabilities):
    #         if probability > eps:
    #             graph.edge("start", state_str(s),
    #                        label="{:.1f}%".format(probability * 100),
    #                        color="green",
    #                        penwidth="{:.1f}".format(4 * probability + 1))
    #     graph.view()
    #     pass
    def train(self, data : List[List[int]],
              num_threads : Optional[int] = None) -> None:
        for i in range(100):
            new_initial, new_transition, new_emission = \
                self.reestimate([seq + [self.num_emissions] for seq in data])
            if new_initial == self.initial_probabilities and \
               new_transition == self.transition_probabilities and \
               new_emission == self.emission_probabilities:
                break
            self.initial_probabilities, \
                self.transition_probabilities, \
                self.emission_probabilities \
                = new_initial, new_transition, new_emission
    def encoding_length(self) -> int:
        return (self.num_states+1) + (self.num_states+1)**2 + \
            (self.num_states+1) * (self.num_emissions+1)
    def encode(self, sequence : List[int]) -> List[float]:
        sequence = sequence + [self.num_emissions]
        stateLikelyhoods = self.individualStateLikelyhoods(sequence)
        overallStateLikelyhoods = [sum(likelyhoods_i) for likelyhoods_i
                                     in zip(*stateLikelyhoods)]
        transitionLikelyhoods = self.expectedTransitionLikelyhoods(sequence)
        new_initial = stateLikelyhoods[0]
        assert len(new_initial) == self.num_states + 1
        new_transitions : List[float] = []
        for state_num_i, overallStateLikelyhood in enumerate(overallStateLikelyhoods):
            if overallStateLikelyhood == 0:
                new_transitions += [0] * (self.num_states+1)
            else:
                new_transitions += [sum([transitionLikelyhood[(state_num_i, state_num_j)]
                                         for transitionLikelyhood
                                         in transitionLikelyhoods])
                                    / overallStateLikelyhood
                                    for state_num_j in range(self.num_states+1)]
        assert len(new_transitions) == (self.num_states+1)**2, \
            "len(new_transitions): {}".format(len(new_transitions))
        new_emissions : List[float] = []
        for state_num, overallStateLikelyhood in enumerate(overallStateLikelyhoods):
            if overallStateLikelyhood == 0:
                new_emissions += [0] * (self.num_emissions+1)
            else:
                new_emissions += [sum([stateLikelyhood_t[state_num]
                                       if emission == O_t else 0
                                       for O_t, stateLikelyhood_t
                                       in zip(sequence, stateLikelyhoods)])
                                  / overallStateLikelyhood
                                  for emission in range(self.num_emissions+1)]
        assert len(new_emissions) == (self.num_states+1)*(self.num_emissions+1)
        return new_initial + new_transitions + new_emissions

def state_chunk_transition_probabilities(num_states_visited_total : int,
                                         num_states : int,
                                         sequenceStateLikelyhoods :
                                         List[List[List[float]]],
                                         sequenceTransitionLikelyhoods :
                                         List[List[Dict[Tuple[int, int], float]]],
                                         state_num_chunk : List[int]) -> List[List[float]]:
    transition_probabilities : List[List[float]] = []
    for state_num_i in state_num_chunk:
        total_likelyhood_i = sum([sum([stateLikelyhood[state_num_i]
                                       for stateLikelyhood in stateLikelyhoods])
                                  for stateLikelyhoods in sequenceStateLikelyhoods])
        expected_times_in_state = num_states_visited_total * total_likelyhood_i
        smoothing_factor = 1 / expected_times_in_state
        transition_probabilities.append([])
        for state_num_j in range(num_states+1):
            transition_likelyhood_i_j = sum([sum([transitionLikelyhood
                                                  [(state_num_i, state_num_j)]
                                                  for transitionLikelyhood
                                                  in transitionLikelyhoods])
                                             for transitionLikelyhoods in
                                             sequenceTransitionLikelyhoods])
            transition_probabilities[-1].append(
                (transition_likelyhood_i_j + smoothing_factor) /
                (total_likelyhood_i + smoothing_factor))
    return transition_probabilities

def state_chunk_emission_probabilities(num_states_visited_total : int,
                                       num_emissions : int,
                                       sequenceStateLikelyhoods : List[List[List[float]]],
                                       sequences : List[List[int]],
                                       state_num_chunk : List[int]) -> List[List[float]]:
    emission_probabilities : List[List[float]] = []
    for state_num in state_num_chunk:
        total_state_likelyhood = \
            sum([sum([t_likelyhood[state_num] for t_likelyhood in stateLikelyhoods])
                 for stateLikelyhoods in sequenceStateLikelyhoods])
        expected_times_in_state = num_states_visited_total * total_state_likelyhood
        emission_probabilities.append([])
        for emission in range(num_emissions+1):
            emission_likelyhood_in_state = 0
            for t, (seq_ts, stateLikelyhoods_t) in \
                enumerate(zip(zip(*sequences), zip(*sequenceStateLikelyhoods))):
                emission_likelyhood_at_t_in_state = \
                    sum([stateLikelyhood_t[state_num]
                         if O_t == emission else 0
                         for O_t, stateLikelyhood_t
                         in zip(list(seq_ts), list(stateLikelyhoods_t))])
                emission_likelyhood_in_state += \
                    emission_likelyhood_at_t_in_state
            smoothing_factor = 1 / expected_times_in_state
            new_emission_prob = ((emission_likelyhood_in_state + smoothing_factor)
                                 /
                                 (total_state_likelyhood + smoothing_factor))

            emission_probabilities[-1].append(new_emission_prob)
    return emission_probabilities

def random_distribution(size : int) -> List[float]:
    random_floats = [random.random() for _ in range(size)]
    floatsum = sum(random_floats)
    return [rfloat / floatsum for rfloat in random_floats]

def sample_distribution(distribution : List[float]):
    continous_sample = random.random()
    for index, probability in enumerate(distribution):
        if continous_sample < probability:
            return index
        else:
            continous_sample -= probability
# model = HiddenMarkovModel(2, 3)
# model.train([[0, 1, 0, 1, 0, 1, 0],
#              [1, 0, 1, 0, 1, 0],
#              [0, 1, 0, 1, 0, 1],
#              [0, 1, 0, 1, 0, 1, 0, 1],
#              [0, 1, 0, 1, 0, 1, 0, 1, 0],
#              [1, 0, 1, 0, 1, 0, 1, 0],
#              ])
# debug = True
