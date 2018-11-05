#!/usr/bin/env python3

import sys
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
    # Set up the model, and randomly initialize weights
    def __init__(self, num_states : int, num_emissions : int) -> None:
        self.num_states = num_states
        self.num_emissions = num_emissions
        self.randomly_initialize_weights()
        pass
    # Predict a state sequence of states of a given length
    def predictStates(self, length : int) -> List[int]:
        # To compute the optimal sequence of a particular length i,
        # we'll need to compute the likelyhood of being in each state,
        # at each timestep (given the emissions so far).
        back_pointers : List[List[int]] = []
        # Start off with our initial probabilities
        state_probabilities : List[float] = self.initial_probabilities
        for i in range(length):
            # For each state, lets find out the best way to get
            # there. We know how likely the previous states were, and
            # we know how likely our transitions are. So we can
            # compute the likelyhood of getting to any next state from
            # any previous state. For each next state, we'll figure
            # out the best previous state to get to it, and then save
            # the index of that state, and the probability of the new
            # state coming from that state.
            seq_probs_and_back_pointers = \
                [max([(state_probabilities[j] * self.transition_probabilities[(j, i)], j)
                      for j in range(self.num_states+1)], key=lambda p: p[0])
                 for i in range(self.num_states+1)]
            # Pull out the indices of, for each new state, the best
            # previous state to get there.
            back_pointers.append([back_pointer for prob, back_pointer
                                  in seq_probs_and_back_pointers])
            # Pull out the new probabilities for each state
            state_probabilities = [prob for prob, back_pointer in
                                   seq_probs_and_back_pointers]
        # The sequence always ends in the final state
        backwards_best_states : List[int] = [self.num_states]
        # From there, look through the backpointers to find out what
        # the best state to get to the final state is, and then the
        # best state to get to that one, and so on.
        for pointer_row in reversed(back_pointers):
            backwards_best_states.append(pointer_row[backwards_best_states[-1]])

        # Finally, strip off the final state and return it
        return list(reversed(backwards_best_states[1:]))

    # Find the most likely set of emissions of length `length` for this model
    def predictSequence(self, length : int) -> List[int]:
        # Since the states don't depend on the emissions, we can just
        # get the most likely states, and then return the most likely
        # emission from each.
        best_states = self.predictStates(length)

        return [max(range(self.num_emissions+1),
                    key=lambda e: self.emission_probabilities[(s, e)])
                for s in best_states]
    # Produce a state sequence by running the markov model
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

    # The backward likelyhoods. This is the likelyhoods, for any time
    # `t` and state `s`, of finishing the rest of the sequence from
    # there.
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
    # The forward likelyhoods. This is the likelyhoods, for any time
    # `t` and state `s`, of having seen the sequence so far.
    def forwardLikelyhoods(self, seq : List[int]) -> List[List[float]]:
        return forwardLikelyhoods(self.initial_probabilities,
                                  self.transition_probabilities,
                                  self.emission_probabilities,
                                  seq)
    # The likelyhood of being in each state, at each time, given a
    # sequence (and a model)
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

    # The likelyhood of every transition, at every time `t`
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
    # Recompute probabilities based on a set of sequences. This is
    # expensive, so it's set up to work across multiple cores.
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

    # Set the weights randomly, given some invariants about the weights
    def randomly_initialize_weights(self) -> None:
        # You can't start in the final state
        self.initial_probabilities : List[float] = \
            random_distribution(self.num_states) + [0]
        # You don't transition from the final state to anything
        self.transition_probabilities = {**{(i, j) : f
                                            for j, f in
                                            enumerate(random_distribution(
                                                self.num_states+1))
                                            for i in range(self.num_states)},
                                         **{(self.num_states, self.num_states) : 0},
                                         **{(self.num_states, j) : 0
                                            for j in range(self.num_states)}}
        self.emission_probabilities : Dict[Tuple[int, int], float]= {}
        # The final state doesn't emit anything but the final token
        for s in range(self.num_states):
            emission_distribution = random_distribution(self.num_emissions)
            for emission, probability in enumerate(emission_distribution):
                self.emission_probabilities[(s, emission)] = probability
            self.emission_probabilities[(s, self.num_emissions)] = 0
        for emission in range(self.num_emissions):
            self.emission_probabilities[(self.num_states, emission)] = 0
        self.emission_probabilities[(self.num_states, self.num_emissions)] = 1
    # This code is super useful for debugging, you just need to
    # install "graphviz" and run it locally.

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
        # Keep calling reestimate 100 times, and pick the best weights
        # (ones that minimize the loss)
        best_loss = float("+inf")
        best_initial = None
        best_transition = None
        best_emission = None
        for i in range(100):
            curtime = time.time()
            print("  Iteration {}:".format(i))
            sys.stdout.flush()

            new_initial, new_transition, new_emission = \
                self.reestimate([seq + [self.num_emissions] for seq in data], num_threads)
            if new_initial == self.initial_probabilities and \
               new_transition == self.transition_probabilities and \
               new_emission == self.emission_probabilities:
                break
            def individualLoss(seq : List[int]):
                return sum(forwardLikelyhoods(new_initial,
                                              new_transition,
                                              new_emission,
                                              seq)[-1])
            loss = sum([-math.log(individualLoss(seq))
                        for seq in data]) / len(data)
            print("  Iteration: {:.2f}s, loss {}".format(time.time() - curtime, loss))

            if loss < best_loss:
                best_initial = new_initial
                best_transition = new_transition
                best_emission = new_emission
                best_loss = loss
            self.initial_probabilities, \
                self.transition_probabilities, \
                self.emission_probabilities \
                = new_initial, new_transition, new_emission
        self.initial_probabilities = best_initial
        self.transition_probabilities = best_transition
        self.emission_probabilities = best_emission

    def encoding_length(self) -> int:
        return (self.num_states+1) + (self.num_states+1)**2 + \
            (self.num_states+1) * (self.num_emissions+1)
    # Encode a sequence by reestimating with it, and then using those
    # probabilities as the encoding (could also do the changes, but
    # it's essentially the same). Most of this code is the same as in
    # the reestimate stuff.
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

def forwardLikelyhoods(initial_probabilities : List[float],
                       transition_probabilities : Dict[Tuple[int, int], float],
                       emission_probabilities : Dict[Tuple[int, int], float],
                       seq : List[int]):
    probabilities : List[List[float]] = []

    # Set up the first row of probabilities
    probabilities.append([])
    for state_num, PrS in enumerate(initial_probabilities):
        # Probability of the emission 'seq[0]' in state 'state_num'
        PrEinS = emission_probabilities[(state_num, seq[0])] * PrS
        probabilities[0].append(PrEinS)

    # Get the rows after that
    for t, emission in enumerate(seq[1:], 1):
        # Set up the next row of probabilities
        probabilities.append([])

        for state_num in range(len(initial_probabilities)):
            # Probability of being in state 'state_num' at time
            # 't', and emitting 'emission'
            PrS = sum([transition_probabilities[(prev_state_num,
                                                 state_num)] * prevPrS
                       for prev_state_num, prevPrS in enumerate(probabilities[t-1])]) \
                           * emission_probabilities[(state_num, emission)]
            probabilities[t].append(PrS)
    return probabilities
