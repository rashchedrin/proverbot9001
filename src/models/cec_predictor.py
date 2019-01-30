#!/usr/bin/env python3

from models.tactic_predictor import (NeuralClassifier, TacticContext,
                                     Prediction)
from models.components import Embedding
from data import (Sentence, Dataset, TokenizedDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *
from tokenizer import Tokenizer

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast)
from dataclasses import dataclass
import threading
import argparse
from argparse import Namespace

'''
CEC : Context EncClass -- look only at the context when making predictions 
The context is represented as a Sentence for now
'''

class CECSample(NamedTuple):
    context : Sentence
    next_tactic : int

@dataclass(init=True, repr=True)
class CECDataset(Dataset):
    data : List[CECSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class CEClassifier(nn.Module):
    def __init__(self, goal_vocab_size : int, hidden_size : int,
                 tactic_vocab_size : int, num_encoder_layers : int,
                 num_decoder_layers : int) -> None:
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.embedding = maybe_cuda(nn.Embedding(goal_vocab_size, hidden_size))
        self.tactic_embedding = maybe_cuda(nn.Embedding(tactic_vocab_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        self.squish = maybe_cuda(nn.Linear(hidden_size * 2, hidden_size))
        self.decoder_layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                               for _ in range(num_decoder_layers-1)]
        self.decoder_out = maybe_cuda(nn.Linear(hidden_size, tactic_vocab_size))

    def forward(self, input : torch.LongTensor, hidden : torch.FloatTensor) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor] :
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.num_encoder_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        return output[0], hidden

    def initHidden(self):
        return maybe_cuda(Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def run(self, input : torch.LongTensor, batch_size : int=1):
        self.batch_size = batch_size
        in_var = maybe_cuda(Variable(input))
        hidden = self.initHidden()
        for i in range(in_var.size()[1]):
            output, hidden = self(in_var[:,i], hidden)
        decoded = self.decoder_out(output)
        return self.softmax(decoded).view(self.batch_size, -1)

    # def forward(self, hyp_tokens : torch.LongTensor) \
    #     -> torch.FloatTensor:
    #     batch_size = hyp_tokens.size()[0]
    #     hidden = maybe_cuda(Variable(torch.zeros(1, batch_size, self.hidden_size)))
    #     for i in range(hyp_tokens.size()[1]):
    #         hyp_data = self.goal_embedding(hyp_tokens[:,i])\
    #                         .view(1, batch_size, self.hidden_size)
    #         for _ in range(self.num_encoder_layers):
    #             hyp_data = F.relu(hyp_data)
    #             hyp_data, hidden = self.gru(hyp_data, hidden)

    #     hyp_output = hyp_data[0]

    #     full_data = hyp_output
    #     for i in range(self.num_decoder_layers-1):
    #         full_data = F.relu(full_data)
    #         full_data = self.decoder_layers[i](full_data)
    #     return self.softmax(self.decoder_out(F.relu(full_data)))

    # def run(self,
    #         hyp_tokens : torch.LongTensor):
    #     return self(maybe_cuda(hyp_tokens))

class CECPredictor(NeuralClassifier[CECDataset, 'CEClassifier']):
    def _predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        tokenized_hyp = self._tokenizer.toTokenList("\n".join(in_data.hypotheses))
        hyp_list = normalizeSentenceLength(tokenized_hyp, self.training_args.max_length)
        hyp_tensor = LongTensor(hyp_list).view(1, -1)
        return self._model.run(hyp_tensor)

    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        if len(in_data) == 0:
            return [], 0
        with self._lock:
            hyp_tensor = LongTensor([
                normalizeSentenceLength(self._tokenizer.toTokenList("\n".join(hypotheses)),
                                        self.training_args.max_length)
                for _, hypotheses, _ in in_data])
            correct_stems = [get_stem(correct) for correct in corrects]
            prediction_distributions = self._model.run(hyp_tensor)
            output_var = maybe_cuda(Variable(
                torch.LongTensor([self._embedding.encode_token(correct_stem)
                                  if self._embedding.has_token(correct_stem)
                                  else 0
                                  for correct_stem in correct_stems])))
            loss = self._criterion(prediction_distributions, output_var).item()
            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()
            certainties_and_idxs_list = [single_distribution.view(-1).topk(k)
                                         for single_distribution in
                                         list(prediction_distributions)]
            results = [[Prediction(self._embedding.decode_token(stem_idx.item()) + ".",
                                   math.exp(certainty.item()))
                        for certainty, stem_idx in zip(*certainties_and_idxs)]
                       for certainties_and_idxs in certainties_and_idxs_list]
        return results, loss
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 100))
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int,
                            default=default_values.get("num-encoder-layers", 3))
        parser.add_argument("--num-decoder-layers", dest="num_decoder_layers", type=int,
                            default=default_values.get("num-decoder-layers", 3))
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               tokenizer : Tokenizer, embedding : Embedding) \
        -> CECDataset:
        return CECDataset([CECSample(context, tactic) for _, context, _, tactic in data])
    def _data_tensors(self, encoded_data : CECDataset, arg_values : Namespace) \
        -> List[torch.Tensor]:
        hyps, nexts = zip(*encoded_data)
        hyp_stream = torch.LongTensor([
            normalizeSentenceLength(hyp, arg_values.max_length)
            for hyp in hyps])
        out_stream = torch.LongTensor(nexts)
        return [hyp_stream, out_stream]

    def _get_model(self, arg_values : Namespace, tactic_vocab_size : int,
                   term_vocab_size : int) \
                   -> CEClassifier:
        return CEClassifier(term_vocab_size, arg_values.hidden_size, tactic_vocab_size,
                            arg_values.num_encoder_layers, arg_values.num_decoder_layers)

    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : CEClassifier) \
                                -> torch.FloatTensor:
        hyp_batch, output_batch = \
            cast(Tuple[torch.LongTensor, torch.LongTensor], data_batch)
        predictionDistribution = model.run(hyp_batch, len(hyp_batch))
        output_var = maybe_cuda(Variable(output_batch))
        return self._criterion(predictionDistribution, output_var)

    def _description(self) -> str:
       return "context encclass classifier pytorch model for proverbot"

def main(arg_list : List[str]) -> None:
    predictor = CECPredictor()
    predictor.train(arg_list)
