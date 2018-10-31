#!/usr/bin/env python3

import argparse
import time
import threading
import math
import sys
import multiprocessing
import functools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as torchdata
import torch.cuda
import models.args as stdargs
import data
from context_filter import get_context_filter
import tokenizer as tk

from util import *
from models.components import SimpleEmbedding, ClassifierDNN
from models.tactic_predictor import TacticPredictor
from models.markov_encoder import HiddenMarkovModel
from serapi_instance import get_stem

from typing import Dict, List, Union, Any, Tuple, Iterable, Callable, NamedTuple
from typing import cast, overload

class HMMEncoderClassifier(TacticPredictor):
    def load_saved_state(self, state_filename : str) -> None:
        checkpoint = torch.load(state_filename)

        # HMM Encoder parameters
        assert checkpoint['context-filter']
        assert checkpoint['num-keywords']
        assert checkpoint['tokenizer-name']
        assert checkpoint['num-hidden-states']
        assert checkpoint['max-length']

        # Classifier parameters
        assert checkpoint['num-decoder-layers']
        assert checkpoint['decoder-hidden-size']
        assert checkpoint['learning-rate']
        assert checkpoint['optimizer-name']

        # Results of training
        assert checkpoint['tokenizer']
        assert checkpoint['stem-embedding']
        assert checkpoint['hmm-encoder']
        assert checkpoint['decoder']
        assert checkpoint['training-loss']
        assert checkpoint['epoch']

        self.options = [("tokenizer", checkpoint['tokenizer-name']),
                        ("# input keywords", checkpoint['num-keywords']),
                        ("# encoder hidden states", checkpoint['num-hidden-states']),
                        ("decoder hidden size", checkpoint['decoder-hidden-size']),
                        ("# decoder layers", checkpoint['num-decoder-layers']),
                        ("context filter", checkpoint['context-filter']),
                        ("optimizer (decoder)",
                         checkpoint['optimizer-name']),
                        ("learning rate (decoder)",
                         checkpoint['learning-rate']),
                        ("training loss (classifier)",
                         "{:.4f}".format(checkpoint['training-loss'])),
                        ("# epochs (classifier)",
                         checkpoint['epoch'] + 1)]

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['stem-embedding']

        self.encoder = checkpoint['hmm-encoder']

        self.decoder = maybe_cuda(ClassifierDNN(self.encoder.encoding_length(),
                                                checkpoint['decoder-hidden-size'],
                                                self.embedding.num_tokens(),
                                                checkpoint['num-decoder-layers']))
        self.decoder.load_state_dict(checkpoint['decoder'])

        self.context_filter = checkpoint['context-filter']
        self.max_length = checkpoint['max-length']

    def __init__(self, options : Dict[str, Any]) -> None:
        self.load_saved_state(options["filename"])
        self.criterion = maybe_cuda(nn.NLLLoss())
        self.lock = threading.Lock()

    def predictDistribution(self, in_data : Dict[str, Union[List[str], str]]) \
        -> torch.FloatTensor:
        return self.decoder.run(FloatTensor(self.encoder.encode(
            self.tokenizer.toTokenList(in_data["goal"])[:self.max_length]))).view(1, -1)

    def predictKTactics(self, in_data : Dict[str, Union[List[str], str]], k : int) \
                        -> List[Tuple[str, float]]:
        self.lock.acquire()
        prediction_distribution = self.predictDistribution(in_data)
        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        results = [(self.embedding.decode_token(stem_idx.data[0]) + ".",
                    math.exp(certainty.data[0]))
                   for certainty, stem_idx in zip(*certainties_and_idxs)]
        self.lock.release()
        return results

    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[List[str], str]],
                                k : int, correct : str) -> \
                                Tuple[List[Tuple[str, float]], float]:
        self.lock.acquire()
        prediction_distribution = self.predictDistribution(in_data)
        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        correct_stem = get_stem(correct)
        if self.embedding.has_token(correct_stem):
            output_var = maybe_cuda(Variable(
                torch.LongTensor([self.embedding.encode_token(correct_stem)])))
            loss = self.criterion(prediction_distribution, output_var).item()
        else:
            loss = 0

        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        results = [(self.embedding.decode_token(stem_idx.item()) + ".",
                    math.exp(certainty.item()))
                   for certainty, stem_idx in zip(*certainties_and_idxs)]

        self.lock.release()
        return results, loss

    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

class Checkpoint(NamedTuple):
    classifier_state : Dict[Any, Any]
    training_loss : float

def train_classifier(dataset : List[Tuple[List[float], int]],
                     encoding_size : int,
                     classifier_hidden_size : int, classifier_num_layers : int,
                     output_vocab_size : int,
                     batch_size : int, learning_rate : float, gamma : float,
                     epoch_step : int, num_epochs : int, print_every : int,
                     optimizer_f : Callable[..., Optimizer]) \
          -> Iterable[Checkpoint]:
    print("Encoding data...")
    in_stream, out_stream = zip(*dataset)
    print("Initializing PyTorch...")
    dataloader = \
        torchdata.DataLoader(torchdata.TensorDataset(torch.FloatTensor(in_stream),
                                                     torch.LongTensor(out_stream)),
                             batch_size=batch_size, num_workers=0,
                             shuffle=True, pin_memory=True, drop_last=True)
    classifier = maybe_cuda(ClassifierDNN(encoding_size,
                                          classifier_hidden_size, output_vocab_size,
                                          classifier_num_layers, batch_size))

    optimizer = optimizer_f(classifier.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, epoch_step, gamma)


    start=time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training classifier...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        adjuster.step()
        for batch_num, (input_batch, output_batch) in enumerate(dataloader):
            # Reset the optimizer
            optimizer.zero_grad()

            # Run the classifier on encoded vectors
            prediction_distribution = classifier.run(input_batch)

            # Get the loss
            output_var = maybe_cuda(Variable(output_batch))
            loss = criterion(prediction_distribution, output_var)

            # Update the weights
            loss.backward()
            optimizer.step()

            # Report progress
            items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
            total_loss += loss.item() * batch_size
            assert isinstance(total_loss, float)

            if (batch_num + 1) % print_every == 0:

                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield Checkpoint(classifier_state=classifier.state_dict(),
                         training_loss=total_loss / ((epoch + 1)* len(dataset)))

def use_tokenizer(tokenizer : tk.Tokenizer, term_strings : str):
    return [tokenizer.toTokenList(term_string)
            for term_string in term_strings]

def encode_point_batch(encoder : HiddenMarkovModel, tokenizer : tk.Tokenizer,
                       max_length : int,
                       point_batch : Tuple[str, int]) -> \
                       List[Tuple[List[float], int]]:
    encoded_batch : List[Tuple[List[float], int]] = []
    for goal, tactic_id in point_batch:
        encoded_batch.append((encoder.encode(tokenizer.toTokenList(goal)[:max_length]), tactic_id))
    return encoded_batch

def main(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "Classifier using hidden markov models as encoders")
    stdargs.add_std_args(parser)
    parser.add_argument("--num-hidden-states", dest="num_hidden_states",
                        default=None, type=int)
    parser.add_argument("--num-classifier-layers", dest="num_classifier_layers",
                        default=3, type=int)
    parser.add_argument("--classifier-hidden-size", dest="classifier_hidden_size",
                        default=128, type=int)
    parser.add_argument("--encoder-max-terms", dest="encoder_max_terms",
                        default=None, type=int)
    parser.add_argument("--hmm-weightsfile", dest="hmm_weightsfile",
                        default=None, type=str)
    parser.add_argument("--num-threads", "-j", dest="num_threads",
                        default=None, type=int)
    parser.add_argument("--train-encoder-only", dest="train_encoder_only",
                        default=False, action='store_const', const=True)
    args = parser.parse_args(arg_list)
    if not args.num_hidden_states:
        args.num_hidden_states = args.num_keywords

    print("Loading data...")
    raw_data = list(data.read_text_data(args.scrape_file, args.max_tuples))
    print("Read {} raw input-output pairs".format(len(raw_data)))
    print("Filtering data based on predicate...")
    filtered_data = list(data.filter_data(raw_data,
                                          get_context_filter(args.context_filter)))
    print("{} input-output pairs left".format(len(filtered_data)))

    print("Extracting all terms...")
    term_strings = list(itertools.chain.from_iterable(
        [[hyp.split(":")[1].strip() for hyp in hyps] + [goal]
         for hyps, goal, tactic in filtered_data]))

    curtime = time.time()
    print("Building tokenizer...", end="")
    sys.stdout.flush()
    tokenizer = tk.make_keyword_tokenizer_topk(term_strings,
                                               tk.tokenizers[args.tokenizer],
                                               args.num_keywords, 0)
    print(" {:.2f}s".format(time.time() - curtime))

    if args.hmm_weightsfile:
        hmm_weights = torch.load(args.hmm_weightsfile)
        assert hmm_weights['hmm-encoder']

        encoder = hmm_weights['hmm-encoder']
        args.max_length = hmm_weights['max-length']
        args.num_hidden_states = hmm_weights['num-hidden-states']
        args.num_keywords = hmm_weights['num-keywords']
        print("Loaded existing hmm encoder from {}".format(args.hmm_weightsfile))
    else:
        curtime = time.time()
        print("Tokenizing {} strings...".format(len(term_strings)), end="")
        sys.stdout.flush()
        with multiprocessing.Pool(args.num_threads) as pool:
            tokenized_data_chunks = pool.imap_unordered(functools.partial(
                use_tokenizer, tokenizer),
                                                        chunks(term_strings, 32768))
            tokenized_data = list(itertools.chain.from_iterable(tokenized_data_chunks))
        print(" {:.2f}s".format(time.time() - curtime))

        for seq in tokenized_data:
            assert seq[0] < args.num_keywords + 1

        encoder = HiddenMarkovModel(args.num_hidden_states, args.num_keywords + 1)

        curtime = time.time()
        print("Training encoder...")
        sys.stdout.flush()
        truncated_data = tokenized_data
        if args.max_length:
            truncated_data = [seq[:args.max_length] for seq in tokenized_data]
        if args.encoder_max_terms:
            truncated_data = truncated_data[:args.encoder_max_terms]
        encoder.train(truncated_data, args.num_threads)
        print("Total: {:.2f}s".format(time.time() - curtime))

        if args.train_encoder_only:
            with open(args.save_file, 'wb') as f:
                torch.save({'hmm-encoder': encoder,
                            'max-length': args.max_length,
                            'num-hidden-states':args.num_hidden_states,
                            'num-keywords': args.num_keywords})
            return

    embedding = SimpleEmbedding()

    curtime = time.time()
    print("Tokenizing/encoding data pairs...", end="")
    sys.stdout.flush()
    with multiprocessing.Pool(args.num_threads) as pool:
        dataset = list(itertools.chain.from_iterable(pool.imap_unordered(
            functools.partial(encode_point_batch, encoder, tokenizer, args.max_length),
            chunks([(goal, embedding.encode_token(get_stem(tactic)))
                    for hyps, goal, tactic in filtered_data],
                   math.ceil(len(filtered_data)/100)))))
    print(" {:.2f}s".format(time.time() - curtime))

    checkpoints = train_classifier(dataset, encoder.encoding_length(),
                                   args.classifier_hidden_size, args.num_classifier_layers,
                                   embedding.num_tokens(),
                                   args.batch_size, args.learning_rate, args.gamma,
                                   args.epoch_step, args.num_epochs, args.print_every,
                                   stdargs.optimizers[args.optimizer])
    for epoch, (decoder_state, training_loss) in enumerate(checkpoints):
        state = {
            # HMM Encoder parameters
            'context-filter' : args.context_filter,
            'num-keywords' : args.num_keywords,
            'tokenizer-name' : args.tokenizer,
            'num-hidden-states' : args.num_hidden_states,
            'max-length' : args.max_length,
            # Classifier parameters
            'num-decoder-layers' : args.num_classifier_layers,
            'decoder-hidden-size' : args.classifier_hidden_size,
            'learning-rate' : args.learning_rate,
            'optimizer-name' : args.optimizer,
            # Results of training
            'tokenizer' : tokenizer,
            'stem-embedding' : embedding,
            'hmm-encoder' : encoder,
            'decoder' : decoder_state,
            'training-loss' : training_loss,
            'epoch': epoch,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)
    pass
