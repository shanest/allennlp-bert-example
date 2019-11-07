import unittest
from typing import List, Tuple, Union, Iterator, Dict

import glob

import numpy as np
import torch
import torch.optim as optim

from allennlp.data import Instance
from allennlp.data.fields \
        import TextField, SequenceLabelField, ArrayField, ListField, SpanField

from allennlp.data.vocabulary import Vocabulary

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator

from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.pretrained_transformer_tokenizer \
        import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer \
        import PretrainedTransformerIndexer

from allennlp.models import Model
from allennlp.modules.text_field_embedders \
        import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

from allennlp.modules.token_embedders.pretrained_transformer_embedder \
        import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders.boe_encoder \
        import BagOfEmbeddingsEncoder

from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.span_extractors.endpoint_span_extractor \
        import EndpointSpanExtractor

from allennlp.modules.time_distributed import TimeDistributed

from allennlp.nn.util import \
        get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.trainer import Trainer
from allennlp.training.metrics import CategoricalAccuracy

# TODO: make repo

# TODO: separate into files

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# TODO: commit to span representation??
# see https://github.com/google-research/bert#tokenization
def token_alignment(
    data_tokens: Union[List[str], List[Token]],
    model_tokenizer: Tokenizer,
    start_tokens: List[str] = None,
    end_tokens: List[str] = None,
    spans: bool = False
) -> Tuple[List[Token], List[int]]:
    """Aligns word tokens (data_tokens), with sub-word tokens.

    The Tokens in data_tokens may or may not be split into sub-words by
    model_tokenizer, e.g. if it's a tokenizer for a model like BERT.

    This method returns:
    (a) the tokens produced by model_tokenizer
        with optional start_tokens (e.g. [CLS]) and end tokens (e.g. [SEP])
    (b) a list of either
        indices: which position in the list of sub-word tokens
            corresponds to each original Token in data_tokens.
        or (if spans=True)
        spans: pairs of (start, inclusive-end) for which spans of sub-words correspond
            to the words
    """
    model_tokens = []
    data_to_model_map = []

    if start_tokens:
        model_tokens.extend([Token(t) for t in start_tokens])

    for token in data_tokens:
        # where in the model tokens are we starting the new data token
        data_to_model_map.append(len(model_tokens))
        # if data_tokens is a list of Tokens, get the text out by using the
        # .text attribute
        if hasattr(token, 'text'):
            token = token.text
        model_tokens.extend(model_tokenizer.tokenize(token))

    if spans:
        data_to_model_map.append(len(model_tokens))
        data_to_model_map = [(data_to_model_map[i], data_to_model_map[i+1]-1)
                             for i in range(len(data_to_model_map)-1)]

    if end_tokens:
        model_tokens.extend([Token(t) for t in end_tokens])

    return model_tokens, data_to_model_map


class SemTagDatasetReader(DatasetReader):
    """
    DatasetReader for Semantic Tagging dataset: https://pmb.let.rug.nl/data.php

    Each file in a directory contains one tab-separated example, e.g.:
        PRO I
        ENS like
        DEF the
        ROL actor
        NIL .
    """

    def __init__(
        self,
        model_tokenizer: Tokenizer,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        spans: bool = False
    ) -> None:
        super().__init__(lazy=False)
        self.model_tokenizer = model_tokenizer
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.token_indexers = (token_indexers or
                               # if none passed:
                               {"tokens": SingleIdTokenIndexer()})
        self.spans = spans

    def text_to_instance(
        self, strings: List[str], tags: List[str] = None, spans: bool = False
    ) -> Instance:

        data_tokens = [Token(string) for string in strings]
        data_sentence = TextField(data_tokens,
                                  {"data_tokens": SingleIdTokenIndexer()})
        fields = {"data_sentence": data_sentence}

        model_tokens, data_to_model_map = token_alignment(
            data_tokens, self.model_tokenizer,
            self.start_tokens, self.end_tokens, spans)

        model_sentence = TextField(model_tokens, self.token_indexers)
        fields["model_sentence"] = model_sentence
        # TODO(sst): is ArrayField the right choice here, or List of Index? or
        # SpanField?
        fields["data_to_model_indices"] = (
            ArrayField(np.array(data_to_model_map)) if not spans else
            ListField([SpanField(span[0], span[1], model_sentence)
                       for span in data_to_model_map])
        )

        if tags:
            # labels has same length as data_tokens
            label_field = SequenceLabelField(labels=tags,
                                             sequence_field=data_sentence)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, path: str) -> Iterator[Instance]:
        for file_name in glob.glob(path + "/*")[:20]: # TODO: take away first-20 restriction
            with open(file_name, 'r') as f:
                lines = f.readlines()
            pairs = [line.strip().split('\t') for line in lines]
            tags, sentence = zip(*pairs)
            yield self.text_to_instance(sentence, tags, self.spans)


class EncoderSpanExtractor(SpanExtractor):

    def __init__(self,
                 input_dim: int,
                 pooler: Seq2VecEncoder):
        super().__init__()
        self._input_dim = input_dim
        self._pooler = pooler

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    # TODO: WRITE .forward, using SelfAttentive as inspiration, but applying
    # pooler:
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/span_extractors/self_attentive_span_extractor.py

class Tagger(Model):

    # TODO: document!

    def __init__(self,
                 subword_embeddings: TextFieldEmbedder,
                 subword_aggregator: SpanExtractor,
                 data_vocab: Vocabulary,
                 freeze_encoder: bool = True):
        super().__init__(data_vocab)

        self.subword_embeddings = subword_embeddings
        self._freeze_encoder = freeze_encoder
        # turn off gradients if don't want to fine tune encoder
        for parameter in self.subword_embeddings.parameters():
            parameter.requires_grad = not self._freeze_encoder

        self.subword_aggregator = subword_aggregator

        self.classifier = TimeDistributed(torch.nn.Linear(
            in_features=subword_aggregator.get_output_dim(),
            out_features=data_vocab.get_vocab_size("labels")))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                data_sentence: Dict[str, torch.Tensor],
                model_sentence: Dict[str, torch.Tensor],
                # TODO: change name to spans
                data_to_model_indices: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # (batch_size, max_subword_seq_len, embedding_dim)
        subword_embeddings = self.subword_embeddings(model_sentence)

        # (batch_size, max_word_seq_len)
        word_mask = get_text_field_mask(data_sentence)

        # (batch_size, max_word_seq_len, embedding_dim)
        word_embeddings = self.subword_aggregator(
            subword_embeddings, data_to_model_indices,
            # spans correspond to words, so the valid spans have the same
            # indices as the valid words
            span_indices_mask=word_mask)

        # (batch_size, max_word_seq_len, num_labels)
        logits = self.classifier(word_embeddings)

        outputs = {'logits': logits}

        if labels is not None:
            self.accuracy(logits, labels, word_mask)
            outputs['loss'] = sequence_cross_entropy_with_logits(
                logits, labels, word_mask)

        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}


if __name__ == '__main__':

    set_seed(1234)

    # TODO: make this tokenizer initialization a method?
    uncased = True
    tokenizer = PretrainedTransformerTokenizer(
        model_name="bert-base-uncased",
        do_lowercase=uncased)
    start_tokens = tokenizer._start_tokens
    end_tokens = tokenizer._end_tokens
    tokenizer._start_tokens = []
    tokenizer._end_tokens = []

    token_indexer = PretrainedTransformerIndexer(
        model_name="bert-base-uncased",
        do_lowercase=uncased)

    reader = SemTagDatasetReader(
        tokenizer, start_tokens, end_tokens,
        {"model_tokens": token_indexer},
        spans=True)
    entire_dataset = reader.read('sem-0.1.0/data/gold')

    # NOTE: PretrainedTransformerIndexer does not implement the
    # count_vocab_items method, so this vocabulary reflects only the new
    # dataset, not the pretrained model's vocabulary
    # see: https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/pretrained_transformer_indexer.py#L47-L50
    data_vocab = Vocabulary.from_instances(entire_dataset)

    [print(entire_dataset[i]) for i in range(8)]

    # TODO: move this splitting to pre-processing on disk, not in memory!
    split_idx = int(0.8 * len(entire_dataset))
    train_dataset = entire_dataset[:split_idx]
    dev_dataset = entire_dataset[split_idx:]

    bert_token_embedder = PretrainedTransformerEmbedder("bert-base-uncased")
    bert_textfield_embedder = BasicTextFieldEmbedder(
        {"model_tokens": bert_token_embedder})

    bag_encoder = BagOfEmbeddingsEncoder(bert_token_embedder.get_output_dim())

    # represent a word by the first sub-word token
    word_first_token_extractor = EndpointSpanExtractor(
        bert_token_embedder.get_output_dim(), combination='x')

    tagger = Tagger(bert_textfield_embedder,
                    word_first_token_extractor,
                    data_vocab)

    iterator = BucketIterator(sorting_keys=[("model_sentence", "num_tokens")], batch_size=8)
    # iterator = BasicIterator(batch_size=8)
    vocab = Vocabulary()
    iterator.index_with(data_vocab)

    """
    batch = next(iter(iterator(train_dataset, shuffle=False)))
    print(batch)
    print(tagger(**batch))
    """
    trainer = Trainer(model=tagger,
                      optimizer=optim.Adam(tagger.parameters()),
                      serialization_dir='/tmp/test',
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=5,
                      num_epochs=30)
    trainer.train()

    """
    # TODO: convert the below into a proper test!
    data_idx_to_str = data_vocab.get_index_to_token_vocabulary()
    data_idx_example = batch['data_sentence']['data_tokens'].numpy()[0]
    print(data_idx_example)
    print([data_idx_to_str[int(idx)] for idx in np.nditer(data_idx_example)])
    print(f"Num_labels: {data_vocab.get_vocab_size('labels')}")
    """
