from typing import Dict

import torch

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed

from allennlp.nn.util import \
        get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

class SubwordWordTagger(Model):

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
                data_to_model_spans: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # (batch_size, max_subword_seq_len, embedding_dim)
        subword_embeddings = self.subword_embeddings(model_sentence)

        # (batch_size, max_word_seq_len)
        word_mask = get_text_field_mask(data_sentence)

        # (batch_size, max_word_seq_len, embedding_dim)
        word_embeddings = self.subword_aggregator(
            subword_embeddings, data_to_model_spans,
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

