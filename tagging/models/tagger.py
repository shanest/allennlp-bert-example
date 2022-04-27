from typing import Dict

import torch

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy


@Model.register("subword_word_tagger")
class SubwordWordTagger(Model):

    # TODO: document!

    def __init__(
        self,
        embedder: TextFieldEmbedder,
        vocab: Vocabulary = None,
        freeze_encoder: bool = True,
    ):
        super().__init__(vocab)

        self._embedder = embedder
        self._freeze_encoder = freeze_encoder
        # turn off gradients if don't want to fine tune encoder
        for parameter in self._embedder.parameters():
            parameter.requires_grad = not self._freeze_encoder

        self.classifier = TimeDistributed(
            torch.nn.Linear(
                in_features=embedder.get_output_dim(),
                out_features=vocab.get_vocab_size("labels"),
            )
        )

        self.accuracy = CategoricalAccuracy()

    def forward(
        self, sentence: Dict[str, torch.Tensor], labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        # (batch_size, max_seq_len, embedding_dim)
        embeddings = self._embedder(sentence)

        # get mask to keep track of which tokens are padding
        # prevent them from contributing to loss
        # (batch_size, max_word_seq_len)
        word_mask = get_text_field_mask(sentence)

        # (batch_size, max_word_seq_len, num_labels)
        logits = self.classifier(embeddings)

        outputs = {"logits": logits}

        if labels is not None:
            self.accuracy(logits, labels, word_mask)
            outputs["loss"] = sequence_cross_entropy_with_logits(
                logits, labels, word_mask
            )

        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
