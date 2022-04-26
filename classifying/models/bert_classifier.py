# modified from allennlp/models/bert_for_classification.py
from typing import Dict, Union

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bert_classifier")
class BertClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        freeze_encoder: bool = True
    ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self.embedder = embedder
        self.freeze_encoder = freeze_encoder

        for parameter in self.embedder.parameters():
            parameter.requires_grad = not self.freeze_encoder

        in_features = self.embedder.get_output_dim()
        out_features = vocab.get_vocab_size(namespace="labels")

        self._classification_layer = torch.nn.Linear(in_features, out_features)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self, tokens: Dict[str, torch.Tensor], label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        # (batch_size, max_len, embedding_dim)
        embeddings = self.embedder(tokens)

        # the first embedding is for the [CLS] token
        # NOTE: this pre-supposes BERT encodings; not the most elegant!
        # (batch_size, embedding_dim)
        cls_embedding = embeddings[:, 0, :]

        # apply classification layer
        # (batch_size, num_labels)
        logits = self._classification_layer(cls_embedding)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
