from itertools import chain

from tagging.data.dataset_readers.semantic_tagging import SemTagDatasetReader
from tagging.models.tagger import SubwordWordTagger
from tagging.modules.span_extractors.encoder_span_extractor import EncoderSpanExtractor

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.data_loaders import MultiProcessDataLoader

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import (
    PretrainedTransformerTokenizer,
)
from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import (
    PretrainedTransformerMismatchedIndexer,
)

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import (
    PretrainedTransformerMismatchedEmbedder,
)

from allennlp.modules.span_extractors.endpoint_span_extractor import (
    EndpointSpanExtractor,
)

from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder

from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    set_seed(1234)

    tokenizer = PretrainedTransformerTokenizer(model_name="bert-base-uncased")

    token_indexer = PretrainedTransformerMismatchedIndexer(
        model_name="bert-base-uncased"
    )

    reader = SemTagDatasetReader(tokenizer, {"tokens": token_indexer})
    train_path = "sem-0.1.0/data/gold/train"
    dev_path = "sem-0.1.0/data/gold/val"

    train_dataset = reader.read(train_path)
    val_dataset = reader.read(dev_path)

    # NOTE: PretrainedTransformerIndexer does not implement the
    # count_vocab_items method, so this vocabulary reflects only the new
    # dataset, not the pretrained model's vocabulary
    # see: https://github.com/allenai/allennlp/blob/master/allennlp/data/
    # token_indexers/pretrained_transformer_indexer.py#L47-L50
    data_vocab = Vocabulary.from_instances(chain(train_dataset, val_dataset))

    bert_token_embedder = PretrainedTransformerMismatchedEmbedder("bert-base-uncased")
    bert_textfield_embedder = BasicTextFieldEmbedder({"tokens": bert_token_embedder})

    tagger = SubwordWordTagger(
        bert_textfield_embedder,
        data_vocab,
    )

    data_loader = MultiProcessDataLoader(reader, train_path, batch_size=32)
    data_loader.index_with(data_vocab)

    trainer = GradientDescentTrainer(
        model=tagger,
        optimizer=optim.Adam(tagger.parameters()),
        serialization_dir="/tmp/test",
        data_loader=data_loader,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        patience=5,
        num_epochs=30,
    )
    trainer.train()
