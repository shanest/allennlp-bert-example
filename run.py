from probing.data.dataset_readers.semantic_tagging import SemTagDatasetReader
from probing.models.tagger import SubwordWordTagger

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.iterators import BucketIterator

from allennlp.data.tokenizers.pretrained_transformer_tokenizer \
        import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer \
        import PretrainedTransformerIndexer

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders \
        import BasicTextFieldEmbedder

from allennlp.modules.token_embedders.pretrained_transformer_embedder \
        import PretrainedTransformerEmbedder

from allennlp.modules.span_extractors.endpoint_span_extractor \
        import EndpointSpanExtractor

from allennlp.training.trainer import Trainer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


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
        model_name="bert-base-cased",
        do_lowercase=uncased)

    reader = SemTagDatasetReader(
        tokenizer, {"model_tokens": token_indexer},
        start_tokens, end_tokens)

    train_dataset = reader.read('sem-0.1.0/data/gold/train')
    val_dataset = reader.read('sem-0.1.0/data/gold/val')

    # NOTE: PretrainedTransformerIndexer does not implement the
    # count_vocab_items method, so this vocabulary reflects only the new
    # dataset, not the pretrained model's vocabulary
    # see: https://github.com/allenai/allennlp/blob/master/allennlp/data/
    # token_indexers/pretrained_transformer_indexer.py#L47-L50
    data_vocab = Vocabulary.from_instances(train_dataset + val_dataset)

    bert_token_embedder = PretrainedTransformerEmbedder("bert-base-uncased")
    bert_textfield_embedder = BasicTextFieldEmbedder(
        {"model_tokens": bert_token_embedder})

    # represent a word by the first sub-word token
    word_last_token_extractor = EndpointSpanExtractor(
        bert_token_embedder.get_output_dim(), combination='y')

    tagger = SubwordWordTagger(bert_textfield_embedder,
                               word_last_token_extractor,
                               data_vocab)

    iterator = BucketIterator(
        sorting_keys=[("model_sentence", "num_tokens")],
        batch_size=32)
    iterator.index_with(data_vocab)

    trainer = Trainer(model=tagger,
                      optimizer=optim.Adam(tagger.parameters()),
                      serialization_dir='/tmp/test',
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=5,
                      num_epochs=30)
    trainer.train()
