import numpy as np
import torch
import torch.optim as optim

from allennlp.data.iterators import BucketIterator

from classifying.data.dataset_readers.sst_reader \
        import SSTDatasetReader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer \
        import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer \
        import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.pretrained_transformer_embedder \
        import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders.basic_text_field_embedder \
        import BasicTextFieldEmbedder

from allennlp.data.vocabulary import Vocabulary

from classifying.models.bert_classifier import BertClassifier

from allennlp.training.trainer import Trainer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    set_seed(1234)

    model_string = "bert-base-uncased"

    tokenizer = PretrainedTransformerTokenizer(
        model_string, do_lowercase=True)
    token_indexer = PretrainedTransformerIndexer(
        model_string, do_lowercase=True)

    reader = SSTDatasetReader(tokenizer, {"tokens": token_indexer})

    train_dataset = reader.read('sst/trees/train.txt')
    val_dataset = reader.read('sst/trees/dev.txt')

    print(train_dataset[0])

    vocab = Vocabulary.from_instances(train_dataset + val_dataset)

    bert_token_embedder = PretrainedTransformerEmbedder(model_string)
    bert_textfield_embedder = BasicTextFieldEmbedder(
        {"tokens": bert_token_embedder})

    model = BertClassifier(
        vocab, bert_textfield_embedder, freeze_encoder=False)

    iterator = BucketIterator(
        sorting_keys=[("tokens", "num_tokens")],
        batch_size=32)
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optim.Adam(model.parameters()),
                      serialization_dir='/tmp/test',
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=5,
                      num_epochs=30)
    trainer.train()
