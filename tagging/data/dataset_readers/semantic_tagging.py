from typing import Iterator

import glob

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("semantic_tag")
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
        tokenizer: Tokenizer = None,
        token_indexers: dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str) -> Iterator[Instance]:
        for file_name in glob.glob(path + "/*"):
            with open(file_name, "r") as f:
                lines = f.readlines()
            pairs = [line.strip().split("\t") for line in lines]
            tags, sentence = zip(*pairs)
            yield self.text_to_instance(sentence, tags)

    def text_to_instance(self, strings: list[str], tags: list[str] = None) -> Instance:
        tokens = [Token(string) for string in strings]
        sentence = TextField(tokens, self._token_indexers)
        fields = {"sentence": sentence}
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence)
            fields["labels"] = label_field

        return Instance(fields)
