from typing import Iterator

import glob

from .tagging_subword import TaggingSubwordReader

from allennlp.data import Instance


class SemTagDatasetReader(TaggingSubwordReader):
    """
    DatasetReader for Semantic Tagging dataset: https://pmb.let.rug.nl/data.php

    Each file in a directory contains one tab-separated example, e.g.:
        PRO I
        ENS like
        DEF the
        ROL actor
        NIL .
    """

    def _read(self, path: str) -> Iterator[Instance]:
        for file_name in glob.glob(path + "/*"):
            with open(file_name, 'r') as f:
                lines = f.readlines()
            pairs = [line.strip().split('\t') for line in lines]
            tags, sentence = zip(*pairs)
            yield self.text_to_instance(sentence, tags)
