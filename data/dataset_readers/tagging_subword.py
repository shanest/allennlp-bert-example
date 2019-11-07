from typing import List, Dict, Iterator

from ..util import token_alignment

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields \
        import TextField, SequenceLabelField, ListField, SpanField
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

class TaggingSubwordReader(DatasetReader):
    """
    Reads sequence-tagged data, while maintaining a mapping from the data
    sequences to alternative tokenization of the sequences.

    Useful when you want to process the data with a model that uses subword
    tokenization as in BERT.
    """
    def __init__(
        self,
        model_tokenizer: Tokenizer,
        model_token_indexers: Dict[str, TokenIndexer],
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        data_token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy=False)
        self.model_tokenizer = model_tokenizer
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.model_token_indexers = model_token_indexers
        self.data_token_indexers = (data_token_indexers or
                                    # if none passed:
                                    {"data_tokens": SingleIdTokenIndexer()})

    def text_to_instance(
        self, strings: List[str], tags: List[str] = None
    ) -> Instance:

        data_tokens = [Token(string) for string in strings]
        data_sentence = TextField(data_tokens, self.data_token_indexers)
        fields = {"data_sentence": data_sentence}

        model_tokens, data_to_model_map = token_alignment(
            data_tokens, self.model_tokenizer,
            self.start_tokens, self.end_tokens)

        model_sentence = TextField(model_tokens, self.model_token_indexers)
        fields["model_sentence"] = model_sentence
        fields["data_to_model_spans"] = (ListField(
            [SpanField(span[0], span[1], model_sentence)
             for span in data_to_model_map]))

        if tags:
            # labels has same length as data_tokens
            label_field = SequenceLabelField(labels=tags,
                                             sequence_field=data_sentence)
            fields["labels"] = label_field

        return Instance(fields)
