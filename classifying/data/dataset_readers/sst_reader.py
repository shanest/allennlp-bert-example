# modified from
# allennlp/data/dataset_readers/stanford_sentiment_treebank_reader
# to include an alternative tokenization (e.g. from pretrained model)
from typing import Dict, List
import logging

from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


@DatasetReader.register("sst_reader")
class SSTDatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.

    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. `"5-class"` uses these labels as is. `"3-class"` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). `"2-class"` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).

    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.

    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`

    # Parameters
    tokenizer: `Tokenizer`, optional (default = None)
        Tokenizer to use if different from the data tokens (e.g. pretrained
        transformer tokenizer)
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    granularity : `str`, optional (default = `"5-class"`)
        One of `"5-class"`, `"3-class"`, or `"2-class"`, indicating the number
        of sentiment labels to use.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        granularity: str = "5-class",
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        allowed_granularities = ["5-class", "3-class", "2-class"]
        if granularity not in allowed_granularities:
            raise ConfigurationError(
                "granularity is {}, but expected one of: {}".format(
                    granularity, allowed_granularities
                )
            )
        self._granularity = granularity

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                parsed_line = Tree.fromstring(line)
                instance = self.text_to_instance(parsed_line.leaves(), parsed_line.label())
                if instance is not None:
                    yield instance

    def text_to_instance(
        self, tokens: List[str], sentiment: str = None
    ) -> Instance:  # type: ignore
        """
        We either take `pre-tokenized` input here, or further split the
        pre-tokenized input based on whether or not a tokenizer was .

        # Parameters

        tokens : `List[str]`, required.
            The tokens in a given sentence.
        sentiment : `str`, optional, (default = None).
            The sentiment for this sentence.

        # Returns

        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence or phrase.
            label : `LabelField`
                The sentiment label of the sentence or phrase.
        """

        if self._tokenizer:
            new_tokens = self._tokenizer.tokenize(' '.join(tokens))
        else:
            new_tokens = [Token(token) for token in tokens]
        text_field = TextField(new_tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if sentiment is not None:
            # 0 and 1 are negative sentiment, 2 is neutral, and 3 and 4 are positive sentiment
            # In 5-class, we use labels as is.
            # 3-class reduces the granularity, and only asks the model to predict
            # negative, neutral, or positive.
            # 2-class further reduces the granularity by only asking the model to
            # predict whether an instance is negative or positive.
            if self._granularity == "3-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    sentiment = "1"
                else:
                    sentiment = "2"
            elif self._granularity == "2-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    return None
                else:
                    sentiment = "1"
            fields["label"] = LabelField(sentiment)
        return Instance(fields)
