from typing import List, Dict, Tuple, Union

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers import Tokenizer

# see https://github.com/google-research/bert#tokenization
def token_alignment(
    data_tokens: Union[List[str], List[Token]],
    model_tokenizer: Tokenizer,
    start_tokens: List[str] = None,
    end_tokens: List[str] = None,
) -> Tuple[List[Token], List[int]]:
    """Aligns word tokens (data_tokens), with sub-word tokens.

    The Tokens in data_tokens may or may not be split into sub-words by
    model_tokenizer, e.g. if it's a tokenizer for a model like BERT.

    This method returns:
    (a) the tokens produced by model_tokenizer
        with optional start_tokens (e.g. [CLS]) and end tokens (e.g. [SEP])
    (b) a list of spans: pairs of (start, inclusive-end) for which spans of sub-words
        correspond to the words
    """
    model_tokens = []
    data_to_model_map = []

    if start_tokens:
        model_tokens.extend([Token(t) for t in start_tokens])

    for token in data_tokens:
        # where in the model tokens are we starting the new data token
        data_to_model_map.append(len(model_tokens))
        # if data_tokens is a list of Tokens, get the text out by using the
        # .text attribute
        if hasattr(token, 'text'):
            token = token.text
        model_tokens.extend(model_tokenizer.tokenize(token))

    data_to_model_map.append(len(model_tokens))
    data_to_model_map = [(data_to_model_map[i], data_to_model_map[i+1]-1)
                         for i in range(len(data_to_model_map)-1)]

    if end_tokens:
        model_tokens.extend([Token(t) for t in end_tokens])

    return model_tokens, data_to_model_map
