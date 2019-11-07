from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


class EncoderSpanExtractor(SpanExtractor):

    def __init__(self,
                 input_dim: int,
                 pooler: Seq2VecEncoder):
        super().__init__()
        self._input_dim = input_dim
        self._pooler = pooler

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim
