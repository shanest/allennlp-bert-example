import torch

from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed

from probing.nn.util import get_full_spans


class EncoderSpanExtractor(SpanExtractor):

    def __init__(self,
                 # input_dim: int,
                 pooler: Seq2VecEncoder):
        super().__init__()
        self._input_dim = pooler.get_output_dim()
        # we distribute the pooler across _spans_, not actual time
        self._pooler = TimeDistributed(pooler)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    # TODO: WRITE .forward, using SelfAttentive as inspiration, but applying
    # pooler:
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/span_extractors/self_attentive_span_extractor.py
    def forward(
        self,
        # (batch_size, sequence_length, embedding_dim)
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.LongTensor = None
    ) -> torch.FloatTensor:

        # (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings, span_mask = get_full_spans(
            sequence_tensor, span_indices)

        print(span_embeddings.shape)

        # (batch_size, num_spans, embedding_dim)
        span_representations = self._pooler(span_embeddings, span_mask)

        return span_representations
