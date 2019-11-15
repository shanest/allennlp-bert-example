import torch

from allennlp.nn import util


def get_full_spans(
    sequence_tensor: torch.Tensor,
    span_indices: torch.LongTensor
) -> torch.Tensor:
    """
    Extracts spans from a sequence tensor.
    Extracted from the forward pass of `self_attentive_span_extractor`:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/span_extractors/self_attentive_span_extractor.py#L46

    Parameters
    ----------
    sequence_tensor : ``torch.Tensor``, required.
        Shape: (batch_size, sequence_length, embedding_dimension)
        This is an embedding of some text, from which you want to extract
        spans.
    span_indices : ``torch.LongTensor``, required.
        Shape: (batch_size, num_spans, 2)
        Start/end (inclusive) indices for spans in the sequence.

    Returns
    -------
    span_embeddings: ``torch.Tensor``
        Tensor of shape (batch_size, num_spans, max_span_width, embedding_dim).
        These are the embeddings of every token in every span. This tensor is
        useful if you want to try different ways of representing spans.
    span_mask: ``torch.Tensor``
        Shape (batch_size, num_spans, max_span_width)
        Binary (1/0) vector, marking whether an embedding belongs to a span.
    """
    # both (batch_size, num_spans, 1)
    span_starts, span_ends = span_indices.split(1, dim=-1)

    # (batch_size, num_spans, 1)
    # NB: widths are 1 too `low' because of inclusive endpoints
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # (1, 1, max_span_width)
    max_span_range_indices = util.get_range_vector(
        max_batch_span_width, util.get_device_of(sequence_tensor)
    ).view(1, 1, -1)

    # create the mask via broadcast comparison
    # for each span, create a vector of length max_span_width
    # with 1s for valid span tokens, 0s for non-span tokens
    #
    # <= for the mask because of the inclusive endpoints convention in
    # span_indices
    #
    # (batch_size, num_spans, max_span_width)
    span_mask = (max_span_range_indices <= span_widths).float()

    # NOTE: here I depart from the base implementation.  Is there a bug in
    # theirs?
    # (batch_size, num_spans, max_span_width)
    raw_span_indices = (span_starts + max_span_range_indices)
    # the actual indices in the sequence for a given span!
    # (batch_size, num_spans, max_span_width)
    final_span_indices = (raw_span_indices *
                          (raw_span_indices < sequence_tensor.size(1)).float())
    # When AllenNLP pads a sequencefield of spanfields, it will generate
    # padding spans with indices [-1, -1], which can lead to errors here
    # so make sure there are no negative values
    final_span_indices = torch.nn.functional.relu(final_span_indices).long()

    # (batch_size, num_spans, max_width, embedding_dim)
    span_embeddings = util.batched_index_select(
        sequence_tensor, final_span_indices
    )

    return span_embeddings, span_mask


if __name__ == '__main__':

    # TODO: make this an actual test!
    sequences = torch.Tensor(
        [
            [[1], [2], [3], [4]],
            [[5], [6], [7], [8]],
            [[9], [10], [11], [12]]
        ]
    )
    span_indices = torch.Tensor([
        [[1, 1], [2, 3]],
        [[0, 2], [2, 3]],
        [[2, 2], [3, 3]]
    ])
    spans, mask = get_full_spans(sequences, span_indices)
