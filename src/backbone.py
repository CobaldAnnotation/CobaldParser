import torch
from torch import nn
from torch import Tensor, LongTensor, BoolTensor

from transformers import AutoTokenizer, AutoModel


class TransformerEmbedder(nn.Model):
    """
    Based on AllenNLP's PretrainedTransformerMismatchedEmbedder.
    """

    def __init__(self,
        model_name: str,
        sub_token_mode: str = "avg",
        transformer_kwargs: dict[str, any] = None
    )
        self._embedder = AutoModel.from_pretrained(model_name, **transformer_kwargs)
        self.sub_token_mode = sub_token_mode

    def forward(
        self,
        token_ids: LongTensor,
        mask: BoolTensor,
        offsets: LongTensor,
        wordpiece_mask: BoolTensor,
        token_type_ids: LongTensor = None,
    ) -> Tensor:
        model_output = self._embedder(
            input_ids=token_ids,
            attention_mask=mask,
            token_type_ids=type_ids
        )
        embeddings = model_output.last_hidden_state

        assert False, "TODO: use offsets or word_ids to combine subtokens embeddings into tokens embeddings"

        # # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # # span_mask: (batch_size, num_orig_tokens, max_span_length)
        # span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)

        # span_mask = span_mask.unsqueeze(-1)

        # # Shape: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_embeddings *= span_mask  # zero out paddings

        # # If "sub_token_mode" is set to "first", return the first sub-token embedding.
        # if self.sub_token_mode == "first":
        #     # Select first sub-token embeddings from span embeddings
        #     # Shape: (batch_size, num_orig_tokens, embedding_size)
        #     orig_embeddings = span_embeddings[:, :, 0, :]

        # # If "sub_token_mode" is set to "avg", return the average of embeddings of all sub-tokens of a word
        # elif self.sub_token_mode == "avg":
        #     # Sum over embeddings of all sub-tokens of a word
        #     # Shape: (batch_size, num_orig_tokens, embedding_size)
        #     span_embeddings_sum = span_embeddings.sum(2)

        #     # Shape (batch_size, num_orig_tokens)
        #     span_embeddings_len = span_mask.sum(2)

        #     # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
        #     # Shape: (batch_size, num_orig_tokens, embedding_size)
        #     orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        #     # All the places where the span length is zero, write in zeros.
        #     orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0


class TransformerBackbone(nn.Model):
    def __init__(self,
        model_name: str,
        sub_token_mode: str = "avg",
        tokenizer_kwargs: dict[str, any] = None,
        transformer_kwargs: dict[str, any] = None
    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.embedder = TransformerEmbedder(model_name, sub_token_mode, transformer_kwargs)

    def forward(self, sentences: list[list[Tokens]]) -> Tensor:
        # BPE subtokenization.
        subtokens = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors='pt'
        )
        # Build subtokens embeddings and aggregate them into tokens embeddings.
        tokens_embeddings = self.embedder(
            subtokens.input_ids,
            subtokens.attention_mask,
        )
        return tokens_embeddings

