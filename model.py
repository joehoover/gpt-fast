# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    """
    Finds the smallest multiple of k that is  >= to n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    # Maximum number of tokens in a sequence

    vocab_size: int = 32000
    # Number of tokens in the model vocabulary

    n_layer: int = 32
    # Number of layers

    n_head: int = 32
    # Number of attention heads

    dim: int = 4096
    # Size of each token embedding

    intermediate_size: int = None
    # Size of the intermediate layer in the feedforward block
    # If not specified, defaults to:
    # int(2*4*dim/3) rounded to the nearest multiple of 256

    n_local_heads: int = -1
    # Number of KV heads. If not specified, defaults to n_head,
    # which implements standard self-attention. If specified,
    # implements GQA if `n_local_heads` is > 1, else implements MQA.

    head_dim: int = 64
    # Size of each head

    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        # If n_local_heads is not specified, set number of KV heads to number of Q heads.
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head

        # If intermediate_size is not specified, set it to 2/3 of 4*dim, rounded to the nearest multiple of 256.
        # e.g., for llama-2-7b, intermediate_size = find_multiple(2/3 * 4 * 4096) = 11008
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        # Determine size of each head by dividing dim by number of heads.
        # e.g., for llama-2-7b, dim = 4096, n_head = 32, head_dim = 128
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        """
        Create an instance of the class based on the given name.
        """
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        """
        Initializes the GPT-Fast model.

        Args:
            max_batch_size (int): The maximum batch size.
            max_seq_length (int): The maximum sequence length.
            n_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            dtype (torch.dtype, optional): The data type of the cache tensors. Defaults to torch.bfloat16.
        """
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        # Define cache as 4D tensor of shape (max_batch_size, n_heads, max_seq_length, head_dim)

        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
        # Register Key and Value caches as buffers. Buffers are tensors that are registered in a module,
        # but are not treated as model parameters. They are typically used for tensors that
        # need to be saved, but not trained, such as running statistics. In this case,
        # we store and increment key and value tensors in the cache buffers.

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        # Update the KV caches at input_pos with the new key and value vectors.

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        # Initialize token embeddings as a matrix of shape (vocab_size, dim)
        # where each row is a token embedding of dimension dim.

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        # Initialize layers as a list of n_layer TransformerBlocks.

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        # Initialize final normalization layer, which is used
        # to normalize the output of the last TransformerBlock before
        # passing it to the output layer.

        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Initialize output layer as a linear layer of shape (dim, vocab_size)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        # `self.max_seq_length` and `self.max_batch_size` are initialized to -1, it seems that this will always be false in this implementation.

        head_dim = self.config.dim // self.config.n_head
        # Determine size of each head by dividing dim by number of heads.
        # This should be correctly specified in self.config, so this is probably redundant.

        max_seq_length = find_multiple(max_seq_length, 8)
        # Find the smallest multiple of 8 that is >= to max_seq_length and this as the new max_seq_length.
        # This confers a variety of efficiency benefits, such as allowing for more efficient memory access.

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )
        # Initialize the KV cache for each TransformerBlock in the Transformer.
        # As mentioned above, the KV cache is a 4D tensor of shape (max_batch_size, n_heads, max_seq_length, head_dim).

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        # Initialize a causal mask of shape (max_seq_length, max_seq_length) with all elements below the diagonal set to True.

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        # Get the causal mask for the current input_pos and add two dimensions to the front.
        # Note: indexing with `None` can be used to add dimensions to a tensor.

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        # Get the token embeddings for the current input sequence.

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
            # Apply TransformerBlock i to the output of the previous layer.
        x = self.norm(x)
        # Apply the final normalization layer to the output of the last TransformerBlock.
        logits = self.output(x)
        # Apply the output layer to the normed output of the last TransformerBlock.
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        # Apply attention to the output of the attention normalization layer and add it to the residual stream.
        out = h + self.feed_forward(self.ffn_norm(h))
        # Apply feed forward to the output of the feed forward normalization layer.
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        # key, query, value projections for all heads, but in a batch
        # If we're using GQA, n_local_heads is < n_head and we wind up with fewer key and value heads than query heads.

        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        # output projection

        self.kv_cache = None
        # KV cache is initialized to None and is set to a KVCache object in Transformer.setup_cache.

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)
        # Register a hook that is called before state_dict is loaded.

    def load_hook(self, state_dict, prefix, *args):
        """
        Make sure that the weights for the key, query, and value projections are concatenated
        """
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
