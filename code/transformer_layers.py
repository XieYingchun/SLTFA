# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


class AMultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(
        self,
        query_type: str,
        query_nb: int,
        num_heads: int,
        size: int,
        dropout: float = 0.1,
    ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(AMultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.query_type = query_type
        self.query_nb = query_nb
        if query_type == "mean":
            self.pooling = nn.AvgPool1d(
                kernel_size=query_nb, stride=1, padding=query_nb // 2
            )
        elif query_type == "attention":
            self.att_layer = nn.Linear(num_heads * head_size, 1)

        # else:
        #     self.pooling = None
        # self.pooling = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # self.pooling = nn.Conv1d(size, size, 3, 1, 1)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        # q aggregation
        if self.query_type == "mean":
            q = self.pooling(q.transpose(1, 2).contiguous()).transpose(1, 2)
        elif self.query_type == "attention":
            q_len = q.shape[1]
            q_att_score = self.att_layer(q).reshape(-1, 1, q_len).repeat(1, q_len, 1)
            q_att_mask = q.new_ones((q_len, q_len), dtype=torch.bool)
            for i in range(q_len):
                start = max(0, i - self.query_nb // 2)
                end = min(q_len, i + self.query_nb // 2 + 1)
                q_att_mask[i, start:end] = False
            q_att_score = q_att_score.masked_fill(q_att_mask, float("-inf"))
            q_att_score = self.softmax(q_att_score)
            # batch x query_len x query_len , batch x query_len x feautre_len --> batch x query_len x feature_len
            q = torch.matmul(q_att_score, q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output

########################################################
########SeparableMultiHeadAttention

class DeformableMultiHeadedAttention(nn.Module):
    """Depthwise separable self-attention, including DSA and PSA."""
    def __init__(
            self, 
            query_type: str, 
            query_nb: int,
            num_heads: int, 
            size: int, 
            dropout: float = 0.1, 
            num_keys : int = 7, 
            ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        :param window_size: The number of window size
        :param query_type: the type of q aggregate
        :param query_nb: the number of q aggregate by attention
        """
        super(DeformableMultiHeadedAttention, self).__init__()

        assert size % num_heads == 0, f"size {size} should be divided by num_heads {num_heads}."

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.scale = size ** -0.5

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.DSA_attn_drop = nn.Dropout(0.3)

        self.win_tokens_norm = nn.LayerNorm(size)
        self.win_tokens_act = nn.GELU()

        self.PSA_k_layer = nn.Linear(size, num_heads * head_size)
        self.PSA_q_layer = nn.Linear(size, num_heads * head_size)
        self.PSA_attn_drop = nn.Dropout(0.3)


        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.query_type = query_type
        self.query_nb = query_nb

        if query_type == "mean":
            self.pooling = nn.AvgPool1d(
                kernel_size=query_nb, stride=1, padding=query_nb // 2
            )
        elif query_type == "attention":
            self.att_layer = nn.Linear(num_heads * head_size, 1)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, query_len: int, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        query_len = query_len
        #orignal_x_norm = q
        
        # calculate the number of windows
        win_size = self.num_keys
        
        ############################ValueError: only one element tensors can be converted to Python scalars
        win_num = math.ceil(query_len / self.num_keys)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        
        

        # q aggregation
        if self.query_type == "mean":
            q = self.pooling(q.transpose(1, 2).contiguous()).transpose(1, 2)
        elif self.query_type == "attention":
            q_len = q.shape[1]
            q_att_score = self.att_layer(q).reshape(-1, 1, q_len).repeat(1, q_len, 1)
            q_att_mask = q.new_ones((q_len, q_len), dtype=torch.bool)
            for i in range(q_len):
                start = max(0, i - self.query_nb // 2)
                end = min(q_len, i + self.query_nb // 2 + 1)
                q_att_mask[i, start:end] = False
            q_att_score = q_att_score.masked_fill(q_att_mask, float("-inf"))
            q_att_score = self.softmax(q_att_score)
            # batch x query_len x query_len , batch x query_len x feautre_len --> batch x query_len x feature_len
            q = torch.matmul(q_att_score, q)
            

        # Depthwise Self-Attention (DSA)
        # reshape q, k, v for our computation to [batch_size, win_num, win_size+1, num_heads, head_size]  
        DSA_k = k.reshape(batch_size, win_num, -1, self.num_heads, self.head_size).transpose(2, 3) # B, win_num, n_head, win_size+1, head_dim
        DSA_v = v.reshape(batch_size, win_num, -1, self.num_heads, self.head_size).transpose(2, 3)
        DSA_q = q.reshape(batch_size, win_num, -1, self.num_heads, self.head_size).transpose(2, 3)

        DSA_attn = (DSA_q @ DSA_k.transpose(-2, -1)) * self.scale # Q@K = B, win_num, n_head, win_size+1, win_size+1
        DSA_attn = self.softmax(DSA_attn)
        DSA_attn = self.DSA_attn_drop(DSA_attn)
        attn_out = (DSA_attn @ DSA_v).transpose(2, 3).reshape(batch_size, win_num, -1, self.model_size) # attn @ V --> B, win_num, n_head, win_size+1, model_size//n_head -> (t(2,3)) -> B, win_num, win_size+1, model_size
        #attn_out = attn_out + orignal_x_norm    # short cut

        # Pointwise Self-Attention (PSA)
        # slice window tokens (win_tokens) and feature maps (attn_x)
        attn_win_tokens = attn_out[:, :, 0, :] # B, win_num, model_size
        attn_x = attn_out[:, :, 1:, :] # B, win_num, win_size, model_size

        # LN & Act
        attn_win_tokens = self.win_tokens_norm(attn_win_tokens)
        attn_win_tokens = self.win_tokens_act(attn_win_tokens)

        PSA_q = self.PSA_q_layer(attn_win_tokens)
        PSA_k = self.PSA_k_layer(attn_win_tokens)
        PSA_q = PSA_q.reshape(batch_size, win_num, self.num_heads, -1).transpose(1, 2)
        PSA_k = PSA_k.reshape(batch_size, win_num, self.num_heads, -1).transpose(1, 2)

        # resahpe attn_x to multi_head
        PSA_v = attn_x.reshape(batch_size, win_num, win_size, self.num_heads, -1).permute(0, 3, 1, 2, 4) # B, win_num, win_size, n_head, head_dim -> B, n_head, win_num, win_size, head_dim
        PSA_v = PSA_v.reshape(batch_size, self.num_heads, win_num, -1) # (B, n_head, win_num, win_size*head_dim)

        PSA_attn = (PSA_q @ PSA_k.transpose(-2, -1)) * self.scale # Q@K = B, n_head, win_num, win_num
        PSA_attn = self.softmax(PSA_attn)
        PSA_attn = self.PSA_attn_drop(PSA_attn)
        attn_out = (PSA_attn @ PSA_v)  # (B, n_head, win_num, win_num) @ (B, n_head, win_num, win_size*head_dim) = (B, n_head, win_num, win_size*head_dim)

        # delete padding and reshape to B, M, D
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, win_num, self.num_heads, win_size, -1) # B, win_num, n_head, win_size*head_dim -> B, win_num, n_head, win_size, head_dim
        attn_out = attn_out.transpose(2, 3).reshape(batch_size, win_num, win_size, self.model_size) # B, win_num, win_size, model_size
        
        attn_out = attn_out + attn_x    # short cut
        attn_out = attn_out.reshape(batch_size, -1, self.model_size)  ## B win_num*win_size D
        attn_out = attn_out[:, :query_len, :]

        x = self.output_layer(attn_out)
        x = self.dropout(x)

        return x
################################################################

# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o



############################################
##########SepViTEncoderLayer
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
            self,   
            query_type: str, 
            query_nb: int,
            size: int = 0,
            ff_size: int = 0,
            num_heads: int = 0, 
            dropout: float = 0.1, 
            num_keys : int = 7,    
            ):
        super(DeformableTransformerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.act_layer = nn.GELU()

        self.src_src_att = DeformableMultiHeadedAttention(query_type, query_nb, num_heads, size, dropout=dropout, num_keys=num_keys)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.num_keys =num_keys
    

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_size = x.size(0)
        query_len = x.size(1)
        win_num = math.ceil(query_len / self.num_keys)
        pad_len = (self.num_keys- query_len % self.num_keys)% self.num_keys
        if pad_len > 0:
            padding = torch.zeros((batch_size, pad_len, self.size), device=x.device)  # Create the padding tensor
            attn_x = torch.cat([x, padding], dim=1)  # Concatenate the padding tensor to the sequence
        else:
            attn_x = x
        assert attn_x.shape[1] == win_num * self.num_keys, 'The wrong padding.'


        attn_x = attn_x.reshape(batch_size, win_num, self.num_keys, self.size)
        win_tokens = torch.zeros((batch_size, win_num, 1, self.size), device=x.device)
        attn_x = torch.cat((win_tokens, attn_x), dim=2) # B, win_num, win_size+1,model_size

        x_norm = self.layer_norm(attn_x).reshape(batch_size, -1, self.size)
        #print('x_norm shape:',x_norm.shape)
        h = self.src_src_att(x_norm, x_norm, x_norm, query_len, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o

#########################################################

class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o


class CrossTransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(CrossTransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.corss_src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.cross_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        memory2: Tensor = None,
        src_mask: Tensor = None,
        src_mask2: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        tgt = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        tgt = self.dropout(tgt) + x

        # source-target attention
        tgt_norm = self.dec_layer_norm(tgt)
        tgt_2 = self.src_trg_att(memory, memory, tgt_norm, mask=src_mask)
        tgt = self.dropout(tgt_2) + tgt
        tgt_norm = self.cross_layer_norm(tgt)
        tgt_2 = self.corss_src_trg_att(memory2, memory2, tgt_norm, mask=src_mask2)
        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(tgt_2) + tgt)

        return o
