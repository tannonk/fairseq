import torch.nn as nn
import torch
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models import FairseqDecoder
from fairseq.models import FairseqEncoderDecoderModel, register_model
from fairseq.models import register_model_architecture
from fairseq.modules import FairseqDropout
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor


'''The one I worked with'''

DEFAULT_MAX_SOURCE_POSITIONS = 500
DEFAULT_MAX_TARGET_POSITIONS = 500


@register_model('simple_lstm')
class SimpleLSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder2 = encoder
        assert isinstance(self.encoder2, FairseqEncoder)

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        # parser.add_argument('--knowledge', default=None,
                            # help='include ground knowledge')
        # parser.add_argument('-t2', '--target2', default=None, metavar='SRC',
        #                     help='responses aligned to ground knowledge')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        # parser.add_argument(
        #     '--encoder-dropout', type=float, default=0.1,
        #     help='encoder dropout probability',
        # )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        # parser.add_argument(
        #     '--decoder-dropout', type=float, default=0.1,
        #     help='decoder dropout probability',
        # )

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # print(args)

        max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )
        max_target_positions = getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        max_know_positions = getattr(
            args, "max_know_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )

        encoder = SimpleLSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            # hidden_dim=args.encoder_hidden_dim,
            hidden_size=args.encoder_hidden_size,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            max_source_positions=max_source_positions,
        )

        # def forward(self, src_tokens, src_lengths, enforce_sorted=True):
        encoder2 = SimpleLSTMEncoder(
            args=args,
            # return self.know_dict in @property-source2_dictionary
            dictionary=task.source2_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            max_know_positions=max_know_positions,
        )

        decoder = SimpleLSTMDecoder(
            dictionary=task.target_dictionary,
            # encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            encoder_output_units=encoder.output_units,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            max_target_positions=max_target_positions,
            attention=utils.eval_bool(args.decoder_attention),
        )
        model = SimpleLSTMModel(encoder2, decoder)
        # model = SimpleLSTMModel(encoder2, decoder)

        # Print the model architecture.
        # print(type(model))
        # print(model)

        return model

    def forward(
        self,
        src_tokens,
        src_lengths,
        src2_tokens,
        src2_lengths,
        prev_output_tokens,
        enforce_sorted=True,
    ):
        # encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder2_out = self.encoder2(src2_tokens, src_lengths=src2_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder2_out)

        return decoder_out
    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
    #     return decoder_out


class SimpleLSTMEncoder(FairseqEncoder):

    def __init__(
        self,
        args,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        dropout_in=0.1,
        dropout_out=0.1,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        max_know_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        padding_idx=None,
        num_layers=1
    ):
        super().__init__(dictionary)
        self.args = args

        self.max_source_positions = max_source_positions
        self.num_layers = num_layers
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()

        self.output_units = hidden_size
        self.hidden_size = hidden_size

        # print("ENCODER ARGGS")
        # print(self.max_source_positions)

        # for arg in vars(self.args):
        #     print(arg, ':::', getattr(self.args, arg))

        # Our encoder will embed the inputs before feeding them to the LSTM.
        # nn.Embedding is a simple lookup table that stores embeddings of a fixed dictionary and size.
        # Input: (*), LongTensor of arbitrary shape containing the indices to extract
        # Output: (*, H), where * is the input shape and H=embedding_dim
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),  # size of source vocab
            embedding_dim=embed_dim,  # embedding size
            padding_idx=self.padding_idx,
        )
        # self.dropout = nn.Dropout(p=dropout)
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.,
        )

    # def forward_torchscript(self, net_input: Dict[str, Tensor]):
    #     """A TorchScript-compatible version of forward.
    #
    #     Encoders which use additional arguments may want to override
    #     this method for TorchScript compatibility.
    #     """
    #     if torch.jit.is_scripting():
    #         return self.forward(
    #             src_tokens=net_input["src_tokens"],
    #             src_lengths=net_input["src_lengths"],
    #             src2_tokens=net_input["src2_tokens"]
    #         )
    #     else:
    #         return self.forward_non_torchscript(net_input)

    def forward(self, src_tokens, src_lengths, enforce_sorted=True):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each mini-batch.
        # src_tokens (LongTensor): tokens in the source language of shape `(batch, src_len)`
        # src_lengths (LongTensor): lengths of each source sentence of shape `(batch)`

        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True
            )

        bsz, seqlen = src_tokens.size()

        # Embed the source.
        # self.embed_tokens has shape Vocab size X embed_dim
        # *src_tokens* has shape `(batch, src_len)`
        # produces batches where each sample has a shape (src_len, embed_dim)
        # x should have size (batch_size, src_len, embed_dim)
        x = self.embed_tokens(src_tokens)

        # Apply dropout.
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
        # T is the length of the longest sequence (equal to lengths[0])
        # B is the batch size

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        # batch_first (bool, optional) – if True, the input is expected in B x T x * format.
        # * is any number of dimensions (including 0).
        x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data, batch_first=True, enforce_sorted=enforce_sorted)

        # state_size = self.num_layers, bsz, self.hidden_size
        # h0 = x.new_zeros(*state_size)
        # c0 = x.new_zeros(*state_size)

        # Get the output from the LSTM.
        _outputs, (final_hidden, _final_cell) = self.lstm(x)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            _outputs, padding_value=self.padding_idx * 1.0
        )
        # x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        # return {
        #     # this will have shape `(bsz, hidden_dim)`
        #     'final_hidden': final_hidden.squeeze(0),
        # }

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # The input is put through an encoder model which gives us the encoder output of shape
        # (batch_size, max_length, hidden_size) and the encoder hidden state of shape (batch_size, hidden_size).

        # This last output is sometimes called the context vector as it encodes context from the entire sequence.
        # This context vector is used as the initial hidden state of the decoder.

        return tuple(
            (
                x,  # (seq_len x batch x hidden)
                final_hidden,  # num_layers x batch x num_directions*hidden
                _final_cell,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
            )
        )

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        # final_hidden = encoder_out['final_hidden']
        # return {
        #     'final_hidden': final_hidden.index_select(0, new_order),
        # }
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
            )
        )

    def max_positions(self):
        return self.max_source_positions


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        # nn.Linear applies a linear transformation to the incoming data: y = xA^T + b

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)

        # I assume this is the current target hidden state
        self.output_proj = Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim

        # are these source states?
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # #1. compute attention weights ###################################
        # are these attention weights?
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        # torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
        # softmax(Xi) = exp(Xi)/SUMj(exp(Xj))
        # It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.

        # The current target hidden state is compared with all source states to derive attention weights.
        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # ########################## ######################################
        # #2.compute a context vector #####################################

        # Based on the attention weights we compute a context vector
        # as the weighted average of the source states.
        context_vector = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        # ########################## ######################################
        # #3.compute attention vector #####################################

        # Combine the context vector with the current target hidden state
        # to yield the final attention vector

        attention_vector = torch.tanh(self.output_proj(torch.cat((context_vector, input), dim=1)))
        return attention_vector, attn_scores


class SimpleLSTMDecoder(FairseqDecoder):

    def __init__(
        self,
        dictionary,
        out_embed_dim=128,  # decoder_out_embedding_dim
        embed_dim=128,  # embedded target tokens?
        hidden_size=128,
        encoder_output_units=128,  # encoder's final hidden state?
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention=True,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
    ):
        super().__init__(dictionary)

        self.need_attn = True
        self.hidden_size = hidden_size
        self.max_target_positions = max_target_positions
        self.num_layers = num_layers
        self.encoder_output_units = encoder_output_units

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        # Our decoder will embed the inputs before feeding them to the LSTM.
        # the target sentence
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx)

        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=input_feed_size + embed_dim
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )

        # # We'll use a single-layer, unidirectional LSTM for simplicity.
        # self.lstm = nn.LSTM(
        #     # For the first layer we'll concatenate the Encoder's final hidden
        #     # state with the embedded target tokens.
        #     input_size=encoder_hidden_dim + embed_dim,
        #     hidden_size=hidden_dim,
        #     num_layers=1,
        #     bidirectional=False,
        # )

        # Define the output projection.
        # Left from the simple LSTM. Do I need it?
        # self.output_projection = nn.Linear(hidden_dim, len(dictionary))

        if attention:
            # TODO make bias configurable
            #
            self.attention = AttentionLayer(
                hidden_size, encoder_output_units, hidden_size, bias=False
            )
        else:
            self.attention = None

        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """

        # Extract the final hidden state from the Encoder.
        #  final_encoder_hidden are called encoder_outs in LSTM source code
        # final_encoder_hidden = encoder_out['final_hidden']
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)

        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # tgt_len is 'seqlen' in the LSTM sourcecode
        bsz, tgt_len = prev_output_tokens.size()

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(prev_output_tokens)

        # Apply dropout.
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        # bsz, tgt_len, vec_size -> tgt_len, bsz, vec_size
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        # attention
        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores = (
            x.new_zeros(srclen, tgt_len, bsz) if self.attention is not None else None
        )
        outs = []

        for j in range(tgt_len):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)

                # ##@Just skipping this for now
                # if self.residuals:
                # input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                # hidden = Tensor(bsz, hidden_size)
                # encoder_outs = Tensor(srclen, bsz, hidden_size)
                # encoder_padding_mask = Tensor(srclen, bsz)
                # out is attention_vector
                out, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask
                )
            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # #@Just skipping this for now

        # Stack all the necessary tensors together and store
        # prev_hiddens_tensor = torch.stack(prev_hiddens)
        # prev_cells_tensor = torch.stack(prev_cells)
        # cache_state = torch.jit.annotate(
        #     Dict[str, Optional[Tensor]],
        #     {
        #         "prev_hiddens": prev_hiddens_tensor,
        #         "prev_cells": prev_cells_tensor,
        #         "input_feed": input_feed,
        #     },
        # )
        # #@Just skipping this for now
        # self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # Concatenate the Encoder's final hidden state to *every* embedded
        # target token.

        # RuntimeError: expand(torch.cuda.FloatTensor{[219, 1, 32, 256]}, size=[32, 298, -1]):
        # the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)
        # x = torch.cat(
        #     [x, encoder_outs.unsqueeze(1).expand(bsz, tgt_len, -1)],
        #     dim=2,
        # )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(tgt_len, bsz, self.hidden_size)

        # # Using PackedSequence objects in the Decoder is harder than in the
        # # Encoder, since the targets are not sorted in descending length order,
        # # which is a requirement of ``pack_padded_sequence()``. Instead we'll
        # # feed nn.LSTM directly.
        # initial_state = (
        #     encoder_outs.unsqueeze(0),  # hidden
        #     torch.zeros_like(encoder_outs).unsqueeze(0),  # cell
        # )
        # output, _ = self.lstm(
        #     x.transpose(0, 1),  # convert to shape `(tgt_len, bsz, dim)`
        #     initial_state,
        # )

        x = x.transpose(1, 0)  # convert to shape `(bsz, tgt_len, hidden)`

        # Project the outputs to the size of the vocabulary.
        # x = self.output_projection(x)

        # Return the logits and ``None`` for the attention weights
        # return x, None

        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return self.output_layer(x), attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        # if self.adaptive_softmax is None:
        #     if self.share_input_output_embed:
        #         x = F.linear(x, self.embed_tokens.weight)
        #     else:
        #
        x = self.fc_out(x)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

# define a named architecture with the configuration for our
# model. This is done with the register_model_architecture()
# function decorator.

# this named architecture can be used with the --arch
# command-line argument, e.g., --arch tutorial_simple_lstm:

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.


@register_model_architecture('simple_lstm', 'tutorial_simple_lstm')
def tutorial_simple_lstm(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    # args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    # args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)

    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 256)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    # args.knowledge = getattr(args, 'knowledge', None)


# #############################################################################
# #############################################################################

# class SimpleLSTMEncoder(FairseqEncoder):
#
#     def __init__(
#         self, args, dictionary, embed_dim=128, hidden_dim=128, dropout=0.1,
#     ):
#         super().__init__(dictionary)
#         self.args = args
#
#         # Our encoder will embed the inputs before feeding them to the LSTM.
#         self.embed_tokens = nn.Embedding(
#             num_embeddings=len(dictionary),
#             embedding_dim=embed_dim,
#             padding_idx=dictionary.pad(),
#         )
#         self.dropout = nn.Dropout(p=dropout)
#
#         # We'll use a single-layer, unidirectional LSTM for simplicity.
#         self.lstm = nn.LSTM(
#             input_size=embed_dim,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             bidirectional=False,
#             batch_first=True,
#         )
#
#     def forward(self, src_tokens, src_lengths):
#         # The inputs to the ``forward()`` function are determined by the
#         # Task, and in particular the ``'net_input'`` key in each
#         # mini-batch. We discuss Tasks in the next tutorial, but for now just
#         # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
#         # has shape `(batch)`.
#
#         # Note that the source is typically padded on the left. This can be
#         # configured by adding the `--left-pad-source "False"` command-line
#         # argument, but here we'll make the Encoder handle either kind of
#         # padding by converting everything to be right-padded.
#         if self.args.left_pad_source:
#             # Convert left-padding to right-padding.
#             src_tokens = utils.convert_padding_direction(
#                 src_tokens,
#                 padding_idx=self.dictionary.pad(),
#                 left_to_right=True
#             )
#
#         # Embed the source.
#         x = self.embed_tokens(src_tokens)
#
#         # Apply dropout.
#         x = self.dropout(x)
#
#         # Pack the sequence into a PackedSequence object to feed to the LSTM.
#         x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)
#
#         # Get the output from the LSTM.
#         _outputs, (final_hidden, _final_cell) = self.lstm(x)
#
#         # Return the Encoder's output. This can be any object and will be
#         # passed directly to the Decoder.
#         return {
#             # this will have shape `(bsz, hidden_dim)`
#             'final_hidden': final_hidden.squeeze(0),
#         }
#
#     # Encoders are required to implement this method so that we can rearrange
#     # the order of the batch elements during inference (e.g., beam search).
#     def reorder_encoder_out(self, encoder_out, new_order):
#         """
#         Reorder encoder output according to `new_order`.
#
#         Args:
#             encoder_out: output from the ``forward()`` method
#             new_order (LongTensor): desired order
#
#         Returns:
#             `encoder_out` rearranged according to `new_order`
#         """
#         final_hidden = encoder_out['final_hidden']
#         return {
#             'final_hidden': final_hidden.index_select(0, new_order),
#         }
