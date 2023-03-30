import numpy as np
import tensorflow.compat.v1 as tf

# required for TFA MultiHeadAttention
import typing
import warnings

# MultiHead Attention layer.
# https://www.tensorflow.org/addons/api_docs/python/tfa/layers/MultiHeadAttention
#
# Defines the MultiHead Attention operation as described in
# [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
# which takes in the tensors `query`, `key`, and `value`,
# and returns the dot-product attention between them:
#
# q (query) has shape (batch_size, seq_len_q, depth_q)
# k (keys) has shape (batch_size, seq_len_k, depth_k)
# v (values) has shape (batch_size, seq_len_v, depth_v)
#
# k, v are key-value pairs, then seq_len_k = seq_len_v
#
# multi-head attention (output) returns a value based on the compatibility
# of a query over a set of keys (per each query in the query set) then:
#
# multi-head attention has shape (batch_size, seq_len_q, depth_v)
# multi-head attention shape is independent from multi-head parameters
# (head_size, num_heads)
#
# in this implementation of multi-head attention (from TensorFlow AddOns)
# attn_coef is shape (batch_size, num_heads, seq_len_q, seq_len_k)


class MultiHeadAttention(tf.keras.layers.Layer):
    r"""
        Args:
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
            input dimension of `value` or `key` will be used,
            default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
            applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
            output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as
            an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.
    Call Args:
        inputs:  List of `[query, key, value]` where
            * `query`: Tensor of shape `(..., query_elements, query_depth)`
            * `key`: `Tensor of shape '(..., key_elements, key_depth)`
            * `value`: Tensor of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
        mask: a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
            which specifies which query elements can attend to which key elements,
            `1` indicates attention and `0` indicates no attention.
    Output shape:
        * `(..., query_elements, output_size)` if `output_size` is given, else
        * `(..., query_elements, value_depth)` if `value` is given, else
        * `(..., query_elements, key_depth)`
    """

    def __init__(
            self,
            head_size: int,
            num_heads: int,
            output_size: int = None,
            dropout: float = 0.0,
            use_projection_bias: bool = True,
            return_attn_coef: bool = False,
            kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
            kernel_regularizer: typing.Union[str, typing.Callable] = None,
            kernel_constraint: typing.Union[str, typing.Callable] = None,
            bias_initializer: typing.Union[str, typing.Callable] = "zeros",
            bias_regularizer: typing.Union[str, typing.Callable] = None,
            bias_constraint: typing.Union[str, typing.Callable] = None,
            **kwargs
    ):
        warnings.warn(
            "`MultiHeadAttention` will be deprecated in Addons 0.13. "
            "Please use `tf.keras.layers.MultiHeadAttention` instead.",
            DeprecationWarning,
        )

        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self._dropout_rate = dropout

    def build(self, input_shape):

        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = tf.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = tf.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = tf.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key instead of their product
        # saves some computation
        depth = tf.constant(self.head_size, dtype=query.dtype)
        query /= tf.sqrt(depth)

        # Calculate dot product attention (attention scores)
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        # softmax the attention scores
        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer.
        # Recombining heads is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._dropout_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
        )

        return config


# build a mask for self-attention on the transformer decoder layer
def get_decoder_mask(self_attention_inputs):
    # self_attention_input shape is (?, n_timesteps, n_features)
    # get the dimension value of n_timesteps and build the mask
    n_timesteps = self_attention_inputs.shape[1]
    mask = tf.convert_to_tensor(np.tril(np.ones([n_timesteps, n_timesteps]), 0),
                                dtype=tf.float32)
    return mask


# from here, use Transformer structure and notation from
# https://www.tensorflow.org/text/tutorials/transformer

# d_model is the internal dimensionality across all the Transformer layers
# d_model variable comes from a newer version of MHA implementation,
# which does not run on TensorFlow 1.15

# for that reason, this code implements an older MHA implementation
# from TensorFlow AddOns, (to run everything on TF 1.15 CloudTPU) therefore:

# d_model in transformer layers must be equal to (older) MHA.output_size

# MHA.output_size may be equal to MHA.num_heads * MHA.head_size (to simplify everything)

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # project MHA output from (batch_size, seq_len_q, mha.output_size/d_model)
        # to (batch_size, seq_len_q, feed-forward dimensionality (dff)), RELU-activated
        tf.keras.layers.Dense(dff, activation='relu'),
        # project back from (batch_size, seq_len_q, dff)
        # to (batch_size, seq_len_q, mha.output_size/d_model)
        tf.keras.layers.Dense(d_model)
    ])



class EncoderLayer(tf.keras.layers.Layer):
    # keep this encoder-layer implementation intact, therefore
    # get head_size (required for older MHA implementation) as d_model/num_heads

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # self.mha = MultiHeadAttention(d_model=d_model, num_heads)
        # adjust to use the older implementation of MHA
        self.mha = MultiHeadAttention(head_size=int(d_model / num_heads),
                                      num_heads=num_heads,
                                      # do not assign output_size,
                                      # project input to required d_model instead
                                      # (do it before entering the encoder object)
                                      # output_size=d_model,
                                      # ToDo: verify attention coefficients implementation
                                      return_attn_coef=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):

        # attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        # adjust to use the older implementation of MHA
        attn_output = self.mha(inputs=[x, x, x], mask=mask)
        # attn_output has shape (batch_size, input_seq_len, MHA.output_size/d_model)
        attn_output = self.dropout1(attn_output, training=training)

        # residual connection and layer normalization for MHA
        out1 = self.layernorm1(x + attn_output)
        # out1 has shape (batch_size, input_seq_len, MHA.output_size/d_model)

        # point-wise feed-forward
        ffn_output = self.ffn(out1)
        # ffn_output has shape (batch_size, input_seq_len, MHA.output_size/d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        # residual connection and layer normalization for point-wise feed-forward
        out2 = self.layernorm2(out1 + ffn_output)
        # out2 has shape (batch_size, input_seq_len, MHA.output_size/d_model)

        return out2



class DecoderLayer(tf.keras.layers.Layer):
    # keep this decoder-layer implementation intact, therefore
    # get head_size (required for older MHA implementation) as d_model/num_heads
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # self.mha1 = MultiHeadAttention(d_model, num_heads)
        # adjust to use the older implementation of MHA
        self.mha1 = MultiHeadAttention(head_size=int(d_model / num_heads),
                                       num_heads=num_heads,
                                       # do not assign output_size,
                                       # project input tensor to required d_model instead
                                       # (do it before entering the decoder object)
                                       # output_size=d_model,
                                       # ToDo: verify attention coefficients implementation
                                       return_attn_coef=False)

        # self.mha2 = MultiHeadAttention(d_model, num_heads)
        # adjust to use the older implementation of MHA
        self.mha2 = MultiHeadAttention(head_size=int(d_model / num_heads),
                                       num_heads=num_heads,
                                       # do not assign output_size,
                                       # project input tensor to required d_model instead
                                       # (do it before entering the decoder object)
                                       # output_size=d_model,
                                       # ToDo: verify attention coefficients implementation
                                       return_attn_coef=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output has shape (batch_size, input_seq_len, MHA.output_size/d_model)

        attn1 = self.mha1(inputs=[x, x, x],
                          mask=look_ahead_mask)
        # attn1 has shape (batch_size, target_seq_len, MHA.output_size/d_model)

        attn1 = self.dropout1(attn1, training=training)

        # residual connection and layer normalization for masked multi-head attention
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(inputs=[out1, enc_output, enc_output],
                          mask=padding_mask)
        # here is the trick for encoder-decoder attention in Transformer:
        # Q is the decoder masked multi-head attention output (batch_size, target_seq_length, d_model)
        # K is the encoder output (batch_size, input_seq_length, d_model)
        # V is the encoder output (batch_size, input_seq_length, d_model)

        # attn2 has shape (batch_size, target_seq_len, MHA.output_size/d_model)

        attn2 = self.dropout2(attn2, training=training)

        # residual connection and layer normalization for encoder-decoder multi-head attention
        out2 = self.layernorm2(attn2 + out1)
        # out2 has shape (batch_size, target_seq_len, MHA.output_size/d_model)

        ffn_output = self.ffn(out2)
        # ffn_output has shape (batch_size, target_seq_len, MHA.output_size/d_model)

        ffn_output = self.dropout3(ffn_output, training=training)

        # residual connection and layer normalization for point-wise feed-forward
        out3 = self.layernorm3(ffn_output + out2)
        # decoder layer output has shape (batch_size, target_seq_len, MHA.output_size/d_model)

        # return out3, attn_weights_block1, attn_weights_block2
        return out3


class ARDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(ARDecoderLayer, self).__init__()

        # self.mha1 = MultiHeadAttention(d_model=d_model, num_heads)
        # adjust to use the older implementation of MHA
        self.mha1 = MultiHeadAttention(head_size=int(d_model / num_heads),
                                       num_heads=num_heads,
                                       # do not assign output_size,
                                       # project input to required d_model instead
                                       # (do it before entering the decoder object)
                                       # output_size=d_model,
                                       # ToDo: verify attention coefficients implementation
                                       return_attn_coef=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, look_ahead_mask):
        attn1 = self.mha1(inputs=[x, x, x],
                          mask=look_ahead_mask)
        # attn1 has shape (batch_size, no_targets, d_model)

        attn1 = self.dropout1(attn1, training=training)

        # skip the encoder-decoder MHA and go directly to the output layer
        out1 = self.layernorm1(attn1 + x)
        # out1 has shape(batch_size, no_targets, d_model)

        ffn_output = self.ffn(out1)
        # ffn_output has shape (batch_size, no_targets, d_model)

        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)
        # out2 has shape (batch_size, no_targets, d_model)

        return out2


# first: test Transformer-encoder as a basic forecaster
# second: test Transformer-decoder (autoregressive) as a basic forecaster
class BSCTRFM(object):
    # pass features (source tensors: values and positional encodings) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the transformer encoder

        # ToDo: initially, get this value from the configuration dictionary,
        #   (later, it will have to be automatically obtained from the source/target tensors shape)

        # ToDo: use value embedding to a high-dimensional space and compare results
        # ToDo: use a different global positional encoding system and compare results

        # abstract the architecture model to pass all architecture parameters
        # from configuration file

        # IMPORTANT!
        # features['encoder_input'] has shape
        # (batch_size, encoder_input.n_timesteps, encoder_input.depth)
        # (?, 168, 7)
        # but
        # d_model = config_dict['d_model']
        # 256

        # it is required to project the encoder input to d_model
        # to ensure the residual connection works!
        # otherwise trying to add (?, 168, 7) to (?, 168, 256), will raise an error

        # for a multi-series, global forecasting model,
        # use and embedding layer for series id (customer number, for example)
        # then make projection from the encoder input affected by the embedding

        # for a single-series forecasting model
        # make the projection directly from the encoder input

        # ToDo: implement three options for using a embedding layer:
        #       model_params['id_embedding']['use'] = None
        #       model_params['id_embedding']['use'] = 'original'
        #       model_params['id_embedding']['use'] = 'sequential'

        if model_params['id_embedding']['use']:
            # the encoder input will be affected with an id embedding layer
            # there are two alternative ways to build the embedding layer:

            # over features['id'], when the time series identifiers are ordered
            # as in the electricity dataset, or

            # over features['sequential_id], when the time series identifiers are not ordered
            # as in the traffic dataset, then

            # use original (time series) ids, and ignore sequential ids
            if not model_params['id_embedding']['use_sequential']:
                # electricity dataset
                # features['id'] ranges from 1 to 370
                # but the values in the embedding layer will range from 0 to 369
                # then add a -1 scalar tensor to features['id'] first,
                # or another scalar value according to the experiment
                series_id_embedding = tf.keras.layers.Embedding(
                    input_dim=model_params['id_embedding']['input_dim'],
                    output_dim=model_params['id_embedding']['output_dim'],
                    input_length=model_params['id_embedding']['input_length']
                )(features['id'] -
                  tf.convert_to_tensor(model_params['id_embedding']['starting_token']))
                # series_id_embedding has shape (batch_size, input_length, output, dim) (?, 1, 24)
                # cast it to shape (?, output_dim) in order to be repeated

            # use sequential ids, and ignore original (time series) ids
            if model_params['id_embedding']['use_sequential']:
                # traffic dataset
                # features['sequential_id'] ranges from 1 to 963, then subtract 1.0 (starting_token)
                series_id_embedding = tf.keras.layers.Embedding(
                    input_dim=model_params['id_embedding']['input_dim'],
                    output_dim=model_params['id_embedding']['output_dim'],
                    input_length=model_params['id_embedding']['input_length']
                )(features['sequential_id'] -
                  tf.convert_to_tensor(model_params['id_embedding']['starting_token']))
                # series_id_embedding has shape (batch_size, input_length, output, dim) (?, 1, 24)
                # cast it to shape (?, output_dim) in order to be repeated

            series_id_embedding = tf.keras.layers.Reshape(
                target_shape=(model_params['id_embedding']['output_dim'],)
            )(series_id_embedding)
            # series_id_embedding has shape (batch_size, output, dim) (?, 24)
            # repeat it for the number of timesteps of the encoder input

            series_id_embedding = tf.keras.layers.RepeatVector(
                n=model_params['embedding']['hourly']
            )(series_id_embedding)
            # series_id_embedding has shape (batch_size, encoder_input.timesteps, output, dim)
            # (?, 168, 24)
            # it can be now concatenated to features['encoder_input'] which has shape
            # (batch_size, encoder_input.timesteps, encoder_input.depth) (?, 168, 7)
            # to obtain a shape (batch_size, encoder_input.timesteps, output_dim+encoder_input.depth)
            # (?, 168, 31=7+24)
            # 7 from positional encoding + 24 from series id embedding

            composite_input = tf.keras.layers.concatenate(
                inputs=[features['encoder_input'], series_id_embedding]
            )
            # composite_input is (batch_size, encoder_input.timesteps, output_dim+encoder_input.depth)
            # (?, 168, 31)

            #  calculate now the encoder input projection with a series identifier embedding
            encoder_input_projection = tf.keras.layers.Dense(
                model_params['encoder']['d_model'])(composite_input)

        else:
            # the encoder input will be directly projected to d_model
            # ToDo: try a more efficient way to project (CNN, kernel)
            encoder_input_projection = tf.keras.layers.Dense(
                model_params['encoder']['d_model'])(features['encoder_input'])

        # pass a dropout before ingesting the input projection to encoder object
        encoder_input_projection = tf.keras.layers.Dropout(
            model_params['encoder']['input_dropout'])(encoder_input_projection)

        # encoder_input_projection is
        # (?, 168, 31)

        # a list to concatenate all the encoder layers
        encoder_object_list = list()

        # first input to the encoder object is the encoder input projected to d_model
        encoder_object_list.append(encoder_input_projection)

        # iterate on the layers of the encoder object
        for _ in np.arange(model_params['encoder']['num_layers']):
            encoder_object_list.append(
                EncoderLayer(d_model=model_params['encoder']['d_model'],
                             num_heads=model_params['encoder']['num_heads'],
                             dff=model_params['encoder']['dff'],
                             dropout=model_params['encoder']['layer_dropout'])(
                    encoder_object_list[-1],
                    training=True,
                    # fixed sequence-length, no padding mask required
                    mask=None
                )
            )

        # identify encoder object outputs to serve the decoder object
        # encoder_object_list[0] <- encoder_input_projection to d_model
        # encoder_object_list[1] <- output from encoder layer 1
        # encoder_object_list[2] <- output from encoder layer 2
        # ...
        # encoder_object_list[model_params['encoder']['num_layers']] <- encoder output
        # that means the output from the uppermost layer of the encoder object

        encoder_output = encoder_object_list[model_params['encoder']['num_layers']]

        # it has shape (batch_size, encoder_input.n_timesteps, encoder_object.d_model)
        # (?, 168, 256)


        # ToDo: try a more efficient way to project (CNN, kernel)
        decoder_input_projection = tf.keras.layers.Dense(
            model_params['decoder']['d_model'])(features['decoder_input'])
        # decoder_input shape is (?, 168, 7)
        # decoder_input_projection shape is (?, 168, 256)

        # pass a dropout before ingesting the input projection to decoder object
        decoder_input_projection = tf.keras.layers.Dropout(
            model_params['decoder']['input_dropout'])(decoder_input_projection)

        # a list to concatenate all the decoder layers
        decoder_object_list = list()

        # first input to the decoder object is the decoder input projected to d_model
        decoder_object_list.append(decoder_input_projection)

        # use the same mask for all the DecoderLayers in the decoder object
        look_ahead_mask = get_decoder_mask(features['decoder_input'])

        # use the DecoderLayer to complete the base Transformer
        for level in np.arange(model_params['decoder']['num_layers']):
            decoder_object_list.append(
                DecoderLayer(d_model=model_params['decoder']['d_model'],
                             num_heads=model_params['decoder']['num_heads'],
                             dff=model_params['decoder']['dff'],
                             dropout=model_params['decoder']['layer_dropout'])(
                    decoder_object_list[-1],
                    # pass the output from the parallel encoder layer
                    enc_output=encoder_object_list[level + 1],
                    training=True,
                    look_ahead_mask=look_ahead_mask,
                    # fixed sequence-length, no padding mask required
                    padding_mask=None)
            )

        # ToDo: verify if changing encoder-decoder flow from parallel to serial
        #       renders a better predictive performance
        #       (DecoderLayer would have to be changed, so MHA2 receives [out1, out1, out1]
        #        instead of [out1, enc_output, enc_output])

        # identify decoder object outputs
        # decoder_object_list[0] <- decoder_input_projection to d_model
        # decoder_object_list[1] <- output from decoder layer 1
        # decoder_object_list[2] <- output from decoder layer 2
        # ...
        # decoder_object_list[model_params['decoder']['num_layers']] <- decoder output
        # that means the output from the uppermost layer of the decoder object

        decoder_output = decoder_object_list[model_params['decoder']['num_layers']]
        # decoder_output has shape (batch_size, target_seq_length, d_model)
        # (?, 24, 256)

        # decoder_dense_units = 1
        # decoder_dense = tf.keras.layers.Dense(decoder_dense_units, activation="sigmoid")
        #
        # distributed_decoder_dense = tf.keras.layers.TimeDistributed(decoder_dense)(
        #     decoder_output)
        # # distributed_decoder_dense has shape (?, 24, 1)
        #
        # forecast = distributed_decoder_dense

        # ToDo: use a pyramidal structure for the dense output layer
        # build a TimeDistributed Dense flow to produce a multi-layer output

        # get dense layer structure and activations (two lists)
        structure = model_params['dense']['structure']
        activation = model_params['dense']['activation']
        # get indexes for structure levels (as a list)
        indexes = list(np.arange(len(structure)))

        # a dictionary to store the dense layer levels
        dense = dict()
        # iterate via zip on indexes, cells, and activations
        for index, no_units, activation in zip(indexes, structure, activation):
            level_key = 'level_{}'.format(index)
            dense[level_key] = tf.keras.layers.Dense(
                units=no_units,
                activation=activation,
                name='dense_layer_{}'.format(level_key)
            )

        # generalize the dense layer outputs using a list
        output = list()
        # initialize the output list with the output from the decoder object
        output.append(decoder_output)

        # build a list of level keys (on indexes) to iterate on
        level_keys = ['level_{}'.format(index) for index in indexes]

        for level_key in level_keys:
            output.append(tf.keras.layers.TimeDistributed(
                dense[level_key])(output[-1]))

        # at the end of the building loop, the uppermost level of the dense layer
        # is located in the final position of the output list
        forecast = output[-1]

        return forecast
