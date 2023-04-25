#' Wav2Vec2Model
#'
#' Acoustic model used in *wav2vec 2.0* (`Baevski et al., 2020`)
#'
#' @param feature_extractor (nn_module) Feature extractor that extracts feature
#' vectors from raw audio Tensor.
#' @param encoder (nn_module) Encoder that converts the audio features into the
#' sequence of probability distribution (in negative log-likelihood) over labels.
#' @param aux (nn_module, optional) Auxiliary module. If provided, the output
#' from encoder is passed to this module.
#'
#' @note To build the model, please use one of the factory functions.
#'
#' @references
#'  - Baevski et al. 2020. wav2vec 2.0: A Framework for Self-Supervised Learning
#'   of Speech Representations.
#'
#' @seealso [model_wav2vec2bundle()], [model_wav2vec2asrbundle()]
#'
#' @export
model_wav2vec2 <- torch::nn_module(
  "Wav2Vec2Model",
  initialize = function(
    feature_extractor,
    encoder,
    aux = NULL
  ) {
    self$feature_extractor = feature_extractor
    self$encoder = encoder
    self$aux = aux
  },

  #' @description
  #' This returns the list of outputs from the intermediate layers of
  #' transformer block in encoder.
  #'
  #' @param waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
  #' @param lengths (Tensor, optional): Indicates the valid length of
  #' each audio in the batch. Shape: `(batch, )`.When the `waveforms` contains
  #' audios with different durations, by providing `lengths` argument, the
  #' model will compute the corresponding valid output lengths and apply proper
  #' mask in transformer attention layer. If `NULL`, it is assumed that the
  #' entire audio waveform length is valid.
  #' @param num_layers (int, optional): If given, limit the number of
  #' intermediate layers to go through. Providing `1` will stop the computation
  #' after going through one intermediate layers. If not given, the outputs from
  #' all the intermediate layers are returned.
  #'
  #' @returns list(List of Tensors, Tensor or NULL):
  #' List of Tensors:
  #'  Features from requested layers.
  #'  Each Tensor is of shape: `(batch, time frame, feature dimension)`
  #' Tensor or NULL:
  #'  If `lengths` argument was provided, a Tensor of shape `(batch, )` is
  #'  returned. It indicates the valid length in time axis of each feature
  #'  Tensor.
  extract_features = function(
    waveforms,
    lengths = NULL,
    num_layers = NULL
  ) {
    c(x, lengths) %<-% self$feature_extractor(waveforms, lengths)
    x = self$encoder$extract_features(x, lengths, num_layers)
    return(list(x, lengths))
  },

  #' @description
  #' Compute the sequence of probability distribution over labels.
  #'
  #' @param waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
  #' @param lengths (Tensor, optional): Indicates the valid length of
  #' each audio in the batch. Shape: `(batch, )`. When the `waveforms`
  #' contains audios with different durations, by providing `lengths` argument,
  #' the model will compute the corresponding valid output lengths and apply
  #' proper mask in transformer attention layer. If NULL, it is assumed that
  #' all the audio in `waveforms` have valid length. Default: NULL.
  #'
  #' @returns list(Tensor, Tensor or NULL):
  #' Tensor:
  #'  The sequences of probability distribution (in logit) over labels.
  #'  Shape: `(batch, frames, num labels)`.
  #' Tensor or NULL:
  #'  If `lengths` argument was provided, a Tensor of shape `(batch, )` is
  #'  returned. It indicates the valid length in time axis of the output Tensor.
  forward = function(
    waveforms,
    lengths = NULL
  ) {
    c(x, lengths) %<-% self$feature_extractor(waveforms, lengths)
    x = self$encoder(x, lengths)

    if (!is.null(self$aux)) {
      x = self$aux(x)
    }

    return(list(x, lengths))
  }
)

#' model_wav2vec2_build
#'
#' Builds custom `~torchaudio.models.Wav2Vec2Model`.
#'
#' @details
#' The "feature extractor" below corresponds to
#' [ConvFeatureExtractionModel](https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736)
#' in the original `fairseq` implementation.
#' This is referred as " (convolutional) feature encoder" in the *wav2vec 2.0*
#' paper(Bevski et al., 2020).
#'
#' The "encoder" below corresponds to [TransformerEncoder](https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817),
#' and this is referred as "Transformer" in the paper.
#'
#' @param extractor_mode  (str): Operation mode of feature extractor.
#' Valid values are `"group_norm"` or `"layer_norm"`.
#' If `"group_norm"`, then a single normalization is applied
#' in the first convolution block. Otherwise, all the convolution
#' blocks will have layer normalization. This option corresponds to
#' `extractor_mode` from `fairseq`.
#'
#' @param extractor_conv_layer_config  (list of integer vectors or NULL):
#' Configuration of convolution layers in feature extractor. List of convolution
#' configuration, i.e. `[ (output_channel, kernel_size, stride), ...]`.
#' If `NULL` is provided, then the following default value is used.
#'
#' list(
#'  c(512, 10, 5),
#'  c(512, 3, 2),
#'  c(512, 3, 2),
#'  c(512, 3, 2),
#'  c(512, 3, 2),
#'  c(512, 2, 2),
#'  c(512, 2, 2)
#' )
#'
#' This option corresponds to `conv_feature_layers` from `fairseq`.
#'
#' @param extractor_conv_bias  (bool): Whether to include bias term to each
#' convolution operation. This option corresponds to `conv_bias` from `fairseq`.
#'
#' @param encoder_embed_dim  (int): The dimension of embedding in encoder. This
#' option corresponds to `encoder_embed_dim` from `fairseq`.
#'
#' @param encoder_projection_dropout  (float): The dropout probability applied
#' after the input feature is projected to `encoder_embed_dim`. This option
#' corresponds to `dropout_input` from `fairseq`.
#'
#' @param encoder_pos_conv_kernel  (int): The kernel size of convolutional
#' positional embeddings. This option corresponds to `conv_pos` from `fairseq`.
#'
#' @param encoder_pos_conv_groups  (int): The number of groups of convolutional
#' positional embeddings. This option corresponds to `conv_pos_groups` from `fairseq`.
#'
#' @param encoder_num_layers  (int): The number of self attention layers in
#' transformer block. This option corresponds to `encoder_layers` from `fairseq`.
#'
#' @param encoder_num_heads  (int): The number of heads in self attention layers.
#' This option corresponds to `encoder_attention_heads` from `fairseq`.
#'
#' @param encoder_attention_dropout  (float): The dropout probability applied
#' after softmax in self-attention layer. This option corresponds to
#' `attention_dropout` from `fairseq`.
#'
#' @param encoder_ff_interm_features  (int): The dimension of hidden features in
#'  feed forward layer. This option corresponds to `encoder_ffn_embed_dim` from `fairseq`.
#'
#' @param encoder_ff_interm_dropout  (float): The dropout probability applied in
#'  feedforward layer. This option correspinds to `activation_dropout` from `fairseq`.
#'
#' @param encoder_dropout  (float): The dropout probability applied at the end
#' of feed forward layer. This option corresponds to `dropout` from `fairseq`.
#'
#' @param encoder_layer_norm_first  (bool):
#' Control the order of layer norm in transformer layer and each encoder layer.
#' If TRUE, in transformer layer, layer norm is applied before features are fed
#' to encoder layers. In encoder layer, two layer norms are applied before and after
#' self attention.
#' If FALSE, in transformer layer, layer norm is applied after features are fed
#' to encoder layers. In encoder layer, two layer norms are applied after self
#' attention, before and after feed forward.
#' This option corresponds to `layer_norm_first` from `fairseq`.
#'
#' @param encoder_layer_drop  (float): Probability to drop each encoder layer
#' during training. This option corresponds to `layerdrop` from `fairseq`.
#'
#' @param aux_num_out  (int or NULL): When provided, attach an extra linear
#' layer on top of encoder, which can be used for fine-tuning.
#'
#' @returns Wav2Vec2Model: The resulting model.
#'
#' @export
model_wav2vec2_build <- function(
    extractor_mode,
    extractor_conv_layer_config = NULL,
    extractor_conv_bias,
    encoder_embed_dim,
    encoder_projection_dropout,
    encoder_pos_conv_kernel,
    encoder_pos_conv_groups,
    encoder_num_layers,
    encoder_num_heads,
    encoder_attention_dropout,
    encoder_ff_interm_features,
    encoder_ff_interm_dropout,
    encoder_dropout,
    encoder_layer_norm_first,
    encoder_layer_drop,
    aux_num_out = NULL
) {
  if(is.null(extractor_conv_layer_config)) {
    extractor_conv_layer_config = list(
      c(512, 10, 5),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 2, 2),
      c(512, 2, 2)
    )
  }

  feature_extractor = .get_feature_extractor(
    extractor_mode,
    extractor_conv_layer_config,
    extractor_conv_bias
  )

  encoder = .get_encoder(
    in_features = tail(extractor_conv_layer_config, 1)[[1]][1],
    embed_dim = encoder_embed_dim,
    dropout_input = encoder_projection_dropout,
    pos_conv_kernel = encoder_pos_conv_kernel,
    pos_conv_groups = encoder_pos_conv_groups,
    num_layers = encoder_num_layers,
    num_heads = encoder_num_heads,
    attention_dropout = encoder_attention_dropout,
    ff_interm_features = encoder_ff_interm_features,
    ff_interm_dropout = encoder_ff_interm_dropout,
    dropout = encoder_dropout,
    layer_norm_first = encoder_layer_norm_first,
    layer_drop = encoder_layer_drop
  )
  aux = NULL

  if(!is.null(aux_num_out)) {
    aux = torch::nn_linear(
      in_features = encoder_embed_dim,
      out_features = aux_num_out
    )
  }

  return(model_wav2vec2(feature_extractor, encoder, aux))
}

#' Builds custom WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
#'    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output object is
#'    :class:`~torchaudio.models.Wav2Vec2Model`. Most of the arguments have the same meaning
#'    as in :py:func:`~torchaudio.models.wav2vec2_model` so please refer there for documentation.
#' @param extractor_mode  (str): Operation mode of feature extractor.
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param extractor_conv_layer_config  (list of integer tuples or NULL):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param extractor_conv_bias  (bool):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_embed_dim  (int):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_projection_dropout  (float):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_pos_conv_kernel  (int):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_pos_conv_groups  (int):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_num_layers  (int):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_num_heads  (int):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_num_buckets  (int):
#'            Number of buckets for relative position embedding.
#' @param encoder_max_distance  (int):
#'            Maximum distance for relative position embedding.
#' @param encoder_attention_dropout  (float):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_ff_interm_features  (int):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_ff_interm_dropout  (float):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_dropout  (float):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_layer_norm_first  (bool):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param encoder_layer_drop  (float):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @param aux_num_out  (int or NULL):
#'            See :py:func:`~torchaudio.models.wav2vec2_model`.
#' @return Wav2Vec2Model:
#'            The resulting model.
#'
#' @export
model_wavlm_build <- function(
    extractor_mode,
    extractor_conv_layer_config,
    extractor_conv_bias,
    encoder_embed_dim,
    encoder_projection_dropout,
    encoder_pos_conv_kernel,
    encoder_pos_conv_groups,
    encoder_num_layers,
    encoder_num_heads,
    encoder_num_buckets,
    encoder_max_distance,
    encoder_attention_dropout,
    encoder_ff_interm_features,
    encoder_ff_interm_dropout,
    encoder_dropout,
    encoder_layer_norm_first,
    encoder_layer_drop,
    aux_num_out
) {
  if(is.null(extractor_conv_layer_config)) {
    extractor_conv_layer_config = list(
      c(512, 10, 5),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 2, 2),
      c(512, 2, 2)
    )
  }
  feature_extractor = .get_feature_extractor(
    extractor_mode,
    extractor_conv_layer_config,
    extractor_conv_bias
  )

  encoder = .get_wavlm_encoder(
    in_features = tail(extractor_conv_layer_config, 1)[[1]][1],
    embed_dim = encoder_embed_dim,
    dropout_input = encoder_projection_dropout,
    pos_conv_kernel = encoder_pos_conv_kernel,
    pos_conv_groups = encoder_pos_conv_groups,
    num_layers = encoder_num_layers,
    num_heads = encoder_num_heads,
    num_buckets = encoder_num_buckets,
    max_distance = encoder_max_distance,
    attention_dropout = encoder_attention_dropout,
    ff_interm_features = encoder_ff_interm_features,
    ff_interm_dropout = encoder_ff_interm_dropout,
    dropout = encoder_dropout,
    layer_norm_first = encoder_layer_norm_first,
    layer_drop = encoder_layer_drop,
  )
  aux = NULL

  if(!is.null(aux_num_out)) {
    aux = torch::nn_linear(
      in_features = encoder_embed_dim,
      out_features = aux_num_out
    )
  }

  return(model_wav2vec2(feature_extractor, encoder, aux))
}




#' .get_encoder
#'
#' @param in_features  (int): The number of input features.
#' @param embed_dim  (int):
#'            The dimension of embedding.
#'            This option corresponds to "encoder_embed_dim" from fairseq.
#'            Expected values are 768 for Base arch, and 1024 for Large arch.
#' @param dropout_input  (float):
#'            The dropout probability applied after the input feature is projected
#'            to `embed_dim`.
#'            This option corresponds to "dropout_input" from fairseq.
#'            Expected values are 0.1 for both Base and Large arch.
#' @param pos_conv_kernel  (int):
#'            The kernel size of convolutional positional embeddings.
#'            This option corresponds to "conv_pos" from fairseq.
#'            Expected values are 128 for both Base and Large arch.
#' @param pos_conv_groups  (int):
#'            The number of groups of convolutional positional embeddings.
#'            This option corresponds to "conv_pos_groups" from fairseq.
#'            Expected values are 16 for both Base and Large arch.
#' @param num_layers  (int):
#'            The number of self attention layers in transformer block.
#'            This option corresponds to "encoder_layers" from fairseq.
#'            Expected values are 12 for Base and 24 for Large arch.
#' @param num_heads  (int):
#'            The number of heads in self attention layers.
#'            This option corresponds to "encoder_attention_heads" from fairseq.
#'            Expected values are 12 for Base and 16 for Large arch.
#' @param attention_dropout  (float):
#'            The dropout probability applied after softmax in self-attention layer.
#'            This option corresponds to "attention_dropout" from fairseq.
#'            Expected values are 0.1 for Base and 0.0 for Large arch.
#' @param ff_interm_features  (int):
#'            The dimension of hidden features in feed forward layer.
#'            This option corresponds to "encoder_ffn_embed_dim" from fairseq.
#'            Expected values are 3072 for Base and 4096 for Large arch.
#' @param ff_interm_dropout  (float):
#'            The dropout probability applied in feedforward layer.
#'            This option correspinds to "activation_dropout" from fairseq.
#'            Expected values are 0.1 for both Base and Large arch.
#' @param dropout  (float):
#'            The dropout probability applied at the end of feed forward layer.
#'            This option corresponds to "dropout" from fairseq.
#'            Expected values are 0.1 for Base and 0.0 for Large arch.
#' @param layer_norm_first  (bool):
#'            Control the order of layer norm in transformer layer and each encoder layer.
#'            If TRUE, in transformer layer, layer norm is applied before features are fed
#'            to encoder layers. In encoder layer, two layer norms are applied before and after
#'            self attention.
#'            If FALSE, in transformer layer, layer norm is applied after features are fed
#'            to encoder layers. In encoder layer, two layer norms are applied after self
#'            attention, before and after feed forward.
#'            This option corresponds to "layer_norm_first" from fairseq.
#'            Expected values are FALSE for Base and TRUE for Large arch.
#' @param layer_drop  (float):
#'            Probability to drop each encoder layer during training.
#'            This option corresponds to "layerdrop" from fairseq.
#'            Expected values are 0.1 for both Base and Large arch.
#'
#'  @details
#'  See also:
#'
#'        * "encoder_embed_dim"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L49-L51
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L64
#'        * "dropout_input"
#'          - Def, base and large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L75-L78
#'        * "conv_pos"
#'          - Def, base and large
#'            NOTE: The description is wrong.
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L204-L207
#'          - Usage
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L756
#'        * "conv_pos_groups"
#'          - Def, base and large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L208-L211
#'        * "encoder_layers"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L46-L48
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L63
#'        * "encoder_attention_heads"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L55-L57
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L66
#'        * "attention_dropout"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L66-L68
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L60
#'        * "encoder_ffn_embed_dim"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L52-L54
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L65
#'        * "activation_dropout"
#'          - Def
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L69-L71
#'          - Base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L55
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L55
#'        * "dropout"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L63-L65
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L59
#'        * "layer_norm_first"
#'          - Def and base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L91-L93
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L53
#'        * "layerdrop"
#'          - Def
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L72-L74
#'          - Base
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L54
#'          - Large
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L54
#'
#' @keywords internal
.get_encoder <- function(
    in_features,
    embed_dim,
    dropout_input,
    pos_conv_kernel,
    pos_conv_groups,
    num_layers,
    num_heads,
    attention_dropout,
    ff_interm_features,
    ff_interm_dropout,
    dropout,
    layer_norm_first,
    layer_drop
) {
  feature_projection = model_feature_projection(
    in_features,
    embed_dim,
    dropout_input
  )
  pos_conv = model_convolutional_positional_embedding(
    embed_dim,
    pos_conv_kernel,
    pos_conv_groups
  )

  # Original impl
  # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
  encoder_layers = torch::nn_module_list()
  for (. in  seq.int(num_layers)) {
    attention = model_self_attention(
      embed_dim = embed_dim,
      num_heads = num_heads,
      dropout = attention_dropout
    )

    feed_forward = model_feed_forward(
      io_features = embed_dim,
      intermediate_features = ff_interm_features,
      intermediate_dropout = ff_interm_dropout,
      output_dropout = dropout
    )

    encoder_layers$append(
      model_encoder_layer(
        attention = attention,
        dropout = dropout,
        layer_norm_first = layer_norm_first,
        feed_forward = feed_forward
      )
    )
  }

  transformer = model_transformer(
    pos_conv_embed = pos_conv,
    dropout = dropout,
    layers = encoder_layers,
    layer_norm_first = !layer_norm_first,
    layer_drop = layer_drop
  )

  return(model_encoder(feature_projection, transformer))
}

# https://github.com/pytorch/audio/blob/bcfa9eed5d96ccf4c302c0594af77e103c482555/torchaudio/models/wav2vec2/components.py
.get_wavlm_encoder <- function(...) stop("To do!")

#' .get_feature_extractor
#'
#' @param norm_mode  (str):
#'            Either "group_norm" or "layer_norm".
#'            If "group_norm", then a single normalization is applied
#'            in the first convolution block. Otherwise, all the convolution
#'            blocks will have layer normalization.
#'            This option corresponds to "extractor_mode" from fairseq.
#'            Expected values are "group_norm" for Base arch, and
#'            "layer_norm" for Large arch.
#' @param shapes  (list of tuple of int):
#'            Configuration of convolution layers. List of convolution
#'            configuration, i.e. `[ (output_channel, kernel_size, stride), ...]`
#'            This option corresponds to "conv_feature_layers" from fairseq.
#'            Expected values are `[ (512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2`
#'            for all the architectures.
#' @param bias  (bool):
#'            Whether to include bias term to each convolution operation.
#'            This option corresponds to "conv_bias" from fairseq.
#'            Expected values are FALSE for Base arch, and TRUE for Large arch.
#'
#' @details
#'
#'    See Also:
#'        * Original implementation
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L666-L733
#'        * "extractor_mode"
#'          - Def and base:
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L38-L45
#'          - Large:
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L52
#'        * "conv_feature_layers"
#'          - Def, base and large:
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L94-L100
#'        * "conv_bias"
#'          - Def and base:
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L101-L103
#'          - Large:
#'            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L61
#'
#' @keywords internal
.get_feature_extractor <- function(
    norm_mode,
    shapes,
    bias
) {
  if(norm_mode %not_in% c("group_norm", "layer_norm"))
    value_error("Invalid norm mode")

  blocks = list()
  in_channels = 1
  for (i in seq_along(shapes)) {
    c(out_channels, kernel_size, stride) %<-% shapes[[i]]
    normalization = NULL

    if(norm_mode == "group_norm" & i == 1) {
      normalization = torch::nn_group_norm(
        num_groups = out_channels,
        num_channels = out_channels,
        affine = TRUE
      )
    } else if (norm_mode == "layer_norm") {
      normalization = model_layer_norm(
        normalized_shape = out_channels,
        elementwise_affine = TRUE
      )
    }

    blocks = c(
      blocks,
      model_conv_layer_block(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        bias = bias,
        layer_norm = normalization
      )
    )
    in_channels = out_channels
  }

  return(module_feature_extractor(torch::nn_module_list(blocks)))
}

#' Multihead Self Attention module
#'
#' @param embed_dim  (int): Total dimension of the model.
#' @param num_heads  (int): The number of heads.
#' @param dropout  (float, optional): Dropout probability on attn_output_weights.
#' Default: `0.0`
#'
#' @export
model_self_attention <- torch::nn_module(
  "SelfAttention",
  initialize = function(
    embed_dim,
    num_heads,
    dropout = 0.0
  ) {
    head_dim = embed_dim %/% num_heads
    if(head_dim * num_heads != embed_dim)
      value_error(f("`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`"))

    self$embed_dim = embed_dim
    self$num_heads = num_heads
    self$dropout = torch::nn_dropout(dropout)
    self$head_dim = head_dim

    self$scaling = self$head_dim^(-0.5)

    self$k_proj = torch::nn_linear(embed_dim, embed_dim, bias = TRUE)
    self$v_proj = torch::nn_linear(embed_dim, embed_dim, bias = TRUE)
    self$q_proj = torch::nn_linear(embed_dim, embed_dim, bias = TRUE)
    self$out_proj = torch::nn_linear(embed_dim, embed_dim, bias = TRUE)
  },

  #' @param x  (Tensor): shape: `[batch_size, sequence_length, embed_dim]`.
  #' @param attention_mask  (Tensor or `NULL`, optional):
  #' shape: `[batch_size, 1, sequence_length, sequence_length]`
  #' position_bias: Not used. Only for the compatibility with `WavLMSelfAttention`.
  #' @param key_padding_mask  (Tensor or `NULL`): Not used. Only for the
  #' compatibility with `WavLMSelfAttention`.
  #' @returns list(Tensor, `NULL`): The resulting attention output and `NULL`
  #' (necessary for compatibility with `WavLMSelAttention`).
  #' Attention output shape: `[batch, sequence_length, embed_dim]`.
  forward = function(
    x,
    attention_mask = NULL,
    position_bias = NULL,
    key_padding_mask = NULL
  ) {
    if(n_dim(x) != 3 || dim(x)[3] != self$embed_dim)
      value_error(
        f("The expected input shape is c(batch, sequence,
               embed_dim=={self$embed_dim}). Found {toString(dim(x))}.")
      )

    c(batch_size, length, embed_dim) %<-% dim(x)
    if(!is.null(attention_mask)) {
      shape_ = c(batch_size, 1, length, length)
      if(!identical(dim(attention_mask), shape_))
        value_error(
          f("The expected attention mask shape is {toString(shape_)}.
                  Found {toString(dim(attention_mask))}.")
        )
    }

    shape = c(batch_size, length, self$num_heads, self$head_dim)
    q = self$q_proj(x)$view(shape)$transpose(3, 2) # B, nH, L, Hd
    k = self$k_proj(x)$view(shape)$permute(c(1, 3, 4, 2)) # B, nH, Hd, L
    v = self$v_proj(x)$view(shape)$transpose(3, 2) # B, nH, L, Hd

    # scale down q to avoid value overflow.
    weights = torch::torch_matmul(self$scaling * q, k) # B, nH, L, L
    if(!is.null(attention_mask))
      weights = weights + attention_mask

    # subtracting a constant value from the tensor won't change the output of softmax.
    # apply the subtraction to avoid value overflow in torch::torch_nn.functional.softmax.
    # for more details, please see Equation 7 in https://arxiv.org/abs/2112.08778
    weights = weights - weights$max(dim = -1, keepdim = TRUE)[[1]]

    weights = torch::nnf_softmax(weights, dim = -1)
    weights = self$dropout(weights)

    output = torch::torch_matmul(weights, v) # B, nH, L, Hd
    output = output$transpose(3, 2)$reshape(c(batch_size, length, embed_dim))

    output = self$out_proj(output)
    return(list(output, NULL)) # Necessary for compatibility with WavLMSelAttention)
  }
)


#' FeedForward
#'
#' Layer that follows attention layer in encoder layer.
#'
#' @seealso [model_self_attention()]
#'
#' @export
model_feed_forward <- torch::nn_module(
  "FeedForward",
  initialize = function(
    io_features,
    intermediate_features,
    intermediate_dropout,
    output_dropout
  ) {
    self$intermediate_dense = torch::nn_linear(io_features, intermediate_features)
    self$intermediate_dropout = torch::nn_dropout(intermediate_dropout)
    self$output_dense = torch::nn_linear(intermediate_features, io_features)
    self$output_dropout = torch::nn_dropout(output_dropout)
  },

  #' @param x  (Tensor): shape: `(batch, sequence_length, io_features)`
  #'
  #' @returns (Tensor): shape: `(batch, sequence_length, io_features)`
  forward = function(x) {
    x = self$intermediate_dense(x)
    x = torch::nnf_gelu(x)
    x = self$intermediate_dropout(x)

    x = self$output_dense(x)
    x = self$output_dropout(x)
    return(x)
  }
)

#' EncoderLayer
#'
#' A layer unit in encoder. Combines multihead self attention and feed forward.
#'
#' @export
model_encoder_layer <- torch::nn_module(
  "EncoderLayer",
  initialize = function(
    attention,
    dropout,
    layer_norm_first,
    feed_forward
  ) {
    self$attention = attention
    self$dropout = torch::nn_dropout(dropout)
    self$layer_norm = torch::nn_layer_norm(attention$embed_dim)
    self$layer_norm_first = layer_norm_first
    self$feed_forward = feed_forward
    self$final_layer_norm = torch::nn_layer_norm(attention$embed_dim)
  },

  #' @param x  (Tensor): Input of shape `(batch, sequence_length, embed_dim)`.
  #' @param attention_mask  (Tensor or `NULL`, optional): attention mask
  #' of shape ` (batch, 1, sequence_length, sequence_length)`. (Default: `NULL`)
  #' @param position_bias  (Tensor or `NULL`, optional): position bias of shape
  #' `(batch_size * num_heads, src_len, src_len)`.
  #' Only necessary for WavLM model, `NULL` otherwise.  (Default: `NULL`)
  #' @param key_padding_mask  (Tensor or `NULL`, optional): key padding mask of
  #' shape `(batch_size, src_len)`.
  #' Only used for WavLM model, ignored otherwise.  (Default: `NULL`)
  #'
  #' @returns (x, position_bias): Shapes are the same as in the input. Position
  #' bias is only relevant for WaLM model, `NULL` otherwise.
  forward = function(
    x,
    attention_mask = NULL,
    position_bias = NULL,
    key_padding_mask = NULL
  ) {
    residual = x

    if(self$layer_norm_first)
      x = self$layer_norm(x)

    c(x, position_bias) %<-% self$attention(
      x,
      attention_mask = attention_mask,
      position_bias = position_bias,
      key_padding_mask = key_padding_mask
    )

    x = self$dropout(x)
    x = residual + x

    if(self$layer_norm_first) {
      x = x + self$feed_forward(self$final_layer_norm(x))
    } else {
      x = self$layer_norm(x)
      x = self$final_layer_norm(x + self$feed_forward(x))
    }

    return(list(x, position_bias))
  }
)

#' Transformer
#'
#' @export
model_transformer <- torch::nn_module(
  "Transformer",
  initialize = function(
    pos_conv_embed,
    dropout,
    layers,
    layer_norm_first,
    layer_drop
  ) {
    self$pos_conv_embed = pos_conv_embed
    self$layer_norm = torch::nn_layer_norm(pos_conv_embed$embed_dim)
    self$layer_norm_first = layer_norm_first
    self$layer_drop = layer_drop
    self$dropout = torch::nn_dropout(dropout)
    self$layers = layers
  },

  .preprocess = function(x) {
    x = x + self$pos_conv_embed(x)

    if(self$layer_norm_first)
      x = self$layer_norm(x)

    x = self$dropout(x)

    return(x)
  },

  forward = function(
    x,
    attention_mask = NULL,
    position_bias = NULL
  ) {
    x = self$.preprocess(x)
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      if(!(self$training & torch::torch_rand(1)$item() <= self$layer_drop)) {
        c(x, position_bias) %<-% layer(x, attention_mask, position_bias = position_bias)
      }
    }

    if(!self$layer_norm_first)
      x = self$layer_norm(x)

    return(x)
  },

  get_intermediate_outputs = function(
    x,
    attention_mask = NULL,
    num_layers = NULL
  ) {
    if(!is.null(num_layers)) {
      if(num_layers > length(self$layers) | num_layers <= 0)
        value_error(f("`num_layers` must be between [1, {length(self$layers)}]"))
    }

    ret = list()
    x = self$.preprocess(x)
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      c(x, .) %<-% layer(x, attention_mask) # Ignore position_bias
      ret <- c(ret, x)

      if(!is.null(num_layers) && (length(ret) >= num_layers))
        return(ret)
    }
    return(ret)
  }
)

#' ConvolutionalPositionalEmbedding
#'
#' Positional embedding which is placed at the beginning of Transformer.
#'
#' @param embed_dim  (int): Feature dimension of the input Tensor.
#' @param kernel_size  (int): The number of frames to be use.
#' @param groups  (int): The number of groups in feature dimensions.
#'
#' @export
model_convolutional_positional_embedding <- torch::nn_module(
  "ConvolutionalPositionalEmbedding",
  initialize = function(
    embed_dim,
    kernel_size,
    groups
  ) {
    self$embed_dim = embed_dim
    self$kernel_size = kernel_size
    self$conv = torch::nn_conv1d(
      in_channels = embed_dim,
      out_channels = embed_dim,
      kernel_size = kernel_size,
      padding = kernel_size %/% 2,
      groups = groups
    )
    self$name = "weight"
    self$dim = 3
    self$weight_norm = torch::nn_utils_weight_norm$new("weight", 3)
    self$conv = self$weight_norm$apply(self$conv)
    self$num_remove = if (kernel_size %% 2 == 0) 1 else 0
  },

  #' @param x  (Tensor): shape `[batch, frame, feature]`.
  #' @returns Tensor: The resulting feature. Shape `[batch, frame, feature]`.
  forward = function(x) {
    # recompute weight before every forward()
    self$conv = self$weight_norm$recompute(self$conv)

    x = x$transpose(-2, -1)
    x = self$conv(x)
    if(self$num_remove > 0)
      x = x[..,  1:(tail(dim(x),1) - self$num_remove)]
    x = torch::nnf_gelu(x)
    x = x$transpose(-2, -1)

    return(x)
  }
)

#' FeatureProjection
#'
#' Layer that connects FeatureExtractor and Encoder
#' Projects features to encoder dimension.
#'
#' @param in_features  (int): Input feature dim.
#' @param out_features  (int): Output feature dim.
#' @param dropout  (float): Dropout probability.
#'
#' @export
model_feature_projection <- torch::nn_module(
  "FeatureProjection",

  initialize = function(
    in_features,
    out_features,
    dropout
  ) {
    self$layer_norm = torch::nn_layer_norm(in_features)
    self$projection = torch::nn_linear(
      in_features,
      out_features
    )
    self$dropout = torch::nn_dropout(dropout)
  },

  #' @param x  (Tensor): Feature Tensor. shape: `[batch, frame, in_feature]`
  #' @returns Tensor: Projected features. `[batch, frame, out_feature]`.
  forward = function(x) {
    x = self$layer_norm(x)
    x = self$projection(x)
    x = self$dropout(x)
    return(x)
  }
)

model_encoder <- torch::nn_module(
  "Encoder",

  initialize = function(
    feature_projection,
    transformer
  ) {
    self$feature_projection = feature_projection
    self$transformer = transformer
  },

  .preprocess = function(
    features,
    lengths = NULL
  ) {
    x = self$feature_projection(features)

    mask = NULL
    if(!is.null(lengths)) {
      c(batch_size, max_len, .) %<-% x$shape
      # create mask for padded elements and zero-out them
      mask = torch::torch_arange(max_len, device = lengths$device)$expand(batch_size, max_len) >= lengths[, NULL]
      x[mask] = 0.0
      # extend the mask to attention shape and set weight
      mask = -10000.0 * mask[ , NULL, NULL,  ]$to(dtype = features$dtype)
      mask = mask.expand(batch_size, 1, max_len, max_len)
    }
    return(list(x, mask))
  },

  forward = function(
    features,
    lengths = NULL
  ) {
    c(x, mask) %<-% self$.preprocess(features, lengths)
    x = self$transformer(x, attention_mask = mask)
    return(x)
  },

  extract_features = function(
    features,
    lengths = NULL,
    num_layers = NULL
  ) {
    c(x, masks) %<-% self$.preprocess(features, lengths)
    output = self$transformer$get_intermediate_outputs(
      x,
      attention_mask = masks,
      num_layers = num_layers
    )
    return(output)
  }
)

#' LayerNorm
#'
#' Layer norm with transpose
#'
#' @export
model_layer_norm <- torch::nn_module(
  classname = "LayerNorm",
  inherit = torch::nn_layer_norm,

  forward = function(input) {
    x = input$transpose(-2, -1)
    x = torch::nnf_layer_norm(
      x,
      self$normalized_shape,
      self$weight,
      self$bias,
      self$eps
    )
    x = x$transpose(-2, -1)
    return(x)
  }
)

#' ConvLayerBlock
#'
#' Convolution unit of model_feature_extractor
#'
#' @export
model_conv_layer_block <- torch::nn_module(
  classname = "ConvLayerBlock",

  initialize = function(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    bias,
    layer_norm
  ) {
    self$kernel_size = kernel_size
    self$stride = stride
    self$layer_norm = layer_norm
    self$conv = torch::nn_conv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      bias=bias
    )
  },

  #' @param x  (Tensor): Shape: `[batch, in_channels, in_frame]`.
  #' @param length  (Tensor or NULL, optional): Shape `[batch, ]`.
  #' @returns Tensor: Shape `[batch, out_channels, out_frames]`.
  #'          Optional[Tensor]: Shape `[batch, ]`.
  forward = function(
    x,
    length
  ) {
    x = self$conv(x)
    if(!is.null(self$layer_norm)) {
      x = self$layer_norm(x)
    }
    x = torch::nnf_gelu(x)

    if(!is.null(length)) {
      length = torch::torch_div(length - self$kernel_size,
                                self$stride,
                                rounding_mode = "floor") + 1
      # When input length is 0, the resulting length can be negative. So fix it here.
      length = torch::torch_max(torch::torch_zeros_like(length), length)
    }
    return(list(x, length))
  }
)

#' FeatureExtractor
#'
#' Extract features from audio
#'
#' @param conv_layers  (torch::nn_module_list): convolution layers
#'
#' @export
module_feature_extractor <- torch::nn_module(
  "FeatureExtractor",

  initialize = function(conv_layers) {
    self$conv_layers = conv_layers
  },

  #' @param x  (Tensor): Input Tensor representing a batch of audio,
  #' shape: `[batch, time]`.
  #' @param length  (Tensor or NULL, optional): Valid length of each input
  #' sample. shape: `[batch, ]`.
  #' @returns
  #' Tensor: The resulting feature, shape: `[batch, frame, feature]`
  #' Optional (Tensor): Valid length of each output sample. shape: `[batch, ]`.
  forward = function(x, length) {
    if(x$ndim != 2) {
      value_error(
        f("Expected the input Tensor to be 2D (batch, time).
          Found: ({paste(x$shape, collapse = ', ')})")
      )
    }
    x = x$unsqueeze(2) # (batch, channel==1, frame)
    for (i in seq_along(self$conv_layers)) {
      layer <- self$conv_layers[[i]]
      c(x, length) %<-% layer(x, length) # (batch, feature, frame)
    }
    x = x$transpose(2, 3) # (batch, frame, feature)
    return(list(x, length))
  }
)
