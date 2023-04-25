
# model_wav2vec2
# extract_features
# model_wav2vec2_build
# .get_encoder - OK - py
# .get_feature_extractor - OK - py
# model_self_attention - OK - py
# model_feed_forward - OK - py
# model_encoder_layer - OK - py
# model_transformer - OK - py
# model_convolutional_positional_embedding - OK - py
# model_feature_projection - OK - py
# model_layer_norm - OK - py
# model_conv_layer_block - OK - py
# module_feature_extractor - OK - py
test_that("model_self_attention", {
  embed_dim = 12
  self_attention = model_self_attention(
    embed_dim = embed_dim,
    num_heads = 6,
    dropout = 0.1
  )

  batch = 3
  sequence_length = 4
  x <- torch::torch_ones(batch, sequence_length, embed_dim)
  expect_no_error(tnsr <- self_attention(x))
  expect_tensor(tnsr[[1]])
  expect_null(tnsr[[2]])
  expect_equal(dim(tnsr[[1]]), c(batch, sequence_length, embed_dim))
})

test_that("model_feed_forward", {
  io_features = 12
  feed_forward = model_feed_forward(
    io_features = io_features,
    intermediate_features = 7,
    intermediate_dropout = 0.5,
    output_dropout = 0.3
  )

  batch = 3
  sequence_length = 4
  x <- torch::torch_ones(batch, sequence_length, io_features)
  expect_no_error(tnsr <- feed_forward(x))
  expect_tensor(tnsr)
  expect_equal(dim(tnsr), dim(x))
})

test_that("model_encoder_layer", {
  embed_dim = 12
  self_attention = model_self_attention(
    embed_dim = embed_dim,
    num_heads = 6,
    dropout = 0
  )
  feed_forward = model_feed_forward(
    io_features = embed_dim,
    intermediate_features = 7,
    intermediate_dropout = 0.5,
    output_dropout = 0.3
  )
  encoder_layer = model_encoder_layer(
    attention = self_attention,
    dropout = 0.4,
    layer_norm_first = TRUE,
    feed_forward = feed_forward
  )

  batch = 3
  sequence_length = 4
  x <- torch::torch_ones(batch, sequence_length, embed_dim)
  expect_no_error(tnsr <- encoder_layer(x))
  expect_tensor(tnsr[[1]])
  expect_null(tnsr[[2]])
  expect_equal(dim(tnsr[[1]]), dim(x))
})

test_that("model_convolutional_positional_embedding", {
  embed_dim = 12
  convolutional_positional_embedding = model_convolutional_positional_embedding(
    embed_dim = embed_dim,
    kernel_size = 8,
    groups = 4
  )

  batch = 3
  sequence_length = 4
  x <- torch::torch_ones(batch, sequence_length, embed_dim)
  expect_no_error(tnsr <- convolutional_positional_embedding(x))
  expect_tensor(tnsr)
  expect_equal(dim(tnsr), dim(x))
})

test_that("model_transformer", {
  embed_dim = 12
  layer_norm_first = TRUE
  self_attention = model_self_attention(
    embed_dim = embed_dim,
    num_heads = 6,
    dropout = 0
  )
  feed_forward = model_feed_forward(
    io_features = embed_dim,
    intermediate_features = 7,
    intermediate_dropout = 0.5,
    output_dropout = 0.3
  )
  encoder_layer = model_encoder_layer(
    attention = self_attention,
    dropout = 0.4,
    layer_norm_first = layer_norm_first,
    feed_forward = feed_forward
  )
  encoder_layers = torch::nn_module_list(
    list(
      encoder_layer,
      encoder_layer
    )
  )
  convolutional_positional_embedding = model_convolutional_positional_embedding(
    embed_dim = embed_dim,
    kernel_size = 8,
    groups = 4
  )
  transformer = model_transformer(
    pos_conv_embed = convolutional_positional_embedding,
    dropout = 0.3,
    layers = encoder_layers,
    layer_norm_first = !layer_norm_first,
    layer_drop = 0.1
  )
  x <- torch::torch_ones(3, 4, embed_dim)
  expect_no_error(tnsr <- transformer(x))
  expect_tensor(tnsr)
  expect_equal(dim(x), dim(tnsr))
})

test_that("model_feature_projection", {
  embed_dim = 12
  feature_projection = model_feature_projection(
    in_features = embed_dim,
    out_features = embed_dim,
    dropout = 0.5
  )

  x <- torch::torch_ones(3, 4, embed_dim)
  expect_no_error(tnsr <- feature_projection(x))
  expect_tensor(tnsr)
  expect_equal(dim(x), dim(tnsr))
})

test_that("model_layer_norm", {
  layer_norm = model_layer_norm(
    normalized_shape = 12,
    eps = 1e-3,
    elementwise_affine = FALSE
  )

  x <- torch::torch_ones(12, 4)
  expect_no_error(tnsr <- layer_norm(x))
  expect_tensor(tnsr)
  expect_equal(dim(x), dim(tnsr))

})

test_that("model_conv_layer_block", {
  conv_layer_block_gen <- function(in_channels, out_channels) {
    conv_layer_block <- model_conv_layer_block(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = 2,
      stride = 1,
      bias = TRUE,
      layer_norm = NULL
    )
    conv_layer_block
  }
  conv_layer_block = conv_layer_block_gen(12, 2)

  x <- torch::torch_ones(12, 40)
  expect_no_error(tnsr <- conv_layer_block(x, length = NULL))
  expect_tensor(tnsr[[1]])
  expect_null(tnsr[[2]])
  expect_equal(dim(tnsr[[1]]), c(2, 39))
})

test_that("module_feature_extractor", {
  conv_layers = torch::nn_module_list(
    list(
      conv_layer_block_gen(1, 2),
      conv_layer_block_gen(2, 3),
      conv_layer_block_gen(3, 4)
    )
  )

  feature_extractor <- module_feature_extractor(
    conv_layers
  )

  x <- torch::torch_ones(1, 40)
  expect_no_error(tnsr <- feature_extractor(x, length = NULL))
  expect_tensor(tnsr[[1]])
  expect_null(tnsr[[2]])
  expect_equal(dim(tnsr[[1]]), c(1, 37, 4))
})

test_that(".get_feature_extractor", {
  extractor_conv_layer_config = list(
    c(512, 4, 5),
    c(512, 3, 2),
    c(512, 3, 2),
    c(512, 3, 2),
    c(512, 3, 2),
    c(512, 2, 2),
    c(512, 2, 2)
  )

  x <- torch::torch_ones(3, 512)

  # group_norm
  feature_extractor <- .get_feature_extractor(
    norm_mode = "group_norm",
    shapes = extractor_conv_layer_config,
    bias = TRUE
  )
  expect_no_error(tnsr <- feature_extractor(x, length = NULL))
  expect_tensor(tnsr[[1]])
  expect_null(tnsr[[2]])
  expect_equal(dim(tnsr[[1]]), c(3, 1, 512))

  # layer_norm
  feature_extractor <- .get_feature_extractor(
    norm_mode = "layer_norm",
    shapes = extractor_conv_layer_config,
    bias = TRUE
  )
  expect_no_error(tnsr <- feature_extractor(x, length = NULL))
  expect_tensor(tnsr[[1]])
  expect_null(tnsr[[2]])
  expect_equal(dim(tnsr[[1]]), c(3, 1, 512))
})

test_that(".get_encoder", {
  encoder = .get_encoder(
    in_features = 8,
    embed_dim = 4,
    dropout_input = 0.1,
    pos_conv_kernel = 2,
    pos_conv_groups = 2,
    num_layers = 2,
    num_heads = 4,
    attention_dropout = 0.1,
    ff_interm_features = 3,
    ff_interm_dropout = 0.1,
    dropout = 0.1,
    layer_norm_first = TRUE,
    layer_drop = 0.1
  )

  x <- torch::torch_ones(2, 6, 8)
  expect_no_error(tnsr <- encoder(x))
  expect_tensor(tnsr)
  expect_equal(dim(tnsr), c(2, 6, 4))
})

test_that(".get_wavlm_encoder", {
  expect_no_error(.get_wavlm_encoder())
})

test_that("model_wav2vec2_build", {

  wav2vec2build <- model_wav2vec2_build(
    extractor_mode = "group_norm",
    extractor_conv_layer_config = list(
      c(512, 10, 5),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 3, 2),
      c(512, 2, 2),
      c(512, 2, 2)
    ),
    extractor_conv_bias = FALSE,
    encoder_embed_dim = 768,
    encoder_projection_dropout = 0.1,
    encoder_pos_conv_kernel = 128,
    encoder_pos_conv_groups = 16,
    encoder_num_layers = 12,
    encoder_num_heads = 12,
    encoder_attention_dropout = 0.1,
    encoder_ff_interm_features = 3072,
    encoder_ff_interm_dropout = 0.0,
    encoder_dropout = 0.1,
    encoder_layer_norm_first = FALSE,
    encoder_layer_drop = 0.05,
    aux_num_out = 29
  )


  x <- torch::torch_ones(3, 768)
  wav2vec2build$feature_extractor(x, NULL)
  # wav2vec2build$encoder(x, NULL)
  wav2vec2build$extract_features(x, lengths = NULL)
  wav2vec2build$forward(x)
})

