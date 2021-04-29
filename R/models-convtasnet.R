#' 1D Convolutional block.
#'
#' @param io_channels  (int): The number of input/output channels, (B, Sc)
#' @param hidden_channels  (int): The number of channels in the internal layers, H.
#' @param kernel_size  (int): The convolution kernel size of the middle layer, P.
#' @param padding  (int): Padding value of the convolution in the middle layer.
#' @param dilation  (int): Dilation value of the convolution in the middle layer.
#' @param no_redisual  (bool): Disable residual block/output.
#'
#' @details
#'  This implementation corresponds to the "non-causal" setting in the paper.
#'
#' @section Reference:
#'  - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
#'    Luo, Yi and Mesgarani, Nima - [https://arxiv.org/abs/1809.07454]()
#'
#' @export
model_conv_block <- torch::nn_module(
  "ConvBlock",
  initialize = function(
    io_channels,
    hidden_channels,
    kernel_size,
    padding,
    dilation = 1,
    no_residual = FALSE
  ) {
    self$conv_layers <- torch::nn_sequential(
      torch::nn_conv1d(in_channels = io_channels, out_channels = hidden_channels, kernel_size = 1),
      torch::nn_prelu(),
      torch::nn_group_norm(num_groups = 1, num_channels = hidden_channels, eps = 1e-08),
      torch::nn_conv1d(
        in_channels = hidden_channels,
        out_channels = hidden_channels,
        kernel_size = kernel_size,
        padding = padding,
        dilation = dilation,
        groups = hidden_channels
      ),
      torch::nn_prelu(),
      torch::nn_group_norm(num_groups = 1, num_channels = hidden_channels, eps = 1e-08),
    )

    self$res_out <- if(no_residual) {
      NULL
    } else {
      torch::nn_conv1d(in_channels = hidden_channels, out_channels = io_channels, kernel_size = 1)
    }

    self$skip_out <- torch::nn_conv1d(in_channels = hidden_channels, out_channels = io_channels, kernel_size = 1)
  },

  forward = function(input) {
    feature <- self$conv_layers(input)
    residual <- if(is.null(self$res_out)) NULL else  self$res_out(feature)

    skip_out <- self$skip_out(feature)

    return(list(residual, skip_out))
  }
)

#' Temporal Convolution Network (TCN) Separation Module
#'
#' Generates masks for separation. This implementation corresponds to
#' the "non-causal" setting in the paper.
#'
#' @param input_dim  (int): Input feature dimension, N.
#' @param num_sources  (int): The number of sources to separate.
#' @param kernel_size  (int): The convolution kernel size of conv blocks, P.
#' @param num_featrs  (int): Input/output feature dimenstion of conv blocks, (B, Sc).
#' @param num_hidden  (int): Intermediate feature dimention of conv blocks, H
#' @param num_layers  (int): The number of conv blocks in one stack, X.
#' @param num_stacks  (int): The number of conv block stacks, R.
#'
#' @details forward param
#' - input (Tensor): 3D Tensor with shape [batch, features, frames]
#'
#' @section References:
#'        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
#'          Luo, Yi and Mesgarani, Nima [https://arxiv.org/abs/1809.07454]()
#'
#' @return  (Tensor): shape [batch, num_sources, features, frames]
#'
#' @export
model_mask_generator <- torch::nn_module(
  "MaskGenerator",

  initialize = function(
    input_dim,
    num_sources,
    kernel_size,
    num_feats,
    num_hidden,
    num_layers,
    num_stacks
  ) {

    self$input_dim = input_dim
    self$num_sources = num_sources

    self$input_norm = torch::nn_group_norm(num_groups = 1, num_channels = input_dim, eps = 1e-8)
    self$input_conv = torch::nn_conv1d(in_channels = input_dim, out_channels = num_feats, kernel_size = 1)

    self$receptive_field = 0
    self$conv_layers = torch::nn_module_list()
    for(s in range(num_stacks)) {
      for(l in range(num_layers)) {
        multi = 2^l
        self$conv_layers$append(
          model_conv_block(
            io_channels = num_feats,
            hidden_channels = num_hidden,
            kernel_size = kernel_size,
            dilation = multi,
            padding = multi,
            # The last ConvBlock does not need residual
            no_residual = (l == (num_layers - 1) && s == (num_stacks - 1)),
          )
        )
        self$receptive_field = self$receptive_field + (if(s == 0 && l == 0) kernel_size else (kernel_size - 1) * multi)
      }
    }

    self$output_prelu = torch::nn_prelu()
    self$output_conv = torch::nn_conv1d(in_channels = num_feats, out_channels = input_dim * num_sources, kernel_size = 1)
  },

  forward = function(input) {
    batch_size = input$shape[1]
    feats = self$input_norm(input)
    feats = self$input_conv(feats)
    output = 0.0
    for(layer in self$conv_layers) {
      residual_skip = layer(feats)
      residual = residual_skip[[1]]
      skip = residual_skip[[2]]
      if(!is.null(residual))  # the last conv layer does not produce residual
        feats = feats + residual
      output = output + skip
    }
    output = self$output_prelu(output)
    output = self$output_conv(output)
    output = torch::torch_sigmoid(output)
    return(output$view(batch_size, self$num_sources, self$input_dim, -1))
  }
)

#' Align frames with strides
#'
#' Pad input Tensor so that the end of the input tensor corresponds with.
#'
#' @param input  (Tensor): 3D Tensor with shape (batch_size, channels==1, frames)
#' @param enc_kernel_size  (int): The convolution kernel size of the encoder/decoder, L.
#' @param enc_num_feats  (int): The feature dimensions passed to mask generator, N.
#'
#' @section Assumption:
#'
#'  The resulting Tensor will be padded with the size of stride  (== kernel_width // 2)
#'  on the both ends in Conv1D
#'
#'        |<--- k_1 --->|
#'        |      |            |<-- k_n-1 -->|
#'        |      |                  |  |<--- k_n --->|
#'        |      |                  |         |      |
#'        |      |                  |         |      |
#'        |      v                  v         v      |
#'        |<---->|<--- input signal --->|<--->|<---->|
#'         stride                         PAD  stride
#'
#'  1. (If kernel size is odd) the center of the last convolution kernel.
#'  2. (If kernel size is even) the end of the first half of the last convolution kernel.
#'
#' @return list(padded_tensor, number_of_paddings_performed)
#'
#' @export
.align_num_frames_with_strides = function(input, enc_kernel_size, enc_stride) {
  batch_size = input$shape[1]
  num_channels  = input$shape[2]
  num_frames = input$shape[3]

  is_odd = enc_kernel_size %% 2
  num_strides = (num_frames - is_odd) %/% enc_stride
  num_remainings = num_frames - (is_odd + num_strides * enc_stride)
  if(num_remainings == 0)
    return(list(input, 0))

  num_paddings = enc_stride - num_remainings
  pad = torch::torch_zeros(
    batch_size,
    num_channels,
    num_paddings,
    dtype = input$dtype(),
    device = input$device(),
  )
  return(list(torch::torch_cat(list(input, pad), 2), num_paddings))
}

#' Conv-TasNet
#'
#' A fully-convolutional time-domain audio separation network. Perform source separation.
#' Generate audio source waveforms. This implementation corresponds to the "non-causal"
#' setting in the paper.
#'
#' @param num_sources  (int): The number of sources to split.
#' @param enc_kernel_size  (int): The convolution kernel size of the encoder/decoder, L.
#' @param enc_num_feats  (int): The feature dimensions passed to mask generator, N.
#' @param msk_kernel_size  (int): The convolution kernel size of the mask generator, P.
#' @param msk_num_feats  (int): The input/output feature dimension of conv block in the mask generator, (B, Sc).
#' @param msk_num_hidden_feats  (int): The internal feature dimension of conv block of the mask generator, H.
#' @param msk_num_layers  (int): The number of layers in one conv block of the mask generator, X.
#' @param msk_num_stacks  (int): The numbr of conv blocks of the mask generator, R.
#'
#' @details forward param:
#' - input  (Tensor): 3D Tensor with shape [batch, channel==1, frames]
#' - .align_num_frames_with_strides
#'
#' @section Reference:
#' - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
#'   Luo, Yi and Mesgarani, Nima [https://arxiv.org/abs/1809.07454]()
#'
#' @return Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
#'
#' @export
model_conv_tasnet <- torch::nn_module(
  "ConvTasNet",
  initialize = function(
    num_sources = 2,
    # encoder/decoder parameters
    enc_kernel_size = 16,
    enc_num_feats = 512,
    # mask generator parameters
    msk_kernel_size = 3,
    msk_num_feats = 128,
    msk_num_hidden_feats = 512,
    msk_num_layers = 8,
    msk_num_stacks = 3
  ) {
    self$num_sources = num_sources
    self$enc_num_feats = enc_num_feats
    self$enc_kernel_size = enc_kernel_size
    self$enc_stride = enc_kernel_size %/% 2

    self$encoder = torch::nn_conv1d(
      in_channels = 1,
      out_channels = enc_num_feats,
      kernel_size = enc_kernel_size,
      stride = self$enc_stride,
      padding = self$enc_stride,
      bias = FALSE
    )
    self$mask_generator = model_mask_generator(
      input_dim = enc_num_feats,
      num_sources = num_sources,
      kernel_size = msk_kernel_size,
      num_feats = msk_num_feats,
      num_hidden = msk_num_hidden_feats,
      num_layers = msk_num_layers,
      num_stacks = msk_num_stacks
    )
    self$decoder = torch::nn_conv_transpose1d(
      in_channels = enc_num_feats,
      out_channels = 1,
      kernel_size = enc_kernel_size,
      stride = self$enc_stride,
      padding = self$enc_stride,
      bias = FALSE
    )
  },

  forward = function(input) {
    if(input.ndim != 3 || input$shape[2] != 1)
      value_error(glue::glue("Expected 3D tensor (batch, channel==1, frames). Found: {input$shape}"))

      # B: batch size
      # L: input frame length
      # L': padded input frame length
      # F: feature dimension
      # M: feature frame length
      # S: number of sources

      padded_num_pads = .align_num_frames_with_strides(input, self$enc_kernel_size, self$enc_stride) # B, 1, L'
      padded = padded_num_pads[[1]]
      num_pads = padded_num_pads[[2]]
      batch_size = padded$shape[1]
      num_padded_frames = padded$shape[3]
      feats = self$encoder(padded) # B, F, M
      masked = self$mask_generator(feats) * feats.unsqueeze(2) # B, S, F, M
      masked = masked$view(batch_size * self$num_sources, self$enc_num_feats, -1) # B*S, F, M
      decoded = self$decoder(masked) # B*S, 1, L'
      output = decoded$view(batch_size, self$num_sources, num_padded_frames) # B, S, L'
      if(num_pads > 0)
        output = output[.., num_pads:N] # B, S, L

      return(output)
    }
)
