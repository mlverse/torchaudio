#' ResBlock
#'
#' ResNet block based on "Deep Residual Learning for Image Recognition".
#' Pass the input through the ResBlock layer. The paper link is [https://arxiv.org/pdf/1512.03385.pdf]().
#'
#' @param specgram  (Tensor): the input sequence to the ResBlock layer (n_batch, n_freq, n_time).
#' @param n_freq: the number of bins in a spectrogram.  (Default: ``128``)
#'
#' @return
#' Tensor shape:  (n_batch, n_freq, n_time)
#'
#' @examples
#'  resblock = ResBlock ()
#'  input = torch$rand (10, 128, 512)  # a random spectrogram
#'  output = resblock (input)  # shape: (10, 128, 512)
#'
#' @export
model_resblock <- torch::nn_module(
  "ResBlock",
  initialize = function(n_freq = 128) {
    self$resblock_model = torch::nn_sequential(
      torch::nn_conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=FALSE),
      torch::nn_batch_norm1d(n_freq),
      torch::nn_relu(inplace=TRUE),
      torch::nn_conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=FALSE),
      torch::nn_batch_norm1d(n_freq)
    )
  },

  forward = function(specgram) {
    return(self$resblock_model(specgram) + specgram)
  }
)

#' MelResNet
#'
#' MelResNet layer uses a stack of ResBlocks on spectrogram.
#' Pass the input through the MelResNet layer.
#'
#' @param specgram  (Tensor): the input sequence to the MelResNet layer (n_batch, n_freq, n_time).
#' @param n_res_block: the number of ResBlock in stack.  (Default: ``10``)
#' @param n_freq: the number of bins in a spectrogram.  (Default: ``128``)
#' @param n_hidden: the number of hidden dimensions of resblock.  (Default: ``128``)
#' @param n_output: the number of output dimensions of melresnet.  (Default: ``128``)
#' @param kernel_size: the number of kernel size in the first Conv1d layer.  (Default: ``5``)
#'
#' @return
#' Tensor shape:  (n_batch, n_output, n_time - kernel_size + 1)
#'
#' @examples
#'  melresnet = model_melresnet()
#'  input = torch::torch_rand(10, 128, 512)  # a random spectrogram
#'  output = melresnet(input)  # shape: (10, 128, 508)
#'
#' @export
model_melresnet <- torch::nn_module(
  "MelResNet",
  initialize = function(
    n_res_block = 10,
    n_freq = 128,
    n_hidden = 128,
    n_output = 128,
    kernel_size = 5
  ) {

    ResBlocks = replicate(n_res_block, model_resblock(n_hidden))

    self$melresnet_model = torch::nn_sequential(
      torch::nn_conv1d(in_channels=n_freq, out_channels=n_hidden, kernel_size=kernel_size, bias=FALSE),
      torch::nn_batch_norm1d(n_hidden),
      torch::nn_relu(inplace=TRUE),
      !!!(ResBlocks),
      torch::nn_conv1d(in_channels=n_hidden, out_channels=n_output, kernel_size=1)
    )
  },

  forward = function(specgram) {
    return(self$melresnet_model(specgram))
  }
)

#' Stretch2d
#'
#' Upscale the frequency and time dimensions of a spectrogram.
#' Pass the input through the Stretch2d layer.
#'
#' @param specgram  (Tensor): the input sequence to the Stretch2d layer (..., n_freq, n_time).
#' @param time_scale: the scale factor in time dimension
#' @param freq_scale: the scale factor in frequency dimension
#'
#' @return
#' Tensor shape:  (..., n_freq * freq_scale, n_time * time_scale)
#'
#' @examples
#'  stretch2d = model_stretch2d(time_scale=10, freq_scale=5)
#'
#'  input = torch::torch_rand(10, 100, 512)  # a random spectrogram
#'  output = stretch2d(input)  # shape: (10, 500, 5120)
#'
#' @export
model_stretch2d <- torch::nn_module(
  "Stretch2d",
  initialize = function(
    time_scale,
    freq_scale
  ) {
    self$freq_scale = as.integer(freq_scale)
    self$time_scale = as.integer(time_scale)
  },
  forward = function(specgram) {
    return(specgram$repeat_interleave(self$freq_scale, -2)$repeat_interleave(self$time_scale, -1))
  }
)

#' UpsampleNetwork
#'
#' Upscale the dimensions of a spectrogram.
#' Pass the input through the UpsampleNetwork layer.
#'
#' @param specgram  (Tensor): the input sequence to the UpsampleNetwork layer (n_batch, n_freq, n_time)
#' @param upsample_scales: the list of upsample scales.
#' @param n_res_block: the number of ResBlock in stack.  (Default: ``10``)
#' @param n_freq: the number of bins in a spectrogram.  (Default: ``128``)
#' @param n_hidden: the number of hidden dimensions of resblock.  (Default: ``128``)
#' @param n_output: the number of output dimensions of melresnet.  (Default: ``128``)
#' @param kernel_size: the number of kernel size in the first Conv1d layer.  (Default: ``5``)
#'
#' @return
#'  Tensor shape:  (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale),
#'  (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
#'  where total_scale is the product of all elements in upsample_scales.
#'
#' @examples
#'  upsamplenetwork = model_upsample_network(upsample_scales=c(4, 4, 16))
#'  input = torch::torch_rand (10, 128, 10)  # a random spectrogram
#'  output = upsamplenetwork (input)  # shape: (10, 1536, 128), (10, 1536, 128)
#'
#' @export
model_upsample_network <- torch::nn_module(
  "UpsampleNetwork",
  initialize = function(
    upsample_scales,
    n_res_block = 10,
    n_freq = 128,
    n_hidden = 128,
    n_output = 128,
    kernel_size = 5
  ) {

    total_scale = prod(upsample_scales)
    self$indent = ((kernel_size - 1) %/% 2) * total_scale
    self$resnet = model_melresnet(n_res_block, n_freq, n_hidden, n_output, kernel_size)
    self$resnet_stretch = model_stretch2d(total_scale, 1)

    up_layers = list()
    for(scale in upsample_scales) {
      stretch = model_stretch2d(scale, 1)
      conv = torch::nn_conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=list(1, scale * 2 + 1),
                              padding=list(0, scale),
                              bias=FALSE)
      conv$parameters$weight$data()$fill_(1. / (scale * 2 + 1))
      up_layers[[length(up_layers) + 1]] <- stretch
      up_layers[[length(up_layers) + 1]] <- conv
    }
    self$upsample_layers = torch::nn_sequential(!!!up_layers)
  },

  forward = function(specgram) {
    resnet_output = self$resnet(specgram)$unsqueeze(2)
    resnet_output = self$resnet_stretch(resnet_output)
    resnet_output = resnet_output$squeeze(2)

    specgram = specgram$unsqueeze(2)
    upsampling_output = self$upsample_layers(specgram)
    upsampling_output_size = upsampling_output$size()
    lu = length(upsampling_output_size)
    upsampling_output = upsampling_output$squeeze(2)[ ,  , (self$indent+1):(upsampling_output_size[lu]-self$indent)]

    return(list(upsampling_output, resnet_output))
  }
)



#' WaveRNN
#'
#' WaveRNN model based on the implementation from [fatchord](https://github.com/fatchord/WaveRNN).
#' The original implementation was introduced in ["Efficient Neural Audio Synthesis"](https://arxiv.org/pdf/1802.08435.pdf).
#'#' Pass the input through the WaveRNN model.
#'
#' @param waveform: the input waveform to the WaveRNN layer  (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
#' @param specgram: the input spectrogram to the WaveRNN layer  (n_batch, 1, n_freq, n_time)
#' @param upsample_scales: the list of upsample scales.
#' @param n_classes: the number of output classes.
#' @param hop_length: the number of samples between the starts of consecutive frames.
#' @param n_res_block: the number of ResBlock in stack.  (Default: ``10``)
#' @param n_rnn: the dimension of RNN layer.  (Default: ``512``)
#' @param n_fc: the dimension of fully connected layer.  (Default: ``512``)
#' @param kernel_size: the number of kernel size in the first Conv1d layer.  (Default: ``5``)
#' @param n_freq: the number of bins in a spectrogram.  (Default: ``128``)
#' @param n_hidden: the number of hidden dimensions of resblock.  (Default: ``128``)
#' @param n_output: the number of output dimensions of melresnet.  (Default: ``128``)
#'
#' @details The input channels of waveform and spectrogram have to be 1. The product of
#'    `upsample_scales` must equal `hop_length`.
#'
#' @return
#' Tensor shape:  (n_batch, 1, (n_time - kernel_size + 1) * hop_length, n_classes)
#'
#' @examples
#'  wavernn = model_wavernn(upsample_scales=c(5,5,8), n_classes=512, hop_length=200)
#'  waveform, sample_rate = torchaudio::torchaudio_load (file)
#'  # waveform shape:  (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
#'  specgram = MelSpectrogram (sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
#'  output = wavernn (waveform, specgram)
#'  # output shape:  (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
#'
#' @export
model_wavernn <- torch::nn_module(
  "WaveRNN",
  initialize = function(
    upsample_scales,
    n_classes,
    hop_length,
    n_res_block = 10,
    n_rnn = 512,
    n_fc = 512,
    kernel_size = 5,
    n_freq = 128,
    n_hidden = 128,
    n_output = 128
    ) {

    self$kernel_size = kernel_size
    self$n_rnn = n_rnn
    self$n_aux = n_output %/% 4
    self$hop_length = hop_length
    self$n_classes = n_classes

    total_scale = prod(upsample_scales)
    if(total_scale != self$hop_length)
      value_error(glue::glue("Expected: total_scale == hop_length, but found {total_scale} != {hop_length}"))

    self$upsample = model_upsample_network(
      upsample_scales,
      n_res_block,
      n_freq,
      n_hidden,
      n_output,
      kernel_size
    )
    self$fc = torch::nn_linear(n_freq + self$n_aux + 1, n_rnn)

    self$rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=TRUE)
    self$rnn2 = nn.GRU(n_rnn + self$n_aux, n_rnn, batch_first=TRUE)

    self$relu1 = torch::nn_relu(inplace=TRUE)
    self$relu2 = torch::nn_relu(inplace=TRUE)

    self$fc1 = torch::nn_linear(n_rnn + self$n_aux, n_fc)
    self$fc2 = torch::nn_linear(n_fc + self$n_aux, n_fc)
    self$fc3 = torch::nn_linear(n_fc, self$n_classes)
  },


  forward = function(waveform, specgram) {
    waveform = torch::torch_rand(4, 1, (100 - kernel_size + 1) * hop_length)
    specgram = torch::torch_rand(4, 1, n_freq, 100)

    if(waveform$size(2) != 1) value_error('Require the input channel of waveform is 1')
    if(specgram$size(2) != 1) value_error('Require the input channel of specgram is 1')

    # remove channel dimension until the end
    waveform = waveform$squeeze(2)
    specgram = specgram$squeeze(2)
    batch_size = waveform$size(1)
    h1 = torch::torch_zeros(1, batch_size, self$n_rnn, dtype=waveform$dtype, device=waveform$device)
    h2 = torch::torch_zeros(1, batch_size, self$n_rnn, dtype=waveform$dtype, device=waveform$device)
    # output of upsample:
    # specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
    # aux: (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
    specgram_and_aux = self$upsample(specgram)
    specgram = specgram_and_aux[[1]]$transpose(2, 3)
    aux = specgram_and_aux[[2]]$transpose(2, 3)

    aux_idx = (self$n_aux*(0:4))
    a1 = aux[ ,  , (aux_idx[0+1] +1):aux_idx[1+1]]
    a2 = aux[ ,  , (aux_idx[1+1] +1):aux_idx[2+1]]
    a3 = aux[ ,  , (aux_idx[2+1] +1):aux_idx[3+1]]
    a4 = aux[ ,  , (aux_idx[3+1] +1):aux_idx[4+1]]

    x = torch::torch_cat(list(waveform$unsqueeze(-1), specgram[ , , ], a1), dim=-1L)
    x = self$fc(x)
    res = x
    # x, _ = self$rnn1(x, h1)

    # x = x + res
    # res = x
    # x = torch::torch_cat([x, a2], dim=-1)
    # x, _ = self$rnn2(x, h2)
    #
    # x = x + res
    # x = torch::torch_cat([x, a3], dim=-1)
    # x = self$fc1(x)
    # x = self$relu1(x)
    #
    # x = torch::torch_cat([x, a4], dim=-1)
    # x = self$fc2(x)
    # x = self$relu2(x)
    # x = self$fc3(x)

    # bring back channel dimension
    return(x$unsqueeze(2))
  }
)
