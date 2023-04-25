#' Wrapper class for `model_wav2vec2`.
#'
#' This is used for layer normalization at the input
#'
#' @param model Wav2Vec2Model object. See [model_wav2vec2()]
#'
#' @keywords internal
.model_wav2vec2 <- torch::nn_module(
  ".Wav2Vec2Model",
  initialize = function(model) {
    self$model = model
  },

  forward = function(waveforms, lengths = NULL) {
    waveforms = torch::nnf_layer_norm(waveforms, waveforms$shape)
    return(self$model(waveforms, lengths))
  },

  extract_features = function(
    waveforms,
    lengths = NULL,
    num_layers = NULL
  ) {
    waveforms = torch::nnf_layer_norm(waveforms, waveforms$shape)
    return(self$model$extract_features(waveforms, lengths, num_layers))
  }
)


#' Data class that bundles associated information to use pretrained [model_wav2vec2].
#'
#'    This class provides interfaces for instantiating the pretrained model along with
#'    the information necessary to retrieve pretrained weights and additional data
#'    to be used with the model.
#'
#'    Torchaudio library instantiates objects of this class, each of which represents
#'    a different pretrained model. Client code should access pretrained models via these
#'    instances.
#'
#'    Please see below for the usage and the available values.
#'
#' @export
pipeline_wav2vec2_bundle <- R6::R6Class(
  "Wav2Vec2Bundle",

  public = list(

    .path = character(0),
    .params =  list(),
    .sample_rate = numeric(0),
    .normalize_waveform = logical(0),
    .model_type = character(0),

    initialize = function(
      path = character(0),
      params =  list(),
      sample_rate = numeric(0),
      normalize_waveform = logical(0),
      model_type = character(0)
    ) {
      self$.path = path
      self$.params = params
      self$.sample_rate = sample_rate
      self$.normalize_waveform = normalize_waveform
      self$.model_type = model_type
    },

    #' sample_rate
    #'
    #' Sample rate of the audio that the model is trained on.
    #'
    #' @return numeric
    sample_rate = function() {
      return(self$.sample_rate)
    },

    .get_state_dict = function(
      endpoint = f("models/{self$.path}"),
      path = f("{torch:::inst_path()}/hub/checkpoints")
    ) {
      file_name = basename(endpoint)
      file_path = file.path(path, file_name)
      .create_dir_if_not_exists(path)
      state_dict_path = download_asset(
        key = endpoint,
        path = file_path,
        storage_url = torchaudio_models_url()
      )

      return(torch::load_state_dict(state_dict_path))
    },

    #' Construct the model and load the pretrained weight.
    #'
    #'  The weight file is downloaded from the internet and cached with
    #'  :func:`torch.hub.load_state_dict_from_url`
    #'
    #' @param dl_kwargs  (dictionary of keyword arguments): Passed to `torch.hub.load_state_dict_from_url`.
    #'
    #' @details
    #'
    #'  For the models listed below, an additional layer normalization is
    #'  performed on the input.
    #'
    #'  For all other models, a Wav2Vec2Model instance is returned.
    #'
    #'  - WAV2VEC2_LARGE_LV60K
    #'  - WAV2VEC2_ASR_LARGE_LV60K_10M
    #'  - WAV2VEC2_ASR_LARGE_LV60K_100H
    #'  - WAV2VEC2_ASR_LARGE_LV60K_960H
    #'  - WAV2VEC2_XLSR53
    #'  - WAV2VEC2_XLSR_300M
    #'  - WAV2VEC2_XLSR_1B
    #'  - WAV2VEC2_XLSR_2B
    #'  - HUBERT_LARGE
    #'  - HUBERT_XLARGE
    #'  - HUBERT_ASR_LARGE
    #'  - HUBERT_ASR_XLARGE
    #'  - WAVLM_LARGE
    #'
    #' @returns Variation of [model_wav2vec2].
    get_model = function() {
      model = if (self$.model_type == "WavLM") model_wavlm_build else model_wav2vec2_build
      model = purrr::exec(model, !!!self$.params)
      model$load_state_dict(self$.get_state_dict())

      if (self$.normalize_waveform)
        model = .model_wav2vec2(model)

      model$eval()
      return(model)
    }
  )
)



#' Data class that bundles associated information to use pretrained `model_wav2vec2`
#'
#'    This class provides interfaces for instantiating the pretrained model along with
#'    the information necessary to retrieve pretrained weights and additional data
#'    to be used with the model.
#'
#'    Torchaudio library instantiates objects of this class, each of which represents
#'    a different pretrained model. Client code should access pretrained models via these
#'    instances.
#'
#'    Please see below for the usage and the available values.
#'
#' @examples - ASR
#' >>> import torchaudio
#' >>>
#' >>> bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
#' >>>
#' >>> # Build the model and load pretrained weight.
#' >>> model = bundle.get_model ()
#' Downloading:
#' 100%|███████████████████████████████| 1.18G/1.18G [00:17<00:00, 73.8MB/s]
#' >>>
#' >>> # Check the corresponding labels of the output.
#' >>> labels = bundle.get_labels ()
#' >>> print (labels)
#'   ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
#' >>>
#' >>> # Resample audio to the expected sampling rate
#' >>> waveform = torchaudio.functional.resample (waveform, sample_rate, bundle.sample_rate)
#' >>>
#' >>> # Infer the label probability distribution
#' >>> emissions, _ = model (waveform)
#' >>>
#' >>> # Pass emission to decoder
#' >>> # `ctc_decode` is for illustration purpose only
#' >>> transcripts = ctc_decode (emissions, labels)
#'
#' @export
pipelines_wav2vec2_asr_bundle <- R6::R6Class(
  "Wav2Vec2ASRBundle",
  inherit = pipeline_wav2vec2_bundle,
  public = list(
    .labels = character(0),
    .remove_aux_axis = c(2, 3, 4),

    initialize = function(
      path = character(0),
      params =  list(),
      sample_rate = numeric(0),
      normalize_waveform = logical(0),
      model_type = character(0),
      labels = character(0)
    ) {
      self$.path = path
      self$.params = params
      self$.sample_rate = sample_rate
      self$.normalize_waveform = normalize_waveform
      self$.model_type = model_type
      self$.labels = labels
    },

    #' The output class labels  (only applicable to fine-tuned bundles)
    #' The first is blank token, and it is customizable.
    #' @param blank  (str, optional): Blank token. (default: `'-'`)
    #' @returns character vector
    #'  For models fine-tuned on ASR, returns the vector of strings representing
    #'  the output class labels.
    get_labels = function(blank = "-") {
      unique(c(blank, self$.labels))
    },

    #' Remove the seemingly unnecessary axis
    #'
    #' @note
    #' For ASR task, the pretrained weights originated from fairseq has unrelated
    #' dimensions at index 2, 3, 4.
    #' It's originated from the Dictionary implementation of fairseq, which was
    #' intended for NLP tasks, but not used during the ASR training.
    #' https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/data/dictionary.py#L21-L37
    #' https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/criterions/ctc.py#L126-L129
    #'
    #' Also, some pretrained weights originated from voxpopuli has an extra
    #' dimensions that almost never used and that resembles mistake.
    #' The label `1` shows up in the training dataset of German (1 out of 16M),
    #' English (1 / 28M), Spanish (1 / 9.4M), Romanian (1 / 4.7M) and Polish (6 / 5.8M)
    .get_state_dict = function() {
      state_dict = super$.get_state_dict()
      if(!is.null(self$.remove_aux_axis)) {
        for(key in c("aux.weight", "aux.bias")) {
          t = state_dict[[key]]
          state_dict[[key]] = t[seq.int(t$size(1)) %not_in% self$.remove_aux_axis, ..]
        }
      }
      return(state_dict)
    }
  )
)

#' Wav2vec 2.0 model ("base" architecture with an extra linear module),
#' pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
#' (the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
#' fine-tuned for ASR on the same audio with the corresponding transcripts.
#' Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
#' redistributed with the same license.
#' [`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
#'   `Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
#' Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
#' @export
pipeline_wav2vec2_asr_base_960h <- function() {
  pipelines_wav2vec2_asr_bundle$new(
    path = "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
    params = list(
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
    ),
    labels = .get_en_labels(),
    sample_rate = 16000,
    normalize_waveform = FALSE,
    model_type = "Wav2Vec2"
  )
}
