# torchaudio (development version)

### Breaking changes

#### Streamline audio loading (#62)
Thanks to superior performance as well as versatility, the default backend for loading audio files is now [`av`](https://docs.ropensci.org/av/). `av` is an efficient wrapper for [Ffmpeg](https://ffmpeg.org/).

The refactorings involved in this update contain breaking changes as to naming and scope. Most mportantly:

- `av` is now the default backend, and it is a mandatory dependency. Linux users please consider the [av installation instructions](https://docs.ropensci.org/av/).

- The only user-visible function to load audio is `torchaudio_load()`. It will delegate to the default backend, or one you set with `set_audio_backend()`.

- As of this time, a supported alternative backend is `tuneR`.

- The only user-visible function to obtain audio file information is now `torchaudio_info()`. 


### New features


### Bug fixes


### Internal


# torchaudio 0.2.2
* fix missing inclusion of `cstdint` in `Audiofile.h`

# torchaudio 0.2.1
* maintenance release

# torchaudio 0.2.0
* `torchaudio_load()` has been replaced by `tuneR_loader()` and `transform_to_tensor()`. Inspired by {torchvision}.
* `av_loader()` (experimental)

# torchaudio 0.1.1.0
* Added a `NEWS.md` file to track changes to the package.
* Depends on {torch} >= 0.2.0
