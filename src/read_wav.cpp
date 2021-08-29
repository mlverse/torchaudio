#include <Rcpp.h>
#include "AudioFile.h"
using namespace Rcpp;


// [[Rcpp::export]]
Rcpp::List audiofile_read_wav_cpp (std::string filepath, int from, int to, std::string unit) {
  AudioFile<float> wav;
  wav.load(filepath.c_str());


  return Rcpp::List::create(
    _["waveform"] = wav.samples,
    _["sample_rate"] = wav.getSampleRate(),
    _["samples"] = wav.getNumSamplesPerChannel(),
    _["channels"] = wav.getNumChannels(),
    _["bit"] = wav.getBitDepth()
  );
}
