#include <Rcpp.h>
#define MINIMP3_IMPLEMENTATION
extern "C"{
  #include "minimp3.h"
  #include "minimp3_ex.h"
}
using namespace Rcpp;


// [[Rcpp::export]]
Rcpp::List get_info_mp3(std::string filepath) {
  mp3dec_ex_t dec;
  if (mp3dec_ex_open(&dec, filepath.c_str(), MP3D_SEEK_TO_SAMPLE)) {
    Rcpp::stop("Error reading mp3 file.");
  }

  return Rcpp::List::create(
    _["hz"] = dec.info.hz,
    _["channels"] = dec.info.channels,
    _["samples"] = dec.samples,
    _["layer"] = dec.info.layer
  );
}

// read_mp3(file, from, to, unit)
