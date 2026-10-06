#include <mutex>
#include <string>
#include <vector>

#include <espeak-ng/speak_lib.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phonemize.hpp"
#include "shared.hpp"

namespace py = pybind11;

namespace {

void EnsureEspeakInitialized(const std::string &dataPath) {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  std::lock_guard<std::mutex> lock(banhmi::espeak_mutex());
  if (initialized) {
    return;
  }
  if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, /*buflength=*/0,
                         dataPath.c_str(), /*options=*/0) < 0) {
    throw std::runtime_error("espeak-ng: failed to initialize from '" +
                              dataPath + "'");
  }
  initialized = true;
}

std::vector<std::string> PhonemizeEspeakRaw(std::string text,
                                             std::string voice,
                                             std::string dataPath) {
  EnsureEspeakInitialized(dataPath);
  return banhmi::phonemize_clauses(text, voice);
}

} // namespace

PYBIND11_MODULE(banhmi_phonemize_cpp, m, py::mod_gil_not_used()) {
  m.doc() = R"pbdoc(
        banhmi_phonemize_cpp: minimal espeak-ng binding for BanhmiTTS.

        Only the raw espeak-ng call lives here; Unicode normalization,
        phoneme/id mapping and codepoint-mode phonemization are implemented
        in pure Python in banhmi_phonemize/__init__.py.
    )pbdoc";

  m.def("phonemize_espeak_raw", &PhonemizeEspeakRaw, R"pbdoc(
        Phonemize text with espeak-ng, one raw (not Unicode-normalized)
        phoneme string per sentence.
    )pbdoc");
}
