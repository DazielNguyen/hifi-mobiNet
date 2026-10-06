#ifndef BANHMI_PHONEMIZE_H_
#define BANHMI_PHONEMIZE_H_

#include <string>
#include <vector>

#include "shared.hpp"

namespace banhmi {

// Phonemizes text with espeak-ng, returning one raw phoneme string per
// detected sentence (punctuation reconstructed, language-switch markers
// stripped, but *not* Unicode-normalized or split into codepoints -- the
// Python layer does that with the standard library, so this module doesn't
// need to link a Unicode library of its own).
//
// Assumes espeak_Initialize() has already been called for `dataPath`.
BANHMI_EXPORT std::vector<std::string>
phonemize_clauses(const std::string &text, const std::string &voice);

} // namespace banhmi

#endif // BANHMI_PHONEMIZE_H_
