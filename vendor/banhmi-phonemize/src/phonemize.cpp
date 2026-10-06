#include "phonemize.hpp"

#include <stdexcept>

#include <espeak-ng/speak_lib.h>

#include "clause_terminator.hpp"

namespace banhmi {
namespace {

// espeak-ng surrounds words from a language other than the current voice
// with "(xx ... )" markers in its IPA output. Parens are ASCII, and ASCII
// bytes never occur inside a multi-byte UTF-8 sequence, so a plain byte scan
// is enough here -- no need to decode the string as Unicode first.
std::string StripLanguageSwitchMarkers(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  bool inMarker = false;
  for (char c : text) {
    if (inMarker) {
      if (c == ')') {
        inMarker = false;
      }
    } else if (c == '(') {
      inMarker = true;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// espeak-ng doesn't include the punctuation character that ended a clause
// in its phoneme output -- only a terminator *code* describing it -- so it
// has to be added back in by hand to keep it in the phoneme stream.
void AppendTerminatorPunctuation(std::string &clause, int terminator) {
  switch (terminator & kPunctuationMask) {
  case kClausePeriod:
    clause += '.';
    break;
  case kClauseQuestion:
    clause += '?';
    break;
  case kClauseExclamation:
    clause += '!';
    break;
  case kClauseComma:
    clause += ", ";
    break;
  case kClauseColon:
    clause += ": ";
    break;
  case kClauseSemicolon:
    clause += "; ";
    break;
  default:
    break;
  }
}

// espeak_SetVoiceByName does non-trivial work (loading a voice's dictionary
// data). Every one of our callers phonemizes a whole dataset with a single
// fixed voice, so skipping the call when the voice hasn't changed avoids
// redoing that work on every single utterance.
std::string &LastVoice() {
  static std::string voice;
  return voice;
}

void EnsureVoiceSelected(const std::string &voice) {
  if (LastVoice() == voice) {
    return;
  }
  if (espeak_SetVoiceByName(voice.c_str()) != EE_OK) {
    throw std::runtime_error("espeak-ng: unknown voice '" + voice + "'");
  }
  LastVoice() = voice;
}

} // namespace

std::vector<std::string> phonemize_clauses(const std::string &text,
                                            const std::string &voice) {
  std::lock_guard<std::mutex> lock(espeak_mutex());

  EnsureVoiceSelected(voice);

  std::vector<std::string> sentences;
  std::string current;

  // espeak_TextToPhonemesWithTerminator advances `cursor` itself and sets it
  // to null once `text` has been fully consumed.
  const void *cursor = text.c_str();
  while (cursor != nullptr) {
    int terminator = 0;
    const char *clausePhonemes = espeak_TextToPhonemesWithTerminator(
        &cursor, espeakCHARS_AUTO, /*phonememode=IPA*/ 0x02, &terminator);

    if (clausePhonemes != nullptr) {
      current += StripLanguageSwitchMarkers(clausePhonemes);
    }
    AppendTerminatorPunctuation(current, terminator);

    if ((terminator & kClauseTypeSentence) == kClauseTypeSentence) {
      sentences.push_back(std::move(current));
      current.clear();
    }
  }

  if (!current.empty()) {
    sentences.push_back(std::move(current));
  }

  return sentences;
}

} // namespace banhmi
