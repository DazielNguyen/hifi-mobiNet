#ifndef BANHMI_CLAUSE_TERMINATOR_H_
#define BANHMI_CLAUSE_TERMINATOR_H_

// espeak-ng's espeak_TextToPhonemesWithTerminator() (a function added by
// https://github.com/rhasspy/espeak-ng, not present in mainline espeak-ng)
// returns an integer terminator code per clause. Its bit layout is defined
// internally by espeak-ng in src/libespeak-ng/translate.h, which isn't part
// of the public speak_lib.h header, so the relevant bits are reproduced here
// -- this is espeak-ng's own wire format, not anything specific to this
// project's phoneme-mapping logic.

namespace banhmi {

constexpr int kClauseIntonationFullStop = 0x00000000;
constexpr int kClauseIntonationComma = 0x00001000;
constexpr int kClauseIntonationQuestion = 0x00002000;
constexpr int kClauseIntonationExclamation = 0x00003000;

constexpr int kClauseTypeClause = 0x00040000;
constexpr int kClauseTypeSentence = 0x00080000;

constexpr int kClausePeriod = 40 | kClauseIntonationFullStop | kClauseTypeSentence;
constexpr int kClauseComma = 20 | kClauseIntonationComma | kClauseTypeClause;
constexpr int kClauseQuestion = 40 | kClauseIntonationQuestion | kClauseTypeSentence;
constexpr int kClauseExclamation = 45 | kClauseIntonationExclamation | kClauseTypeSentence;
constexpr int kClauseColon = 30 | kClauseIntonationFullStop | kClauseTypeClause;
constexpr int kClauseSemicolon = 30 | kClauseIntonationComma | kClauseTypeClause;

// Low 20 bits carry the punctuation/pause code; everything above that is
// flags (clause vs. sentence boundary, voice change, etc).
constexpr int kPunctuationMask = 0x000FFFFF;

} // namespace banhmi

#endif // BANHMI_CLAUSE_TERMINATOR_H_
