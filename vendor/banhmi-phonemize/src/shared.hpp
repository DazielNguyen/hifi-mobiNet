#ifndef BANHMI_SHARED_H_
#define BANHMI_SHARED_H_

#include <mutex>

#ifdef _WIN32
#define BANHMI_EXPORT __declspec(dllexport)
#else
#define BANHMI_EXPORT
#endif

namespace banhmi {

// espeak-ng keeps global, non-reentrant state, so every call into it must
// be serialized -- relevant once banhmi_phonemize_cpp is built with
// py::mod_gil_not_used() and could be entered from multiple Python threads
// at once.
BANHMI_EXPORT std::mutex &espeak_mutex();

} // namespace banhmi

#endif // BANHMI_SHARED_H_
