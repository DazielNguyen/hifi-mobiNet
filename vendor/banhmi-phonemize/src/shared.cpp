#include "shared.hpp"

namespace banhmi {

std::mutex &espeak_mutex() {
  static std::mutex m;
  return m;
}

} // namespace banhmi
