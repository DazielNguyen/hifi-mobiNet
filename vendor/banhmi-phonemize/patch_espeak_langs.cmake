# Limits which espeak-ng dictionaries get compiled, applied as a
# PATCH_COMMAND on the downloaded espeak-ng source before it's built.
#
# espeak-ng's cmake/data.cmake unconditionally does:
#   list(APPEND _dict_compile_list af am an ar ... vi ... yue)   # all ~120
# Overriding the variable right before it's consumed (just before the
# _mbrola_lang_list block, a stable unique anchor) replaces that list with
# only the languages this fork needs, so the (slow) per-language dictionary
# compile step only runs for those.
#
# Usage: cmake -DTARGET_FILE=<path to cmake/data.cmake> -DLANGS=en,vi -P patch_espeak_langs.cmake
# LANGS is comma-separated (not a CMake list) -- see CMakeLists.txt for why.

if(NOT DEFINED TARGET_FILE)
    message(FATAL_ERROR "TARGET_FILE not set")
endif()
if(NOT DEFINED LANGS)
    set(LANGS "en")
endif()

string(REPLACE "," ";" _langs_list "${LANGS}")

file(READ "${TARGET_FILE}" _content)

string(REPLACE
    "list(APPEND _mbrola_lang_list"
    "set(_dict_compile_list ${_langs_list})\n\nlist(APPEND _mbrola_lang_list"
    _content
    "${_content}"
)

file(WRITE "${TARGET_FILE}" "${_content}")
