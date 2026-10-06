cmake_minimum_required(VERSION 3.14)

set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (SKBUILD)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_BUILD_WITN_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

project(
    banhmi_phonemize
    VERSION 1.0.0
    DESCRIPTION "espeak-ng phonemization library for BanhmiTTS -- see NOTICE.md"
    HOMEPAGE_URL "https://github.com/rhasspy/piper-phonemize"
    LANGUAGES CXX
)

if(MSVC)
    # Force compiler to use UTF-8 for IPA constants
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

elseif(NOT APPLE)
    # Linux flags
    string(APPEND CMAKE_CXX_FLAGS "-D_FILE_OFFSET_BITS=64 -Wall -Wextra ")
    string(APPEND CMAKE_C_FLAGS "-D_FILE_OFFSET_BITS=64 -Wall -Wextra ")
endif()

if (APPLE)
    set(RPATH_PREFIX "@loader_path")
else()
    set(RPATH_PREFIX "\$ORIGIN")
endif()

add_library(
    banhmi_phonemize SHARED
    src/phonemize.cpp
    src/shared.cpp
)

include(GNUInstallDirs)

if(SKBUILD)
    set(BANHMI_PHONEMIZE_LIBDIR "banhmi_phonemize/lib")
    set(CMAKE_INSTALL_LIBDIR "banhmi_phonemize/lib")
    set(CMAKE_INSTALL_BINDIR "banhmi_phonemize/lib")

else()
    set(BANHMI_PHONEMIZE_LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

set_target_properties(banhmi_phonemize PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

set_target_properties(
    banhmi_phonemize PROPERTIES
    INSTALL_RPATH "${RPATH_PREFIX}"
)

if (SKBUILD)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG)

    pybind11_add_module(banhmi_phonemize_cpp
        src/python.cpp
    )
endif()

# ---- espeak-ng ---
#
# Compiled from source (a pinned commit of rhasspy/espeak-ng, the fork Piper
# depends on for espeak_TextToPhonemesWithTerminator). The dictionary list is
# patched down to just the languages this fork needs -- see
# patch_espeak_langs.cmake -- since espeak-ng otherwise compiles ~120
# per-language dictionaries every build.

if(NOT DEFINED ESPEAK_NG_LANGS)
    # Comma-separated (not CMake-list/semicolon-separated) so it survives
    # being passed as a single PATCH_COMMAND argument intact -- ExternalProject
    # re-splits semicolons before the child `cmake -P` ever sees them, even
    # when quoted or wrapped in $<JOIN:...,$<SEMICOLON>>.
    set(ESPEAK_NG_LANGS "en,vi")
endif()

if(NOT DEFINED ESPEAK_NG_DIR)
    set(ESPEAK_NG_DIR "${CMAKE_CURRENT_BINARY_DIR}/ei")

        include(ExternalProject)
        ExternalProject_Add(
            espeak_ng_external
            PREFIX "${CMAKE_CURRENT_BINARY_DIR}/e"
            URL "https://github.com/rhasspy/espeak-ng/archive/0f65aa301e0d6bae5e172cc74197d32a6182200f.zip"
            PATCH_COMMAND ${CMAKE_COMMAND}
                -DTARGET_FILE=<SOURCE_DIR>/cmake/data.cmake
                "-DLANGS=${ESPEAK_NG_LANGS}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/patch_espeak_langs.cmake
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ESPEAK_NG_DIR}
            CMAKE_ARGS -DUSE_ASYNC:BOOL=OFF
            CMAKE_ARGS -DBUILD_SHARED_LIBS:BOOL=ON
            CMAKE_ARGS -DUSE_MBROLA:BOOL=OFF
            CMAKE_ARGS -DUSE_LIBSONIC:BOOL=OFF
            CMAKE_ARGS -DUSE_LIBPCAUDIO:BOOL=OFF
            CMAKE_ARGS -DUSE_KLATT:BOOL=OFF
            CMAKE_ARGS -DUSE_SPEECHPLAYER:BOOL=OFF
            CMAKE_ARGS -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
            CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        )
        add_dependencies(banhmi_phonemize espeak_ng_external)
        if (SKBUILD)
            add_dependencies(banhmi_phonemize_cpp espeak_ng_external)
        endif()

    set(EXTERNAL_ESPEAK TRUE)
endif()


# ---- Declare library ----

target_include_directories(
    banhmi_phonemize PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    ${ESPEAK_NG_DIR}/include
)

target_link_directories(
    banhmi_phonemize PUBLIC
    ${ESPEAK_NG_DIR}/lib
)

target_link_libraries(
    banhmi_phonemize
    espeak-ng
)

if (SKBUILD)
    target_include_directories(
        banhmi_phonemize_cpp PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
        ${ESPEAK_NG_DIR}/include
    )

    target_link_directories(
        banhmi_phonemize_cpp PUBLIC
        ${ESPEAK_NG_DIR}/lib
    )
    set_target_properties(banhmi_phonemize_cpp PROPERTIES
        INSTALL_RPATH "${RPATH_PREFIX}/${BANHMI_PHONEMIZE_LIBDIR}")

    target_link_libraries(
        banhmi_phonemize_cpp PUBLIC
        espeak-ng
        banhmi_phonemize)

    install(TARGETS banhmi_phonemize_cpp DESTINATION .)

endif()


target_compile_features(banhmi_phonemize PUBLIC cxx_std_17)

# ---- Declare install targets ----
# NOTE: only the Python module (banhmi_phonemize_cpp, built via SKBUILD/pip)
# is built here -- no standalone CLI executable/test targets.

install(
    TARGETS banhmi_phonemize
    RUNTIME LIBRARY DESTINATION ${BANHMI_PHONEMIZE_LIBDIR})


if (SKBUILD)
    install(
        FILES "${CMAKE_SOURCE_DIR}/banhmi_phonemize/__init__.py"
        DESTINATION "banhmi_phonemize"
    )
    install(
        DIRECTORY ${ESPEAK_NG_DIR}/share/
        DESTINATION "banhmi_phonemize")

    # GPL-3.0 (espeak-ng, linked into this package) and MIT (piper_phonemize,
    # whose output this package stays compatible with -- see NOTICE.md)
    # license texts, so a distributed wheel carries proper
    # attribution/compliance, not just the source repo's README.
    install(
        FILES "${CMAKE_SOURCE_DIR}/LICENSE.md"
        DESTINATION "banhmi_phonemize"
        RENAME "LICENSE-MIT-piper_phonemize.txt"
    )
    install(
        FILES "${CMAKE_SOURCE_DIR}/licenses/espeak-ng.GPL-3.0.txt"
        DESTINATION "banhmi_phonemize"
    )
    install(
        FILES "${CMAKE_SOURCE_DIR}/NOTICE.md"
        DESTINATION "banhmi_phonemize"
    )

endif()

install(
    DIRECTORY ${ESPEAK_NG_DIR}/lib/
    DESTINATION ${BANHMI_PHONEMIZE_LIBDIR})
if (WIN32)
    install(
        DIRECTORY ${ESPEAK_NG_DIR}/bin/
        DESTINATION ${BANHMI_PHONEMIZE_LIBDIR})
endif ()
