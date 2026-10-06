#!/usr/bin/env bash
# Build the two Monotonic Alignment Search Cython extensions used by training.
#
#   scripts/training/build_mas.sh /path/to/venv/bin/python
#
# - Internal baseline: vendor/banhmi/.../core.pyx -> src/hifimobinet/architecture/vits/utils/monotonic_align/
#   (the relocated wrapper imports `.core`).
# - Piper: vendor/piper/vits/monotonic_align/core.pyx -> vendor/piper/vits/monotonic_align/monotonic_align/
#   (upstream's wrapper imports `.monotonic_align.core`; same layout as upstream build_monotonic_align.sh).
#
# Sources are compiled in a temporary directory, so no generated .c file is
# written into the checkout. Only the compiled modules land in ignored paths.
set -euo pipefail

PY="${1:?usage: build_mas.sh /path/to/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

build_one() {
    local pyx="$1" dest="$2" tag="$3"
    mkdir -p "$BUILD/$tag" "$dest"
    cp "$pyx" "$BUILD/$tag/core.pyx"
    (cd "$BUILD/$tag" && "$PY" -m cython -3 core.pyx -o core.c >/dev/null && "$PY" - <<'EOF'
import numpy, sysconfig
from setuptools import Extension, setup
setup(name="core", script_args=["build_ext", "--inplace", "-q"],
      ext_modules=[Extension("core", ["core.c"], include_dirs=[numpy.get_include()])])
EOF
    )
    rm -f "$dest"/core*.so
    cp "$BUILD/$tag"/core*.so "$dest/"
    echo "built $tag -> $dest/$(cd "$dest" && ls core*.so)"
}

build_one "$REPO/vendor/banhmi/vits/utils/monotonic_align/core.pyx" \
          "$REPO/src/hifimobinet/architecture/vits/utils/monotonic_align" banhmi
build_one "$REPO/vendor/piper/vits/monotonic_align/core.pyx" \
          "$REPO/vendor/piper/vits/monotonic_align/monotonic_align" piper
