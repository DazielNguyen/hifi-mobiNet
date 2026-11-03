#!/usr/bin/env bash
DEMO_STEP=setup
source "$(dirname -- "$0")/common.sh"
setup_demo() {
    "$DEMO_PYTHON" -c 'from hifimobinet.n100_demo import host_info, read, CATALOG; assert host_info()["packages"] == read(CATALOG)["expected_packages"], "Runtime versions changed"'
    "$DEMO_PYTHON" -m pip freeze > "$DEMO_LOG_DIR/before.txt"
    if "$DEMO_PYTHON" -m hifimobinet.n100_demo frontend > "$DEMO_LOG_DIR/frontend-before.txt" 2>&1; then
        printf 'Frontend hien co khop 720/720 cau; dung lai.\n'
    else
        sudo apt-get update
        sudo apt-get install --yes git build-essential cmake ninja-build pkg-config python3-dev
        CMAKE_BUILD_PARALLEL_LEVEL=2 "$DEMO_PYTHON" -m pip install \
            --constraint "$DEMO_LOG_DIR/before.txt" --constraint demo/n100_constraints.txt \
            ./vendor/banhmi-phonemize
    fi
    "$DEMO_PYTHON" -m pip check
    "$DEMO_PYTHON" -m pip freeze > "$DEMO_LOG_DIR/after.txt"
    "$DEMO_PYTHON" -m hifimobinet.n100_demo doctor --output "$DEMO_LOG_DIR/doctor.json"
    printf '\nBUOC 01 OK. Giao dien web khong can cai them framework. Tiep tuc buoc 02.\n'
}
setup_demo 2>&1 | tee "$DEMO_LOG_DIR/setup.log"
