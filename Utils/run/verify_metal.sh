#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir}/../.."

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Metal verification requires macOS." >&2
    exit 2
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ortho4xp-metal-probe.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
mkdir -p "${build_dir}/module-cache"

xcrun swiftc \
    -O \
    -module-cache-path "${build_dir}/module-cache" \
    "${script_dir}/MetalProbe.swift" \
    -o "${build_dir}/MetalProbe" \
    -framework Foundation \
    -framework CoreGraphics \
    -framework CoreImage \
    -framework Metal

python_bin="${ORTHO4XP_PYTHON:-}"
if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
    for candidate in \
        "${repo_root}/.venv-test/bin/python" \
        "${repo_root}/.venv/bin/python" \
        "$(command -v python3 || true)"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            python_bin="${candidate}"
            break
        fi
    done
fi

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
    echo "Python 3 with Pillow was not found. Set ORTHO4XP_PYTHON." >&2
    exit 1
fi

exec "${python_bin}" "${script_dir}/verify_metal.py" \
    --probe "${build_dir}/MetalProbe" \
    --helper "${repo_root}/Utils/mac/ASHelper" \
    "$@"
