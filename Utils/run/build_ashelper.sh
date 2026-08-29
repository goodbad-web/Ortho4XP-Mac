#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_binary="${repo_root}/Utils/mac/ASHelper"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ASHelper can only be built on macOS." >&2
    exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ortho4xp-ashelper.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

module_cache_dir="${build_dir}/module-cache"
mkdir -p "${module_cache_dir}"

xcrun swiftc \
    -O \
    -module-cache-path "${module_cache_dir}" \
    "${repo_root}/src/ASHelper.swift" \
    -o "${build_dir}/ASHelper" \
    -framework Foundation \
    -framework CoreGraphics \
    -framework ImageIO \
    -framework UniformTypeIdentifiers \
    -framework Vision \
    -framework CoreImage \
    -framework Metal \
    -framework MetalKit

mv "${build_dir}/ASHelper" "${runtime_binary}"
echo "Built ${runtime_binary}"
