#!/bin/zsh

# Blender's macOS Metal backend can crash before Python starts when the Mac is
# locked or no display/GPU context is available.  Probe the same Metal API
# first so callers get an actionable error instead of a SIGSEGV.
set -euo pipefail

blender_bin="${OR4XP_BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
if [[ ! -x "$blender_bin" ]]; then
  print -u2 "Blender executable not found: $blender_bin"
  print -u2 "Set OR4XP_BLENDER_BIN to the Blender binary to use."
  exit 127
fi

probe_root="${TMPDIR:-/tmp}/ortho4xp-blender-metal-probe"
mkdir -p "$probe_root/clang" "$probe_root/swift"

probe_output=""
probe_status=0
if probe_output="$(
  env \
    CLANG_MODULE_CACHE_PATH="$probe_root/clang" \
    SWIFT_MODULECACHE_OVERRIDE="$probe_root/swift" \
    xcrun swift -e '
      import Foundation
      import Metal
      if MTLCreateSystemDefaultDevice() == nil {
          print("NO_METAL_DEVICE")
          exit(78)
      }
      print("METAL_DEVICE_OK")
    ' 2>&1
)"; then
  :
else
  probe_status=$?
fi

if (( probe_status != 0 )); then
  print -u2 "Metal device is unavailable; Blender was not started."
  print -u2 "Unlock the Mac and run this command again while a GUI session is active."
  if [[ -n "$probe_output" ]]; then
    print -u2 -- "$probe_output"
  fi
  exit 78
fi

exec "$blender_bin" "$@"
