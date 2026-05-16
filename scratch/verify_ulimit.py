import sys
import os
sys.path.append(os.path.abspath("./src"))
import resource
import O4_UI_Utils

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"Current soft limit: {soft}")
print(f"Current hard limit: {hard}")

if soft >= 4096:
    print("Verification SUCCESS: ulimit has been raised.")
else:
    print("Verification FAILED: ulimit is still low.")
