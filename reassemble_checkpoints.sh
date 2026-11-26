#!/usr/bin/env bash
# Rebuild the model checkpoints that are stored as <100 MB parts (GitHub's per-file limit)
# and check them against CHECKPOINTS.sha256.
#
#   bash reassemble_checkpoints.sh
#
# Each checkpoint lands next to its parts, in the run folder the code loads it from.
# The parts can be deleted afterwards (they stay in git history either way).
set -euo pipefail
cd "$(dirname "$0")"

while IFS= read -r first; do
    target="${first%.part-00}"
    if [ -f "$target" ]; then
        echo "exists:   $target"
        continue
    fi
    echo "building: $target"
    cat "$target".part-* > "$target.tmp"
    mv "$target.tmp" "$target"
done < <(find . -name '*.pt.part-00' | sort)

sha256sum -c CHECKPOINTS.sha256
