#!/bin/bash
# One-time setup for building/running the demo + eval locally.
#
# Fetches the G1-specialized engine — a *pinned* fork of PufferLib (recipe.py).
# The browser demo links against its puffernet.h + raylib; eval.py builds on it.
#
# NOTE: local/Spark training also uses this checkout.
set -e
cd "$(dirname "$0")"

read URL BRANCH PIN < <(python3 - <<'PY'
import recipe as R
print(R.FORK, R.FORK_BRANCH, R.FORK_PIN)
PY
)

if [ ! -d vendor/PufferLib/.git ]; then
  echo "cloning engine fork: $BRANCH @ $PIN"
  git clone -b "$BRANCH" "$URL" vendor/PufferLib
fi
git -C vendor/PufferLib fetch -q origin "$BRANCH" || true
git -C vendor/PufferLib checkout -q "$PIN"
echo -n "engine pinned at: "; git -C vendor/PufferLib log --oneline -1

PATCH="$PWD/patches/pufferlib-g1-sim2real.patch"
if [ -f "$PATCH" ]; then
  if git -C vendor/PufferLib apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "local engine patch already applied: $(basename "$PATCH")"
  else
    git -C vendor/PufferLib apply --check "$PATCH"
    git -C vendor/PufferLib apply "$PATCH"
    echo "applied local engine patch: $(basename "$PATCH")"
  fi
fi

echo
echo "✓ setup done. Next:"
echo "    python train_local.py --smoke  # local/Spark stack check"
echo "    python train_local.py          # local/Spark training"
echo "    bash web/build_demo.sh        # native demo  -> ./build/g1demo assets/nanoG1.bin"
echo "    bash web/build_demo.sh --web  # WASM demo     -> build/web/index.html"
echo "    python eval.py assets/nanoG1.bin   # quality gate (does it walk?)"
