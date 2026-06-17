#!/bin/bash
# Build the G1 host-physics walking demo.
#   web/build_demo.sh          -> native (clang + raylib); runnable now
#   web/build_demo.sh --web    -> WASM (needs emscripten `emcc` on PATH)
# Run `bash setup.sh` once first — it clones the engine fork (puffernet.h lives there).
set -e
cd "$(dirname "$0")/.."
WEB=0; [ "$1" = "--web" ] && WEB=1

[ -d vendor/PufferLib/src ] || { echo "engine fork missing — run: bash setup.sh"; exit 1; }

# raylib 5.5 prebuilt (fetched, not vendored), staged where the fork keeps it
RL_BASE="https://github.com/raysan5/raylib/releases/download/5.5"
fetch_raylib() {  # $1 = release file (.tar.gz/.zip), $2 = unpacked dir name
  [ -d "vendor/PufferLib/$2" ] && return
  echo "fetching $2…"
  ( cd vendor/PufferLib
    case "$1" in
      *.zip) curl -sL "$RL_BASE/$1" -o rl.zip && unzip -q rl.zip && rm rl.zip ;;
      *)     curl -sL "$RL_BASE/$1" -o rl.tgz && tar xf rl.tgz && rm rl.tgz ;;
    esac )
}

# model constants + visual mesh asset (baked from g1.mjb; committed, regen via tools/)
[ -f web/g1_model_const.h ] || .venv/bin/python tools/dump_host_consts.py
[ -f web/g1_meshes.bin ]    || .venv/bin/python tools/dump_g1_meshes.py

if [ "$WEB" = "1" ]; then
  command -v emcc >/dev/null || { echo "emcc not found — install emscripten"; exit 1; }
  RLW=vendor/PufferLib/raylib-5.5_webassembly
  fetch_raylib raylib-5.5_webassembly.zip raylib-5.5_webassembly
  mkdir -p build/web
  emcc -O3 web/g1_demo.c -o build/web/index.html \
    -I web -I "$RLW/include" -I vendor/PufferLib/src \
    "$RLW/lib/libraylib.a" \
    -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFORCE_FILESYSTEM=1 \
    -sINITIAL_MEMORY=64MB -sALLOW_MEMORY_GROWTH \
    -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
    --preload-file assets/nanoG1.bin@assets/nanoG1.bin \
    --preload-file web/g1_meshes.bin@web/g1_meshes.bin \
    --preload-file web/assets/font.ttf@web/assets/font.ttf \
    --shell-file web/shell.html
  echo "Built: build/web/index.html"
  echo "  local : (cd build/web && python3 -m http.server) -> http://localhost:8000"
  echo "  deploy: vercel deploy --prod build/web"
else
  case "$(uname -s)" in
    Darwin) fetch_raylib raylib-5.5_macos.tar.gz raylib-5.5_macos
            RL=vendor/PufferLib/raylib-5.5_macos
            FW="-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL" ;;
    *)      fetch_raylib raylib-5.5_linux_amd64.tar.gz raylib-5.5_linux_amd64
            RL=vendor/PufferLib/raylib-5.5_linux_amd64
            FW="-lGL -lpthread -ldl -lrt -lX11" ;;
  esac
  mkdir -p build
  clang -O2 web/g1_demo.c -o build/g1demo \
    -I web -I "$RL/include" -I vendor/PufferLib/src \
    "$RL/lib/libraylib.a" -lm $FW
  echo "Built: build/g1demo   (run: ./build/g1demo assets/nanoG1.bin)"
fi
