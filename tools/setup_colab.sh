#!/usr/bin/env bash
# Colab install for IndexTTS. Quiet on success, full logs on failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE=$(printf '─%.0s' {1..52})
CU128_INDEX="https://download.pytorch.org/whl/cu128"
EXTRA=""

usage() {
    cat <<'EOF'
Usage: bash tools/setup_colab.sh [--extra webui|dub|highlight|intent|novel] [--verbose]

Installs IndexTTS into the current Python (Colab system Python).
Always installs a cu128 torch / torchvision / torchaudio set that match.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --extra)
            EXTRA="${2:-}"
            shift 2
            ;;
        --verbose|-v)
            INDEX_TTS_VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -n "${NO_COLOR:-}" ]; then
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
elif [ -t 1 ] || [ -n "${FORCE_COLOR:-}" ]; then
    C_TITLE=$'\033[38;5;150m'
    C_DIM=$'\033[38;5;245m'
    C_OK=$'\033[38;5;108m'
    C_WARN=$'\033[38;5;136m'
    C_ERR=$'\033[91m'
    C_RST=$'\033[0m'
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi

step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  %-15s${2:-$C_DIM}%s${C_RST}\n" "" "$1"; }

setup_fail() {
    local exit_code=$1
    shift
    [ "$exit_code" -ne 0 ] || exit_code=1
    step "error" "$*" "$C_ERR" >&2
    exit "$exit_code"
}

_is_verbose() { [ "${INDEX_TTS_VERBOSE:-0}" = "1" ]; }

run_quiet() {
    local label=$1
    shift
    if _is_verbose; then
        "$@" && return 0
        setup_fail "$?" "$label failed"
    fi
    local tmplog
    tmplog=$(mktemp) || setup_fail 1 "could not create temp log"
    if "$@" >"$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    fi
    local exit_code=$?
    step "error" "$label failed (exit $exit_code)" "$C_ERR" >&2
    cat "$tmplog" >&2
    rm -f "$tmplog"
    setup_fail "$exit_code" "$label failed"
}

PIP=(python3 -m pip)
if ! command -v python3 >/dev/null 2>&1; then
    setup_fail 1 "python3 not found"
fi

is_colab=0
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -n "${COLAB_BACKEND_VERSION:-}" ]; then
    is_colab=1
elif [ -d /content ] && python3 -c "import google.colab" >/dev/null 2>&1; then
    is_colab=1
fi

cd "$REPO_ROOT"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
printf "  ${C_TITLE}%s${C_RST}\n" "IndexTTS Colab setup"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
step "repo" "$REPO_ROOT"
step "python" "$(python3 -c 'import sys; print(sys.version.split()[0])')"
if [ "$is_colab" -eq 1 ]; then
    step "runtime" "Google Colab"
else
    step "runtime" "local / other"
fi

if [ "$is_colab" -eq 1 ]; then
    step "cleanup" "remove Colab tensorflow/keras (keep torch)"
    run_quiet "uninstall tensorflow/keras" \
        "${PIP[@]}" uninstall -y tensorflow keras
fi

step "ninja" "install build helper"
run_quiet "pip install ninja" "${PIP[@]}" install -q ninja

step "indextts" "editable install + matching torch/vision/audio (cu128)"
run_quiet "pip install indextts webui torch stack" \
    "${PIP[@]}" install -e "${REPO_ROOT}[webui]" torchvision torchaudio \
    --extra-index-url "$CU128_INDEX"

verify_torch_stack() {
    python3 - <<'PY'
import torch
import torchvision
from torchvision.ops import nms

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"torchvision={torchvision.__version__}")
boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
scores = torch.tensor([1.0])
nms(boxes, scores, 0.5)
import numba
print(f"numba={numba.__version__}")
PY
}

step "verify" "import torch / torchvision / numba and nms op"
if ! verify_torch_stack; then
    step "repair" "torchvision does not match torch; reinstall from cu128" "$C_WARN"
    run_quiet "reinstall torchvision/torchaudio" \
        "${PIP[@]}" install -U torchvision torchaudio --extra-index-url "$CU128_INDEX"
    if ! verify_torch_stack; then
        setup_fail 1 "torchvision::nms still broken; Runtime → Restart session, then re-run setup"
    fi
fi

install_dub_extras() {
    if command -v apt-get >/dev/null 2>&1; then
        step "apt" "rubberband-cli"
        run_quiet "apt-get rubberband-cli" \
            bash -lc "apt-get update -qq && apt-get install -y -qq rubberband-cli"
    fi
    step "extras" "demucs whisperx pyrubberband openai"
    run_quiet "pip dub extras" \
        "${PIP[@]}" install -q demucs whisperx pyrubberband soundfile openai
}

install_vl_extras() {
    step "extras" "whisperx qwen-vl-utils bitsandbytes"
    run_quiet "pip highlight extras" \
        "${PIP[@]}" install -q whisperx soundfile openai \
        "transformers>=4.52.1" qwen-vl-utils accelerate bitsandbytes
    step "flash-attn" "optional (ok if this fails)"
    if ! "${PIP[@]}" install -q flash-attn --no-build-isolation; then
        substep "flash-attn skipped" "$C_WARN"
    fi
}

install_novel_extras() {
    step "extras" "openai"
    run_quiet "pip novel extras" \
        "${PIP[@]}" install -q openai soundfile
}

case "$EXTRA" in
    ""|webui) ;;
    dub) install_dub_extras ;;
    highlight|intent) install_vl_extras ;;
    novel) install_novel_extras ;;
    *) setup_fail 2 "unknown --extra $EXTRA (use webui|dub|highlight|intent|novel)" ;;
esac

echo ""
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
printf "  ${C_TITLE}%s${C_RST}\n" "IndexTTS setup complete"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
if [ "$is_colab" -eq 1 ]; then
    substep "from tools.colab import start"
    substep "start()"
    substep "If torchvision was already imported, Runtime → Restart session first."
else
    substep "uv run webui.py --fp16"
fi
echo ""
