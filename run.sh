#!/usr/bin/env bash
# 一个入口跑 g1 项目里的四个 Python：
#   ./run.sh              → 默认 follow（跟随 + 手势 一体）
#   ./run.sh follow       → 同上
#   ./run.sh vision       → 只跟随（人体跟随，原 ~/Vision/main.py）
#   ./run.sh move         → 只手势（手腕→手臂动作，原 ~/Vision/movement.py）
#   ./run.sh talk [args]  → 语音助手（args 透传给 talk.py，比如 --wake --g1）
#   ./run.sh llm          → ollama 交互对话
# 都在 vision conda env 跑，ROS Foxy 已 source；FastDDS 用本目录的 no-shm 配置。
set -e

source /opt/ros/foxy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision

cd "$(dirname "$0")"
export FASTRTPS_DEFAULT_PROFILES_FILE="$(pwd)/fastdds_no_shm.xml"

target="${1:-follow}"
shift || true

case "$target" in
  follow|vision|move|talk|llm)
    exec python "${target}.py" "$@" ;;
  *)
    echo "用法: $0 {follow|vision|move|talk|llm} [args...]" >&2
    exit 1 ;;
esac
