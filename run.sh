#!/usr/bin/env bash
# g1 项目统一入口（vision conda env + ROS Foxy + FastDDS no-shm）。
#
# 默认：./run.sh   → 三件事一起跑（跟随 + 手势 + 语音）
#
#   ./run.sh g1     [args]    跟随 + 手势 + 语音 (推荐)
#   ./run.sh talk   [args]    纯语音
#   ./run.sh llm    [args]    纯 ollama REPL
#
# 任何参数都会透传给底下的 python 脚本，比如：
#   ./run.sh g1 --no-voice          # 只视觉
#   ./run.sh g1 --bridge-host 192.168.x.x
#   ./run.sh talk --hear            # 纯听写
#
# C++ 端先单独跑：
#   ~/unitree_sdk2/build/bin/g1_node eth0
set -e

source /opt/ros/foxy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision

cd "$(dirname "$0")"
export FASTRTPS_DEFAULT_PROFILES_FILE="$(pwd)/fastdds_no_shm.xml"

target="${1:-g1}"
shift || true

case "$target" in
  g1|talk|llm)
    exec python "${target}.py" "$@" ;;
  *)
    echo "用法: $0 {g1|talk|llm} [args...]" >&2
    exit 1 ;;
esac
