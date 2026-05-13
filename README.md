# 宇树 G1 机器人 Python 样例

> 本仓库是存放本人对宇树 G1 机器人二次开发的 Python 样例代码，欢迎大家参考和使用

三件事一起跑：**跟随 + 握手手势 + 语音对话**。
Python 全部封装在 `core/`，C++ 端整合为一个进程 `g1_node`，**只用一个 UDP 端口**。

## 目录

```
g1/
├── core/                 # 高级 API（封装好的类）
│   ├── bridge.py          ─ class Bridge   单端口 UDP，发 move/arm/tts
│   ├── vision.py          ─ class Vision   YOLO-pose：跟随 + 手势 + 障碍
│   ├── voice.py           ─ class Voice    唤醒→ASR→Chat→TTS（后台线程）
│   ├── chat.py            ─ class Chat     ollama 对话 + 历史持久化
│   └── stream.py          ─ class MjpegStream  共享 MJPEG 调试流
├── g1.py                 # ⭐ 主入口：三件事一起跑
├── talk.py               # 纯语音工具（不带视觉/动作）
├── llm.py                # 纯 ollama REPL
├── run.sh                # 统一启动脚本
├── keywords.txt          # 唤醒词拼音（自动生成）
├── fastdds_no_shm.xml    # FastDDS 关闭共享内存
└── yolov8n*.pt / .onnx   # YOLO 模型
```

## 通信协议（一个 UDP 端口 = 9870）

```
byte 0 = type
  0x01 MOVE → +3 floats (dist, err_norm, blocked)   → 13 字节
  0x02 ARM  → +1 uint8  (action_id)                  →  2 字节
  0x03 TTS  → +UTF-8 text                            → 1+N 字节
```

Python `core/bridge.py` 和 C++ `unitree_sdk2/source/g1_node/main.cpp` 两边各自实现这套。改字段记得两边同步。

## 硬件

- 机器人：宇树 G1
- 外挂视觉计算：Jetson Orin NX 16GB
- 相机：Intel RealSense D435i（G1 头部自带）
- 麦克风：USB（语音输入）；扬声器：G1 自带喇叭（云 TTS）

## 软件栈

- Ubuntu 20.04 · JetPack 5.1.1 · CUDA 11.4 · ROS 2 Foxy
- conda env `vision`（Python 3.8）
- ultralytics YOLOv8（TensorRT）· pyrealsense2 · sherpa-onnx · ollama (qwen3:8b)

## 用法

**先在另一终端跑 C++ 端**（绑 UDP :9870，启动三个 G1 客户端：LocoClient/ArmActionClient/AudioClient）：
```bash
~/unitree_sdk2/build/bin/g1_node eth0
```

**然后在 Jetson 这边跑 Python 端：**
```bash
cd ~/g1
./run.sh                          # 默认：g1.py 三件事全开
./run.sh g1 --no-voice            # 只视觉（跟随 + 手势）
./run.sh g1 --no-vision           # 只语音
./run.sh g1 --no-gesture          # 只跟随
./run.sh talk --hear              # 纯听写调试（不唤醒/不对话）
./run.sh llm                      # ollama REPL
```

跨机器（视觉机和机器人不在同一台）：
```bash
./run.sh g1 --bridge-host 192.168.123.164    # 把命令发到 G1 那台
```

## 注意

- `.engine` 文件是 TensorRT 编译产物，机器相关，clone 后首次运行会自动从 `.pt` export
- ROS Domain ID 与 G1 机载电脑对齐：`export ROS_DOMAIN_ID=1`
- D435i 深度图和彩色图必须用 `rs.align` 对齐后再取深度
- 回声坑：G1 喇叭说的话会被麦克风拾到 → 唤醒确认用本地 `beep()`（非 TTS）；TTS 回复按字数估时长，机器人念完前不收麦
- 调试 MJPEG 流默认在 `http://<jetson-ip>:6769/`

## 类的高级接口

```python
from core import Bridge, Vision, Voice, Chat, MjpegStream

bridge = Bridge()                        # 单端口 UDP 客户端
vision = Vision(bridge, stream=MjpegStream(6769))  # follow + gesture
voice  = Voice(bridge, chat=Chat())      # 后台线程跑 wake → ASR → chat → TTS

voice.start()        # 后台
vision.run()         # 主线程阻塞
# Ctrl-C 后
voice.stop(); bridge.close()
```
