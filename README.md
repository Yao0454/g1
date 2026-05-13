# 宇树 G1 机器人 Python 样例

> 本仓库是存放本人对宇树 G1 机器人二次开发的 Python 样例代码，欢迎大家参考和使用

## 目录

| 文件 | 作用 |
|------|------|
| `vision.py` | 人体跟随：YOLO 检测最近的人 + RealSense 深度测距 + 障碍检测 → UDP :9870 → 运动控制 |
| `move.py` | 手势识别：YOLO-pose 检测手腕 → UDP :9871 → C++ 手臂动作节点 |
| `follow.py` | 上面两路合并版（跟随 + 手势，过滤机器人自身手臂） |
| `talk.py` | 语音助手：唤醒词 → VAD 录音 → ASR (SenseVoice) → ollama → TTS（本地 / G1 喇叭） |
| `llm.py` | ollama 对话循环（qwen3:8b），含会话历史持久化 |
| `run.sh` | 统一入口，自动 source ROS + activate conda env |

## 硬件

- 机器人：宇树 G1
- 外挂视觉计算：Jetson Orin NX 16GB
- 相机：Intel RealSense D435i（G1 头部自带）
- 麦克风/喇叭：USB（语音 IO）+ G1 自带喇叭（云 TTS）

## 软件栈

- Ubuntu 20.04 · JetPack 5.1.1 · CUDA 11.4 · ROS 2 Foxy
- conda env `vision`（Python 3.8）
- ultralytics YOLOv8 (TensorRT) · pyrealsense2 · sherpa-onnx · ollama

## 用法

```bash
cd ~/g1
./run.sh              # 默认 follow（跟随 + 手势）
./run.sh vision       # 只跟随
./run.sh move         # 只手势
./run.sh talk --wake --g1 --debug   # 语音助手（唤醒 + G1 喇叭朗读）
./run.sh llm          # ollama 交互对话
```

`talk.py --wake --g1` 需要先在另一终端跑 C++ talk 节点：

```bash
~/unitree_sdk2/build/bin/talk eth0
```

## 注意

- `.engine` 文件是 TensorRT 编译产物，机器相关，clone 后首次运行会从 `.pt` 自动 export
- ROS Domain ID 与 G1 机载电脑对齐：`export ROS_DOMAIN_ID=1`
- D435i 深度图和彩色图必须用 `rs.align` 对齐后再取深度
