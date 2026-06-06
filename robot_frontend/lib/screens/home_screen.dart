import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/robot_service.dart';
import '../widgets/joystick.dart';
import '../widgets/mjpeg_view.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _ttsCtrl = TextEditingController();
  final _scrollCtrl = ScrollController();
  bool _debugOpen = false;

  @override
  void initState() {
    super.initState();
    // 加载设置后自动连接
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final svc = context.read<RobotService>();
      await svc.loadSettings();
      svc.connect();
    });
  }

  @override
  void dispose() {
    _ttsCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  // ─── 构建 ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(),
      body: Consumer<RobotService>(
        builder: (_, svc, child) => _buildBody(svc),
      ),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      title: const Text(
        'G1 Control',
        style: TextStyle(
          fontWeight: FontWeight.w600,
          letterSpacing: 1.2,
          color: Color(0xFF00E5FF),
        ),
      ),
      actions: [
        Consumer<RobotService>(
          builder: (_, svc, child) => _ConnectionDot(
            connected: svc.connected,
            connecting: svc.connecting,
            onTap: () => svc.connected ? svc.disconnect() : svc.connect(),
          ),
        ),
        IconButton(
          icon: const Icon(Icons.settings_outlined),
          tooltip: '设置',
          onPressed: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const SettingsScreen()),
          ),
        ),
      ],
    );
  }

  Widget _buildBody(RobotService svc) {
    // 日志滚到底
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.jumpTo(_scrollCtrl.position.maxScrollExtent);
      }
    });

    return SafeArea(
      child: Column(
        children: [
          // ── 摄像头 ────────────────────────────────────────────────────────
          _CameraSection(url: svc.mjpegUrl),

          // ── 模式开关 ──────────────────────────────────────────────────────
          _ModeRow(svc: svc),

          // ── 遥控区 ────────────────────────────────────────────────────────
          Expanded(child: _ControlRow(svc: svc)),

          // ── TTS 输入 ──────────────────────────────────────────────────────
          _TtsBar(ctrl: _ttsCtrl, svc: svc),

          // ── Debug 面板 ────────────────────────────────────────────────────
          _DebugPanel(
            svc: svc,
            open: _debugOpen,
            scrollCtrl: _scrollCtrl,
            onToggle: () => setState(() => _debugOpen = !_debugOpen),
          ),
        ],
      ),
    );
  }
}

// ─── 摄像头区 ─────────────────────────────────────────────────────────────────

class _CameraSection extends StatelessWidget {
  final String url;
  const _CameraSection({required this.url});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 210,
      width: double.infinity,
      color: const Color(0xFF0A0A16),
      child: Stack(
        children: [
          MjpegView(url: url),
          // 左上角标签
          Positioned(
            left: 8, top: 6,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.videocam, size: 12, color: Color(0xFF00E5FF)),
                  SizedBox(width: 4),
                  Text('LIVE',
                      style: TextStyle(
                          color: Color(0xFF00E5FF),
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── 模式开关行 ───────────────────────────────────────────────────────────────

class _ModeRow extends StatelessWidget {
  final RobotService svc;
  const _ModeRow({required this.svc});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Row(
        children: [
          _ModeChip(
            label: '跟随',
            icon: Icons.directions_walk,
            active: svc.follow,
            onTap: () => svc.toggle('follow', !svc.follow),
          ),
          const SizedBox(width: 8),
          _ModeChip(
            label: '手势',
            icon: Icons.back_hand_outlined,
            active: svc.gesture,
            onTap: () => svc.toggle('gesture', !svc.gesture),
          ),
          const SizedBox(width: 8),
          _ModeChip(
            label: '语音',
            icon: Icons.mic_outlined,
            active: svc.voice,
            onTap: () => svc.toggle('voice', !svc.voice),
          ),
          const Spacer(),
          // 连接状态文字
          Text(
            svc.connected
                ? svc.follow
                    ? '自动跟随中'
                    : '手动遥控'
                : '未连接',
            style: TextStyle(
              color: svc.connected
                  ? svc.follow
                      ? const Color(0xFF00E5FF)
                      : const Color(0xFFFF9800)
                  : const Color(0xFF555577),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

class _ModeChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool active;
  final VoidCallback onTap;
  const _ModeChip({
    required this.label,
    required this.icon,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: active
              ? const Color(0xFF00E5FF).withValues(alpha: 0.15)
              : const Color(0xFF1A1A2E),
          border: Border.all(
            color: active ? const Color(0xFF00E5FF) : const Color(0xFF2A2A4A),
            width: 1.2,
          ),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon,
                size: 14,
                color: active
                    ? const Color(0xFF00E5FF)
                    : const Color(0xFF555577)),
            const SizedBox(width: 4),
            Text(label,
                style: TextStyle(
                  fontSize: 12,
                  color: active
                      ? const Color(0xFF00E5FF)
                      : const Color(0xFF555577),
                  fontWeight:
                      active ? FontWeight.w600 : FontWeight.normal,
                )),
          ],
        ),
      ),
    );
  }
}

// ─── 遥控区（摇杆 + 手臂按钮）────────────────────────────────────────────────

class _ControlRow extends StatelessWidget {
  final RobotService svc;
  const _ControlRow({required this.svc});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          // 摇杆（跟随模式开启时禁用）
          _JoystickPanel(svc: svc),
          const SizedBox(width: 12),
          // 手臂按钮
          Expanded(child: _ArmPanel(svc: svc)),
        ],
      ),
    );
  }
}

class _JoystickPanel extends StatelessWidget {
  final RobotService svc;
  const _JoystickPanel({required this.svc});

  @override
  Widget build(BuildContext context) {
    final disabled = svc.follow || !svc.connected;
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Joystick(
          size: 155,
          enabled: !disabled,
          onChanged: (x, y) {
            if (disabled) return;
            svc.sendJoystick(x, y); // (0,0) → dist=1.0 → vx=0 → 自然停止
          },
        ),
        const SizedBox(height: 4),
        Text(
          disabled
              ? svc.follow
                  ? '跟随模式（关跟随可遥控）'
                  : '未连接'
              : '手动遥控',
          style: const TextStyle(fontSize: 10, color: Color(0xFF555577)),
        ),
      ],
    );
  }
}

class _ArmPanel extends StatelessWidget {
  final RobotService svc;
  const _ArmPanel({required this.svc});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // 手臂动作 2×2
        Row(
          children: [
            _ArmBtn(
              label: '挥手',
              icon: Icons.waving_hand_outlined,
              onTap: () => svc.sendArm('wave'),
            ),
            const SizedBox(width: 8),
            _ArmBtn(
              label: '握手',
              icon: Icons.handshake_outlined,
              onTap: () => svc.sendArm('handshake'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            _ArmBtn(
              label: '举臂',
              icon: Icons.accessibility_new,
              onTap: () => svc.sendArm('raise'),
            ),
            const SizedBox(width: 8),
            _ArmBtn(
              label: '收回',
              icon: Icons.crop_square_outlined,
              onTap: () => svc.sendArm('retract'),
              accent: false,
            ),
          ],
        ),
        const SizedBox(height: 10),
        // 急停按钮
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: svc.connected ? svc.sendStop : null,
            icon: const Icon(Icons.stop_circle_outlined, size: 18),
            label: const Text('急  停',
                style: TextStyle(
                    fontWeight: FontWeight.bold, letterSpacing: 2)),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFB71C1C),
              foregroundColor: Colors.white,
              disabledBackgroundColor: const Color(0xFF2A1A1A),
              disabledForegroundColor: const Color(0xFF555555),
              padding: const EdgeInsets.symmetric(vertical: 10),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8)),
              elevation: 0,
            ),
          ),
        ),
      ],
    );
  }
}

class _ArmBtn extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback onTap;
  final bool accent;

  const _ArmBtn({
    required this.label,
    required this.icon,
    required this.onTap,
    this.accent = true,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: OutlinedButton.icon(
        onPressed: onTap,
        icon: Icon(icon, size: 14),
        label: Text(label, style: const TextStyle(fontSize: 12)),
        style: OutlinedButton.styleFrom(
          foregroundColor: accent
              ? const Color(0xFF00E5FF)
              : const Color(0xFF9E9E9E),
          side: BorderSide(
            color: accent
                ? const Color(0xFF00E5FF).withValues(alpha: 0.5)
                : const Color(0xFF333355),
            width: 1,
          ),
          padding: const EdgeInsets.symmetric(vertical: 8),
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(7)),
        ),
      ),
    );
  }
}

// ─── TTS 输入栏 ───────────────────────────────────────────────────────────────

class _TtsBar extends StatelessWidget {
  final TextEditingController ctrl;
  final RobotService svc;
  const _TtsBar({required this.ctrl, required this.svc});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 4),
      child: Row(
        children: [
          const Icon(Icons.record_voice_over_outlined,
              size: 18, color: Color(0xFF555577)),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: ctrl,
              style: const TextStyle(fontSize: 13, color: Color(0xFFE0E0E0)),
              decoration: const InputDecoration(
                hintText: '让机器人说…',
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                isDense: true,
              ),
              textInputAction: TextInputAction.send,
              onSubmitted: (v) {
                svc.sendTts(v);
                ctrl.clear();
              },
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            onPressed: svc.connected
                ? () {
                    svc.sendTts(ctrl.text);
                    ctrl.clear();
                  }
                : null,
            icon: const Icon(Icons.send_rounded),
            color: const Color(0xFF00E5FF),
            iconSize: 20,
            tooltip: '发送',
            padding: const EdgeInsets.all(8),
          ),
        ],
      ),
    );
  }
}

// ─── Debug 面板 ───────────────────────────────────────────────────────────────

class _DebugPanel extends StatelessWidget {
  final RobotService svc;
  final bool open;
  final ScrollController scrollCtrl;
  final VoidCallback onToggle;

  const _DebugPanel({
    required this.svc,
    required this.open,
    required this.scrollCtrl,
    required this.onToggle,
  });

  Color _levelColor(String level) => switch (level) {
        'error' => const Color(0xFFFF5252),
        'asr' => const Color(0xFF69F0AE),
        'llm' => const Color(0xFFFFD740),
        'system' => const Color(0xFF64B5F6),
        _ => const Color(0xFF9E9E9E),
      };

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xFF0D0D1A),
        border: Border(top: BorderSide(color: Color(0xFF1E1E3A), width: 1)),
      ),
      child: Column(
        children: [
          // 标题栏
          GestureDetector(
            onTap: onToggle,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(
                children: [
                  const Icon(Icons.terminal, size: 14,
                      color: Color(0xFF555577)),
                  const SizedBox(width: 6),
                  const Text('DEBUG',
                      style: TextStyle(
                          fontSize: 11,
                          color: Color(0xFF555577),
                          letterSpacing: 1.5,
                          fontWeight: FontWeight.w600)),
                  const SizedBox(width: 6),
                  // ASR / LLM 快速展示
                  if (svc.lastAsr.isNotEmpty)
                    Flexible(
                      child: Text(
                        '· ${svc.lastAsr}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            fontSize: 11, color: Color(0xFF69F0AE)),
                      ),
                    ),
                  const Spacer(),
                  Icon(
                    open
                        ? Icons.keyboard_arrow_down
                        : Icons.keyboard_arrow_up,
                    size: 16,
                    color: const Color(0xFF555577),
                  ),
                ],
              ),
            ),
          ),

          // 日志列表
          if (open)
            Container(
              height: 160,
              padding: const EdgeInsets.fromLTRB(8, 0, 8, 4),
              child: ListView.builder(
                controller: scrollCtrl,
                itemCount: svc.logs.length,
                itemBuilder: (_, i) {
                  final e = svc.logs[i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 1),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(e.timestamp,
                            style: const TextStyle(
                                fontSize: 10, color: Color(0xFF333355))),
                        const SizedBox(width: 6),
                        Flexible(
                          child: Text(
                            e.message,
                            style: TextStyle(
                                fontSize: 11,
                                color: _levelColor(e.level),
                                height: 1.4),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }
}

// ─── 连接状态指示点 ───────────────────────────────────────────────────────────

class _ConnectionDot extends StatelessWidget {
  final bool connected;
  final bool connecting;
  final VoidCallback onTap;
  const _ConnectionDot({
    required this.connected,
    required this.connecting,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = connected
        ? const Color(0xFF69F0AE)
        : connecting
            ? const Color(0xFFFFD740)
            : const Color(0xFF555577);

    return GestureDetector(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 12),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (connecting)
              const SizedBox(
                width: 12,
                height: 12,
                child: CircularProgressIndicator(
                    strokeWidth: 1.5, color: Color(0xFFFFD740)),
              )
            else
              Container(
                width: 9,
                height: 9,
                decoration: BoxDecoration(
                  color: color,
                  shape: BoxShape.circle,
                  boxShadow: connected
                      ? [
                          BoxShadow(
                              color: color.withValues(alpha: 0.6),
                              blurRadius: 6)
                        ]
                      : null,
                ),
              ),
            const SizedBox(width: 6),
            Text(
              connected ? '已连接' : connecting ? '连接中' : '未连接',
              style: TextStyle(fontSize: 11, color: color),
            ),
          ],
        ),
      ),
    );
  }
}
