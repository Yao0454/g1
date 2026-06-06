import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/robot_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _hostCtrl;
  late final TextEditingController _wsCtrl;
  late final TextEditingController _mjpegCtrl;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    final svc = context.read<RobotService>();
    _hostCtrl = TextEditingController(text: svc.host);
    _wsCtrl = TextEditingController(text: svc.wsPort.toString());
    _mjpegCtrl = TextEditingController(text: svc.mjpegPort.toString());
  }

  @override
  void dispose() {
    _hostCtrl.dispose();
    _wsCtrl.dispose();
    _mjpegCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final host = _hostCtrl.text.trim();
    final ws = int.tryParse(_wsCtrl.text.trim()) ?? 8765;
    final mjpeg = int.tryParse(_mjpegCtrl.text.trim()) ?? 6769;

    if (host.isEmpty) {
      _showSnack('请输入机器人 IP 地址');
      return;
    }

    setState(() => _saving = true);
    final svc = context.read<RobotService>();
    svc.disconnect();
    await svc.saveSettings(
        newHost: host, newWsPort: ws, newMjpegPort: mjpeg);
    svc.connect();
    setState(() => _saving = false);

    if (mounted) Navigator.pop(context);
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: const Color(0xFF1A1A2E)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('设置',
            style: TextStyle(color: Color(0xFF00E5FF))),
        leading: const BackButton(color: Color(0xFF00E5FF)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // ── 网络配置 ────────────────────────────────────────────────────
          _SectionHeader(label: '网络连接'),
          const SizedBox(height: 12),
          _Field(
            label: 'Jetson / 机器人 IP',
            hint: '例：192.168.123.164',
            ctrl: _hostCtrl,
            type: TextInputType.url,
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _Field(
                  label: 'WebSocket 端口',
                  hint: '8765',
                  ctrl: _wsCtrl,
                  type: TextInputType.number,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _Field(
                  label: 'MJPEG 视频端口',
                  hint: '6769',
                  ctrl: _mjpegCtrl,
                  type: TextInputType.number,
                ),
              ),
            ],
          ),

          const SizedBox(height: 28),

          // ── 说明 ────────────────────────────────────────────────────────
          _SectionHeader(label: '使用说明'),
          const SizedBox(height: 10),
          ..._tips.map((t) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('·  ',
                        style: TextStyle(
                            color: Color(0xFF00E5FF), fontSize: 13)),
                    Expanded(
                      child: Text(t,
                          style: const TextStyle(
                              color: Color(0xFF9E9E9E), fontSize: 13,
                              height: 1.5)),
                    ),
                  ],
                ),
              )),

          const SizedBox(height: 32),

          // ── 保存按钮 ─────────────────────────────────────────────────────
          FilledButton(
            onPressed: _saving ? null : _save,
            style: FilledButton.styleFrom(
              minimumSize: const Size.fromHeight(48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10)),
            ),
            child: _saving
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.black),
                  )
                : const Text('保存并重新连接',
                    style: TextStyle(
                        fontSize: 15, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  static const _tips = [
    '手机与 Jetson 须在同一 WiFi 网络下。',
    '关闭"跟随"后摇杆才能遥控移动，否则视觉系统会覆盖指令。',
    '摇杆映射：上推 = 前进 (vx=0.6m/s)，左右 = 转向 (vyaw=1.0rad/s)。',
    '手臂动作不受跟随/遥控模式影响，随时可用。',
    '让机器人说话时，TTS 走 G1 自带喇叭（约 0.22s/字 延迟）。',
    '语音关闭后唤醒词和对话均暂停，不影响其他功能。',
  ];
}

class _SectionHeader extends StatelessWidget {
  final String label;
  const _SectionHeader({required this.label});

  @override
  Widget build(BuildContext context) {
    return Text(
      label,
      style: const TextStyle(
        color: Color(0xFF00E5FF),
        fontSize: 12,
        fontWeight: FontWeight.w700,
        letterSpacing: 1.5,
      ),
    );
  }
}

class _Field extends StatelessWidget {
  final String label;
  final String hint;
  final TextEditingController ctrl;
  final TextInputType type;

  const _Field({
    required this.label,
    required this.hint,
    required this.ctrl,
    this.type = TextInputType.text,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(
                fontSize: 12, color: Color(0xFF9E9E9E))),
        const SizedBox(height: 6),
        TextField(
          controller: ctrl,
          keyboardType: type,
          style: const TextStyle(
              fontSize: 14, color: Color(0xFFE0E0E0)),
          decoration: InputDecoration(hintText: hint),
        ),
      ],
    );
  }
}
