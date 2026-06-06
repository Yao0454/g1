import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

// ─── 数据模型 ────────────────────────────────────────────────────────────────

class LogEntry {
  final String level;    // info / error / asr / llm / system
  final String message;
  final String timestamp;
  const LogEntry({
    required this.level,
    required this.message,
    required this.timestamp,
  });
}

// ─── 主服务类 ─────────────────────────────────────────────────────────────────

class RobotService extends ChangeNotifier {
  static const int _maxLogs = 150;

  // 连接配置（从 SharedPreferences 加载）
  String host = '192.168.123.164';
  int wsPort = 8765;
  int mjpegPort = 6769;

  // 连接状态
  bool connected = false;
  bool connecting = false;

  // 机器人模式状态（与 Jetson 同步）
  bool follow = true;
  bool gesture = true;
  bool voice = true;

  // Debug 面板数据
  List<LogEntry> logs = [];
  String lastAsr = '';
  String lastLlm = '';

  WebSocketChannel? _channel;
  StreamSubscription? _sub;
  Timer? _reconnectTimer;
  bool _wantConnect = false;

  // ─── 计算属性 ────────────────────────────────────────────────────────────

  String get mjpegUrl => 'http://$host:$mjpegPort/';
  String get wsUrl => 'ws://$host:$wsPort';

  /// 从持久化存储加载设置
  Future<void> loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    host = prefs.getString('host') ?? host;
    wsPort = prefs.getInt('wsPort') ?? wsPort;
    mjpegPort = prefs.getInt('mjpegPort') ?? mjpegPort;
    notifyListeners();
  }

  Future<void> saveSettings({
    required String newHost,
    required int newWsPort,
    required int newMjpegPort,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    host = newHost;
    wsPort = newWsPort;
    mjpegPort = newMjpegPort;
    await prefs.setString('host', host);
    await prefs.setInt('wsPort', wsPort);
    await prefs.setInt('mjpegPort', mjpegPort);
    notifyListeners();
  }

  // ─── 连接管理 ────────────────────────────────────────────────────────────

  Future<void> connect() async {
    _wantConnect = true;
    await _doConnect();
  }

  void disconnect() {
    _wantConnect = false;
    _reconnectTimer?.cancel();
    _sub?.cancel();
    _channel?.sink.close();
    connected = false;
    connecting = false;
    notifyListeners();
  }

  Future<void> _doConnect() async {
    if (connecting || connected) return;
    connecting = true;
    notifyListeners();

    try {
      final uri = Uri.parse(wsUrl);
      _channel = WebSocketChannel.connect(uri);
      await _channel!.ready;

      connected = true;
      connecting = false;
      _addLog('system', '已连接 $wsUrl');
      notifyListeners();

      _sub = _channel!.stream.listen(
        _onMessage,
        onError: (_) => _handleDisconnect(),
        onDone: _handleDisconnect,
        cancelOnError: false,
      );
    } catch (e) {
      connected = false;
      connecting = false;
      _addLog('error', '连接失败：$e');
      notifyListeners();
      if (_wantConnect) _scheduleReconnect();
    }
  }

  void _handleDisconnect() {
    _sub?.cancel();
    connected = false;
    _addLog('system', '连接断开');
    notifyListeners();
    if (_wantConnect) _scheduleReconnect();
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 3), () {
      if (_wantConnect && !connected) _doConnect();
    });
  }

  // ─── 消息处理 ────────────────────────────────────────────────────────────

  void _onMessage(dynamic data) {
    try {
      final msg = jsonDecode(data as String) as Map<String, dynamic>;
      switch (msg['type'] as String?) {
        case 'state':
          follow = msg['follow'] as bool? ?? follow;
          gesture = msg['gesture'] as bool? ?? gesture;
          voice = msg['voice'] as bool? ?? voice;
        case 'log':
          _addLog(
            msg['level'] as String? ?? 'info',
            '[${msg['ts'] ?? ''}] ${msg['msg'] ?? ''}',
          );
        case 'asr':
          lastAsr = msg['text'] as String? ?? '';
          _addLog('asr', 'ASR：$lastAsr');
        case 'llm':
          lastLlm = msg['text'] as String? ?? '';
          _addLog('llm', 'LLM：$lastLlm');
      }
      notifyListeners();
    } catch (_) {}
  }

  void _addLog(String level, String message) {
    final entry = LogEntry(
      level: level,
      message: message,
      timestamp: _ts(),
    );
    if (logs.length >= _maxLogs) {
      logs = [...logs.sublist(logs.length - _maxLogs + 1), entry];
    } else {
      logs = [...logs, entry];
    }
  }

  String _ts() {
    final n = DateTime.now();
    return '${n.hour.toString().padLeft(2, '0')}:'
        '${n.minute.toString().padLeft(2, '0')}:'
        '${n.second.toString().padLeft(2, '0')}';
  }

  // ─── 发送命令 ────────────────────────────────────────────────────────────

  /// 遥控移动。
  /// g1_node C++ 控制律：vx = 0.8*(dist-1.0), vyaw = -0.6*err
  /// 摇杆映射：
  ///   dist = 1.0 + joystickY * 0.75  → vx ∈ [-0.6, 0.6] m/s
  ///   err  = joystickX / 0.6         → vyaw ∈ [-1.0, 1.0] rad/s
  void sendJoystick(double jx, double jy) {
    if (!connected) return;
    final dist = 1.0 + jy * 0.75;
    final err = jx / 0.6;
    _send({'type': 'move', 'dist': dist, 'err': err, 'blocked': 0.0});
  }

  void sendStop() => _send({'type': 'stop'});

  void sendArm(String action) => _send({'type': 'arm', 'action': action});

  void sendTts(String text) {
    if (text.trim().isEmpty) return;
    _send({'type': 'tts', 'text': text.trim()});
  }

  void toggle(String feature, bool enabled) =>
      _send({'type': 'toggle', 'feature': feature, 'enabled': enabled});

  void _send(Map<String, dynamic> msg) {
    if (connected) _channel?.sink.add(jsonEncode(msg));
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}
