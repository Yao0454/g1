import 'dart:async';
import 'dart:io' as io;
import 'dart:typed_data';

import 'package:flutter/material.dart';

/// MJPEG 摄像头画面，5s 无新帧自动重连。
/// 用 dart:io HttpClient 直接读字节流，不走 package:http（移动端对无限流支持差）。
class MjpegView extends StatefulWidget {
  final String url;
  final Widget? placeholder;

  const MjpegView({super.key, required this.url, this.placeholder});

  @override
  State<MjpegView> createState() => _MjpegViewState();
}

class _MjpegViewState extends State<MjpegView> {
  static const _timeoutSecs = 5;

  Uint8List? _frame;
  bool _reconnecting = false;
  int _retries = 0;

  io.HttpClient? _client;
  StreamSubscription? _sub;
  Timer? _watchdog;
  Timer? _retryTimer;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void didUpdateWidget(MjpegView old) {
    super.didUpdateWidget(old);
    if (old.url != widget.url) {
      _retries = 0;
      _cancelAll();
      _connect();
    }
  }

  @override
  void dispose() {
    _cancelAll();
    super.dispose();
  }

  void _cancelAll() {
    _watchdog?.cancel();
    _retryTimer?.cancel();
    _sub?.cancel();
    _client?.close(force: true);
    _watchdog = null;
    _retryTimer = null;
    _sub = null;
    _client = null;
  }

  void _connect() {
    if (!mounted || widget.url.isEmpty) return;
    _reconnecting = false;
    _startWatchdog();

    final client = io.HttpClient();
    client.connectionTimeout = const Duration(seconds: _timeoutSecs);
    _client = client;

    _runStream(client);
  }

  Future<void> _runStream(io.HttpClient client) async {
    try {
      final uri = Uri.parse(widget.url);
      final req = await client.getUrl(uri);
      req.headers.set('Cache-Control', 'no-cache');
      req.headers.set('Connection', 'keep-alive');
      final resp = await req.close();

      final buf = <int>[];
      _sub = resp.listen(
        (chunk) {
          buf.addAll(chunk);
          _extractFrames(buf);
        },
        onError: (_) => _scheduleReconnect(),
        onDone: () => _scheduleReconnect(),
        cancelOnError: false,
      );
    } catch (_) {
      _scheduleReconnect();
    }
  }

  // 从 buf 里提取所有完整 JPEG（FF D8 … FF D9）
  void _extractFrames(List<int> buf) {
    while (true) {
      int start = -1;
      for (int i = 0; i < buf.length - 1; i++) {
        if (buf[i] == 0xFF && buf[i + 1] == 0xD8) { start = i; break; }
      }
      if (start == -1) {
        if (buf.length > 1) buf.removeRange(0, buf.length - 1);
        return;
      }
      int end = -1;
      for (int i = start + 2; i < buf.length - 1; i++) {
        if (buf[i] == 0xFF && buf[i + 1] == 0xD9) { end = i + 1; break; }
      }
      if (end == -1) {
        if (start > 0) buf.removeRange(0, start);
        return;
      }
      final frame = Uint8List.fromList(buf.sublist(start, end + 1));
      buf.removeRange(0, end + 1);

      if (mounted) {
        setState(() {
          _frame = frame;
          _reconnecting = false;
          _retries = 0;
        });
        _startWatchdog(); // 每帧重置超时
      }
    }
  }

  void _startWatchdog() {
    _watchdog?.cancel();
    _watchdog = Timer(const Duration(seconds: _timeoutSecs), _scheduleReconnect);
  }

  void _scheduleReconnect() {
    if (!mounted || _reconnecting) return;
    _reconnecting = true;
    _retries++;

    _watchdog?.cancel();
    _sub?.cancel();
    _client?.close(force: true);
    _sub = null;
    _client = null;

    if (mounted) setState(() {});

    final delay = Duration(seconds: _retries <= 1 ? 1 : _retries <= 2 ? 2 : 4);
    _retryTimer?.cancel();
    _retryTimer = Timer(delay, _connect);
  }

  @override
  Widget build(BuildContext context) {
    if (_reconnecting) {
      return Stack(
        fit: StackFit.expand,
        children: [
          if (_frame != null)
            Image.memory(_frame!, gaplessPlayback: true,
                fit: BoxFit.contain, filterQuality: FilterQuality.low)
          else
            Container(color: const Color(0xFF0D0D1A)),
          Positioned(
            right: 8, bottom: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.65),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(
                    width: 11, height: 11,
                    child: CircularProgressIndicator(
                        strokeWidth: 1.5, color: Color(0xFFFFD740)),
                  ),
                  const SizedBox(width: 5),
                  Text('重连中 $_retries',
                      style: const TextStyle(
                          color: Color(0xFFFFD740), fontSize: 11)),
                ],
              ),
            ),
          ),
        ],
      );
    }

    if (_frame == null) {
      return widget.placeholder ??
          Container(
            color: const Color(0xFF0D0D1A),
            child: const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    width: 24, height: 24,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Color(0xFF00E5FF)),
                  ),
                  SizedBox(height: 8),
                  Text('连接摄像头…',
                      style: TextStyle(color: Color(0xFF555577), fontSize: 13)),
                ],
              ),
            ),
          );
    }

    return Image.memory(
      _frame!,
      gaplessPlayback: true,
      fit: BoxFit.contain,
      filterQuality: FilterQuality.low,
    );
  }
}
