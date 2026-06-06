import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// MJPEG 摄像头画面。
///
/// stream.py 返回的格式：
///   --frame\r\nContent-Type: image/jpeg\r\n\r\n<JPEG>\r\n
/// 通过 JPEG 魔数 FF D8 … FF D9 直接提取帧，不依赖 boundary 解析。
class MjpegView extends StatefulWidget {
  final String url;
  final Widget? placeholder;

  const MjpegView({super.key, required this.url, this.placeholder});

  @override
  State<MjpegView> createState() => _MjpegViewState();
}

class _MjpegViewState extends State<MjpegView> {
  Uint8List? _frame;
  bool _error = false;
  StreamSubscription<Uint8List>? _sub;
  http.Client? _client;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void didUpdateWidget(MjpegView old) {
    super.didUpdateWidget(old);
    if (old.url != widget.url) {
      _disconnect();
      _connect();
    }
  }

  void _connect() {
    if (widget.url.isEmpty) return;
    _error = false;
    _client = http.Client();
    _sub = _mjpegFrames(widget.url, _client!).listen(
      (frame) {
        if (mounted) setState(() => _frame = frame);
      },
      onError: (_) {
        if (mounted) setState(() => _error = true);
      },
      onDone: () {
        if (mounted) setState(() => _error = true);
      },
    );
  }

  void _disconnect() {
    _sub?.cancel();
    _client?.close();
    _sub = null;
    _client = null;
  }

  @override
  void dispose() {
    _disconnect();
    super.dispose();
  }

  /// 从 MJPEG HTTP 流里持续提取 JPEG 帧（FF D8 … FF D9）。
  static Stream<Uint8List> _mjpegFrames(String url, http.Client client) async* {
    try {
      final req = http.Request('GET', Uri.parse(url));
      req.headers['Cache-Control'] = 'no-cache';
      final resp = await client.send(req).timeout(const Duration(seconds: 5));

      final buf = <int>[];

      await for (final chunk in resp.stream) {
        buf.addAll(chunk);

        while (true) {
          // 找 JPEG 起始 FF D8
          int start = -1;
          for (int i = 0; i < buf.length - 1; i++) {
            if (buf[i] == 0xFF && buf[i + 1] == 0xD8) {
              start = i;
              break;
            }
          }
          if (start == -1) {
            if (buf.length > 1) buf.removeRange(0, buf.length - 1);
            break;
          }
          // 找 JPEG 结束 FF D9
          int end = -1;
          for (int i = start + 2; i < buf.length - 1; i++) {
            if (buf[i] == 0xFF && buf[i + 1] == 0xD9) {
              end = i + 1;
              break;
            }
          }
          if (end == -1) {
            if (start > 0) buf.removeRange(0, start);
            break;
          }
          yield Uint8List.fromList(buf.sublist(start, end + 1));
          buf.removeRange(0, end + 1);
        }
      }
    } catch (_) {
      // 连接失败由 onError 处理
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error) {
      return Container(
        color: const Color(0xFF0D0D1A),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.videocam_off_rounded,
                  color: Colors.red.shade700, size: 40),
              const SizedBox(height: 8),
              const Text('摄像头不可用',
                  style: TextStyle(color: Color(0xFF555577), fontSize: 13)),
            ],
          ),
        ),
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
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Color(0xFF00E5FF),
                    ),
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
