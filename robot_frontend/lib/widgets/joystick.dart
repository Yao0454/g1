import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';

typedef JoystickCallback = void Function(double x, double y);

/// 虚拟摇杆：GestureDetector 驱动，松手回中，100ms 连续回调。
class Joystick extends StatefulWidget {
  final JoystickCallback onChanged;
  final double size;
  final bool enabled;

  const Joystick({
    super.key,
    required this.onChanged,
    this.size = 160,
    this.enabled = true,
  });

  @override
  State<Joystick> createState() => _JoystickState();
}

class _JoystickState extends State<Joystick> {
  Offset _stick = Offset.zero;
  bool _active = false;
  Timer? _ticker;

  double get _outer => widget.size / 2;
  double get _thumb => widget.size * 0.21;
  double get _maxTravel => _outer - _thumb - 4;

  Offset _clamp(Offset delta) {
    final d = delta.distance;
    return d > _maxTravel ? delta * (_maxTravel / d) : delta;
  }

  void _update(Offset local) {
    final clamped = _clamp(local - Offset(_outer, _outer));
    setState(() => _stick = clamped);
    _emit();
  }

  void _emit() {
    if (_maxTravel <= 0) return;
    widget.onChanged(_stick.dx / _maxTravel, -_stick.dy / _maxTravel);
  }

  void _startTicker() {
    _ticker?.cancel();
    _ticker = Timer.periodic(const Duration(milliseconds: 80), (_) {
      if (_active) _emit();
    });
  }

  void _stopTicker() {
    _ticker?.cancel();
    _ticker = null;
  }

  @override
  void dispose() {
    _stopTicker();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final inner = SizedBox(
      width: widget.size,
      height: widget.size,
      child: CustomPaint(
        painter: _Painter(
          stick: _stick,
          outer: _outer,
          thumb: _thumb,
          maxTravel: _maxTravel,
          active: _active,
        ),
      ),
    );

    if (!widget.enabled) {
      return Opacity(opacity: 0.35, child: inner);
    }

    return GestureDetector(
      onPanStart: (d) {
        _active = true;
        _update(d.localPosition);
        _startTicker();
      },
      onPanUpdate: (d) => _update(d.localPosition),
      onPanEnd: (_) {
        _active = false;
        _stopTicker();
        setState(() => _stick = Offset.zero);
        widget.onChanged(0, 0);
      },
      onPanCancel: () {
        _active = false;
        _stopTicker();
        setState(() => _stick = Offset.zero);
        widget.onChanged(0, 0);
      },
      child: inner,
    );
  }
}

class _Painter extends CustomPainter {
  final Offset stick;
  final double outer;
  final double thumb;
  final double maxTravel;
  final bool active;

  static const _cyan = Color(0xFF00E5FF);
  static const _bg = Color(0xFF111128);

  const _Painter({
    required this.stick,
    required this.outer,
    required this.thumb,
    required this.maxTravel,
    required this.active,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(outer, outer);

    // 底盘
    canvas.drawCircle(c, outer * 0.96,
        Paint()..color = _bg);
    canvas.drawCircle(
        c, outer * 0.96,
        Paint()
          ..color = _cyan.withValues(alpha:0.25)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5);

    // 十字准线
    final lp = Paint()
      ..color = _cyan.withValues(alpha:0.12)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(outer, outer * 0.15), Offset(outer, outer * 1.85), lp);
    canvas.drawLine(Offset(outer * 0.15, outer), Offset(outer * 1.85, outer), lp);

    // 死区圆
    canvas.drawCircle(
        c, outer * 0.22,
        Paint()
          ..color = _cyan.withValues(alpha:0.08)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1);

    // 方向指示弧（当摇杆偏移时）
    if (stick.distance > 4) {
      final angle = atan2(stick.dy, stick.dx);
      final r = outer * 0.65;
      final arcPaint = Paint()
        ..color = _cyan.withValues(alpha:0.35)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round;
      canvas.drawArc(
        Rect.fromCircle(center: c, radius: r),
        angle - 0.35,
        0.7,
        false,
        arcPaint,
      );
    }

    // 摇杆球（光晕 + 主体 + 高光）
    final tp = c + stick;
    if (active) {
      canvas.drawCircle(tp, thumb * 1.6,
          Paint()..color = _cyan.withValues(alpha:0.12));
    }
    canvas.drawCircle(tp, thumb,
        Paint()..color = active ? _cyan.withValues(alpha:0.85) : _cyan.withValues(alpha:0.55));
    canvas.drawCircle(tp, thumb,
        Paint()
          ..color = _cyan
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.8);
    canvas.drawCircle(tp + const Offset(-3, -3), thumb * 0.28,
        Paint()..color = Colors.white.withValues(alpha:0.5));
  }

  @override
  bool shouldRepaint(_Painter o) =>
      o.stick != stick || o.active != active;
}
