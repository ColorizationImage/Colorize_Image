import 'package:flutter/material.dart';
import 'dart:ui' as ui;
class MyPainter extends CustomPainter {

  final ui.Image image;
  final ColorFilter filter;
  final Size screenSize;
  MyPainter({required this.image,required this.filter,required this.screenSize});

  @override
  void paint(Canvas canvas, Size size) {

    double width = size.width > screenSize.width ? size.width / 2 : size.width;
    double height = size.height > screenSize.height/2 ? size.height / 2 : size.height;

    print("Canvas width = $width");
    print("Canvas height = ${size.height}");

    bool isHeightLarger = height > width;
    double radius = isHeightLarger ? height*0.5 : width*0.5;
    Path path = Path()
      ..addOval(Rect.fromCircle(center: Offset(width, height), radius: radius));

    canvas.clipPath(path);
    canvas.drawImage(image, Offset(0.0,0.0), Paint()..colorFilter = filter);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return false;
  }

}