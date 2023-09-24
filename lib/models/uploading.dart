import 'dart:async';
import 'dart:math';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:flutter/material.dart';

class Uploading extends StatefulWidget {
  final String message;
  const Uploading({Key? key, required this.message}) : super(key: key);

  @override
  _UploadingState createState() => _UploadingState();
}

class _UploadingState extends State<Uploading> with SingleTickerProviderStateMixin{

  final Random _random = Random();
  double _width = 100.0;
  double _height = 100.0;
  BorderRadiusGeometry _borderRadiusGeometry = BorderRadius.circular(10.0);
  Color _color = Color.fromRGBO(50, 50, 50, 1);
  late Timer _timer;


  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(Duration(seconds: 1), (timer) { update(); });
    // print(_timer.isActive);
  }

  void update(){
    final newWidth = _random.nextInt(200).toDouble();
    final newHeight = _random.nextInt(200).toDouble();
    final newRadius = _random.nextInt(30).toDouble();
    final newColorR = _random.nextInt(256);
    final newColorG = _random.nextInt(256);
    final newColorB = _random.nextInt(256);
    setState(() {
      _width = newWidth;
      _height = newHeight;
      _borderRadiusGeometry = BorderRadius.circular(newRadius);
      _color = Color.fromRGBO(newColorR, newColorG, newColorB, 1);
    });
  }


  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.max,
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            height: 220,
            width: 200,
            child: SingleChildScrollView(
              child: Center(
                child: AnimatedContainer(
                  duration: const Duration(seconds: 1),
                    decoration: BoxDecoration(
                      borderRadius: _borderRadiusGeometry,
                      color: _color
                    ),
                    width: _width,
                    height: _height,
                ),
              ),
            ),
          ),
          AnimatedTextKit(
            animatedTexts: [TyperAnimatedText(widget.message,
              speed:const Duration(milliseconds: 100),
              textStyle:const TextStyle(fontSize: 20.0,fontWeight: FontWeight.bold,
                  color: Colors.black))],
            isRepeatingAnimation: true,repeatForever: true,)
        ],
      ),
    );
  }
}