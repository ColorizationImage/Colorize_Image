import 'dart:typed_data';
import 'package:flutter/material.dart';

class CustomImage extends StatelessWidget {
  final Uint8List imageBytes;
  const CustomImage({Key? key, required this.imageBytes}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final width = (MediaQuery.of(context).size.width) ~/ 1.1;
    final height = (MediaQuery.of(context).size.height) ~/ 1.5;
    return Image(image: ResizeImage(Image.memory(imageBytes).image,height: height , width: width,allowUpscaling: true),);
  }
}
