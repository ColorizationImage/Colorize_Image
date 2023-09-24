import 'package:flutter/material.dart';

class CustomButton extends StatelessWidget {
  final Function onTap;
  final Alignment alignment;
  final String title;

  const CustomButton({super.key,
    required this.onTap,
    required this.title,
    this.alignment=Alignment.center});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignment,
      child: Container(
        height: 35,
        width: 100,
        child: TextButton(
          style: TextButton.styleFrom(backgroundColor:Theme.of(context).colorScheme.secondary),
          onPressed: (){
            onTap();
          },
          child: Text(
            title,
            style: const TextStyle(color: Colors.white , fontWeight: FontWeight.w600),
          ),
        ),
      ),
    );
  }
}
