import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:colorization/widgets/overview.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import 'package:introduction_screen/introduction_screen.dart';

class IntroScreen extends StatefulWidget {
  static const routeName = '/introduction';

  const IntroScreen({super.key});

  @override
  IntroScreenState createState() => IntroScreenState();
}

class IntroScreenState extends State<IntroScreen> {
  @override
  void initState() {
    super.initState();
  }


  final List<TyperAnimatedText> _typeList = [
    TyperAnimatedText('Welcome to the future',
        speed:const Duration(milliseconds: 100),
        textStyle:const TextStyle(fontSize: 23.0,fontWeight: FontWeight.bold,color: Colors.black)),
    TyperAnimatedText('Where AI becomes more effective',
        speed:const Duration(milliseconds: 100),
        textStyle:const TextStyle(fontSize: 23.0,fontWeight: FontWeight.bold,color: Colors.black)),
    TyperAnimatedText('So you can use it in amazing way',
        speed:const Duration(milliseconds: 100),
        textStyle:const TextStyle(fontSize: 23.0,fontWeight: FontWeight.bold,color: Colors.black)),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 10,
        backgroundColor: const Color(0xFF303F9F),
        foregroundColor: Colors.white,
        centerTitle: true,
        title:const Text('Colorize',style: TextStyle(fontSize: 25,fontWeight: FontWeight.bold),),
      ),
      body: IntroductionScreen(
        globalBackgroundColor: Colors.white,
          autoScrollDuration: 5000,
          pages: [
            PageViewModel(
              titleWidget: AnimatedTextKit(animatedTexts: _typeList,
                isRepeatingAnimation: true,totalRepeatCount: 5,),
              bodyWidget: buildImage("assets/images/co1.png"),
              //getPageDecoration, a method to customise the page style
              decoration: getPageDecoration(),
            ),
            PageViewModel(
              titleWidget: AnimatedTextKit(animatedTexts: _typeList,
                isRepeatingAnimation: true,totalRepeatCount: 5,),
              bodyWidget: buildImage("assets/images/co2.jpg"),
              //getPageDecoration, a method to customise the page style
              decoration: getPageDecoration(),
            ),
          ],
          onDone: () async {
            await GetStorage().write('first', false);
            Get.offAllNamed(OverviewScreen.routeName);
          },
          //ClampingScrollPhysics prevent the scroll offset from exceeding the bounds of the content.
          scrollPhysics: const ClampingScrollPhysics(),
          showDoneButton: true,
          showNextButton: true,
          showSkipButton: true,
          skip:
              const Text("Skip", style: TextStyle(fontWeight: FontWeight.w600,color: Colors.black,fontSize: 19.0)),
          next: const Icon(Icons.forward,color: Colors.black,size: 30.0,),
          done:
              const Text("Done", style: TextStyle(fontWeight: FontWeight.w600,color: Colors.black,fontSize: 19.0)),
          dotsDecorator: getDotsDecorator()),
    );
  }

  //widget to add the image on screen
  Widget buildImage(String imagePath) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20.0), //add border radius
      child: Image(fit: BoxFit.fill,
            image: ResizeImage(
                allowUpscaling: true,
                width: (MediaQuery.of(context).size.width).toInt(),
                height: MediaQuery.of(context).size.height ~/ 1.5,
                Image.asset(imagePath).image)),
    );
  }

  //method to customise the page style
  PageDecoration getPageDecoration() {
    return const PageDecoration(
      pageColor: Colors.white,
      titlePadding: EdgeInsets.only(bottom: 10),
      bodyTextStyle: TextStyle(color: Colors.black, fontSize: 15),
    );
  }

  //method to customize the dots style
  DotsDecorator getDotsDecorator() {
    return const DotsDecorator(
      spacing: EdgeInsets.symmetric(horizontal: 2),
      activeColor: Colors.indigo,
      color: Colors.grey,
      activeSize: Size(30, 15),
      activeShape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(25.0)),
      ),
    );
  }
}
