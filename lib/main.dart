import 'package:colorization/widgets/before_after.dart';
import 'package:colorization/widgets/filter_screen.dart';
import 'package:colorization/widgets/image_by_network.dart';
import 'package:colorization/widgets/intro.dart';
import 'package:colorization/widgets/overview.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';

void main() async {
  await GetStorage.init();
  runApp(
      MyApp()
  );
}

class MyApp extends StatelessWidget {
  final config = GetStorage();
  MyApp({super.key});


  @override
  Widget build(BuildContext context) {
    config.writeIfNull('first', true);
    return GetMaterialApp(
      title: 'Colorize',
      debugShowCheckedModeBanner: false,
      // themeMode: darkMode? ThemeMode.dark : themeProvider.currentTheme,
      theme: ThemeData.light().copyWith(primaryColor: const Color(0xFF3F51B5),
          colorScheme: ColorScheme.fromSwatch().copyWith(secondary: const Color(0xFF009688))),
      darkTheme: ThemeData.dark().copyWith(primaryColor: const Color(0xFF303F9F),
          colorScheme: ColorScheme.fromSwatch().copyWith(secondary: const Color(0xFF009688))
      ),
      home:  config.read('first') ? const IntroScreen() : const OverviewScreen(),
      defaultTransition: Transition.fadeIn,
      routes: {
        OverviewScreen.routeName: (ctx) => const OverviewScreen(),
        BeforeAfterScreen.routeName: (ctx) => BeforeAfterScreen(),
        ImageUploadByNetwork.routeName: (ctx) => const ImageUploadByNetwork(),
        // FilterScreen.routeName: (ctx) => FilterScreen(),
      },
    );
  }
}