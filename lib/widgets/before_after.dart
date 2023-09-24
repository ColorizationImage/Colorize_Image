import 'dart:core';
import 'dart:typed_data';
import 'package:colorization/models/custom_button.dart';
import 'package:colorization/widgets/filter_screen.dart';
import 'package:colorization/widgets/overview.dart';
import 'package:get/get.dart';
import 'package:path/path.dart' as path;
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:colorization/models/custom_image.dart';
import 'package:flutter/material.dart';
import 'package:before_after_image_slider_nullsafty/before_after_image_slider_nullsafty.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:share_plus/share_plus.dart';
import 'dart:io';

class BeforeAfterScreen extends StatelessWidget{
  static const routeName = 'before-after';
  CustomImage? selectedImage;
  CustomImage? receivedImage;
  String? imageName;
  String? imagePath;
  bool _isSaved = false;

  BeforeAfterScreen({super.key});

  void showMessage(BuildContext context , String message , Color color){
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        dismissDirection: DismissDirection.endToStart,
        shape:
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        elevation: 5,
        behavior: SnackBarBehavior.floating,
        backgroundColor: color,
        content: Text(
          message,
        ),
      ));
  }

  Future<void> _saveImage(BuildContext context) async {
    String message = 'Image saved successfully !';
    Color color = Colors.green.shade600;
    if (await Permission.storage.isDenied) {
      await Permission.storage.request();
    }
    if(!_isSaved){
      try {
        var temp = await ImageGallerySaver.saveImage(receivedImage!.imageBytes,name: imageName,quality: 100 , isReturnImagePathOfIOS: true);
        _isSaved = true;
        imagePath = path.absolute(temp['filePath']).replaceAll('%20', ' ').replaceFirst('/file://', '');
      } catch (e) {
        message = 'Something went wrong, try again.';
        color = Colors.red.shade600;
        return;
      }finally{
        if(context.mounted){
          showMessage(context, message, color);
        }
      }
    }else{
      return;
    }
  }

  _onBack(BuildContext context) {
    if (_isSaved) {
      Get.offNamedUntil(OverviewScreen.routeName,(_)=> false);
    } else {
      return showDialog(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Warning !'),
            content:
            const Text('Your work will be lost.Are you sure to get back?'),
            actions: <Widget>[
              TextButton(
                  onPressed: () {
                    Navigator.of(context).pop(false);
                  },
                  child:const Text('No')),
              TextButton(
                  onPressed: () {
                    Navigator.of(context).pop(true);
                    Navigator.pushNamedAndRemoveUntil(
                        context, OverviewScreen.routeName, (_) => false);
                  },
                  child:const Text('Yes'))
            ],
          ));
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = ModalRoute.of(context)!.settings.arguments
        as Map<String, dynamic>; // Gets arguments from navigator
    selectedImage = data['selectedImage'];
    receivedImage = data['receivedImage'];
    imageName = data['imageName'];
    return Scaffold(
        appBar: AppBar(
          leading: IconButton(
            iconSize: 30.0,
              splashRadius: 25.0,
              onPressed: (){
              _onBack(context);
              },
              icon: const Icon(Icons.arrow_back, color: Colors.white,)
          ),
          actions: <Widget>[
            PopupMenuButton(itemBuilder: (_) => [
              PopupMenuItem(
                onTap: () async{
                    if (!_isSaved) {
                       await _saveImage(context);
                       _isSaved = true;
                    }
                    print(imageName);
                    // await Share.shareXFiles([XFile(imagePath!, bytes: receivedImage!.imageBytes)]);
                    await Share.shareXFiles([XFile.fromData(receivedImage!.imageBytes, name: imageName , mimeType: 'jpg')]);
                    return;
                    },
                padding: EdgeInsets.zero,
                child: Container(
                  width: 120,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: const [
                      // SizedBox(width: 1,),
                      Icon(Icons.share ,
                      color: Colors.black,
                      size: 25.0,),
                      Text('Share'),
                    ],
                  ),
                ),
              ),
            ]),
            // IconButton(onPressed: () async{
            //   if(!_isSaved){
            //     await _saveImage(context);
            //   }
            //   await Share.shareXFiles([XFile(imagePath!,bytes: receivedImage!.imageBytes)]);
            //   return;
            // }, icon: const Icon(
            //   Icons.share,
            //   color: Colors.white,
            //   size: 25.0,
            // )),
            // IconButton(
            //     onPressed: () {
            //       _saveImage(context);
            //     },
            //     icon: const Icon(
            //       Icons.save,
            //       color: Colors.white,
            //       size: 25.0,
            //     )),
          ],
          title: const Text('Done'),
          centerTitle: true,
          backgroundColor: Theme.of(context).primaryColor,
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            BeforeAfter(
              imageCornerRadius: 10.0,
              beforeImage: selectedImage as Widget,
              afterImage: receivedImage as Widget,
              thumbRadius: 7.0,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                CustomButton(onTap: (){
                  _saveImage(context);
                  return;
                }, title: 'Download'),
                CustomButton(onTap: (){
                  Navigator.of(context).push(
                      MaterialPageRoute(builder: (ctx) => FilterScreen(imgBytes: receivedImage!.imageBytes))
                  );
                }, title: 'Edit'),
              ],
            ),
          ],
        ));
  }
}
