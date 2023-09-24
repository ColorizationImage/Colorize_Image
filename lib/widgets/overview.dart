import 'dart:convert';
import 'dart:typed_data';
import 'package:colorization/models/custom_button.dart';
import 'package:colorization/models/custom_image.dart';
import 'package:colorization/models/uploading.dart';
import 'package:colorization/widgets/before_after.dart';
import 'package:colorization/widgets/filter_screen.dart';
import 'package:colorization/widgets/image_by_network.dart';
import 'package:flutter/material.dart';
import 'package:flutter_expandable_fab/flutter_expandable_fab.dart';
import 'package:image_picker/image_picker.dart';
import 'package:cancellation_token_http/http.dart' as http;
import 'dart:async';
import 'dart:io';
import 'package:path/path.dart' as path;
import 'package:get/get.dart';

class OverviewScreen extends StatefulWidget {
  static const routeName = "/overview";

  const OverviewScreen({super.key});

  @override
  OverviewScreenState createState() => OverviewScreenState();
}

class OverviewScreenState extends State<OverviewScreen> {
  File? selectedImage;
  Uint8List? received;
  Uint8List? temp;
  String? imageName;
  bool isUploading = false;
  late http.CancellationToken token;

  showError() {
    if (mounted) {
      if (ScaffoldMessenger.of(context).mounted) {
        ScaffoldMessenger.of(context).hideCurrentSnackBar();
      }
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        dismissDirection: DismissDirection.endToStart,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        elevation: 5,
        behavior: SnackBarBehavior.fixed,
        backgroundColor: Colors.red.shade800,
        content: const Text(
          'Something went wrong , please try again !',
        ),
      ));
    }
  }

  onUploadImage() async {
    token = http.CancellationToken();
    if (selectedImage == null) {
      return;
    }
    setState(() {
      isUploading = true;
    });
    // 	http://MuhannadT-44221.portmap.io:44221
    // https://colorize-aspu-pieg.onrender.com
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('http://MuhannadT-44221.portmap.io:44221'),
    );
    Map<String, String> headers = {
      "Content-type": "multipart/form-data",
      "Connection": "keep-alive",
    };
    imageName = '${path.basenameWithoutExtension(selectedImage!.path)}_colored';
    // bool isPNG = false;
    // String convertedImage = path.absolute(selectedImage!.path).replaceAll('.png', '');
    // File tempFile = File('$convertedImage.jpg');
    // print(path.current);
    // if(path.basename(selectedImage!.path).endsWith('.png')){
    //   final image = image_converter.decodeImage(File(path.absolute(selectedImage!.path)).readAsBytesSync())!;
    //   tempFile = await File('$convertedImage.jpg').writeAsBytes(image_converter.encodeJpg(image));
    //   // final codec = await ui.instantiateImageCodec(temp!);
    //   // final frame = await codec.getNextFrame();
    //   // final image = frame.image;
    //   // final data = await image.toByteData();
    //   // final jpg = image_converter.JpegEncoder().encode(image_converter.Image.fromBytes(width: 300, height: 300, bytes: data!.buffer));
    //   // temp = jpg;
    //   isPNG = true;
    //   // tempFile = await File('$convertedImage.jpg').writeAsBytes(temp!);
    // }
    request.files.add(
      http.MultipartFile(
        'image',
        selectedImage!.readAsBytes().asStream(),
        selectedImage!.lengthSync(),
        filename: selectedImage!.path.split('/').last,
      ),
    );
    // isPNG ? tempFile.readAsBytes().asStream():
    // isPNG ? tempFile.lengthSync():
    request.headers.addAll(headers);
    await const Duration(seconds: 1).delay();
    await request.send(cancellationToken: token).then((res) async {
      var response = await http.Response.fromStream(res);
      // Test

      // var responseTest = await http.get(Uri.parse(jsonDecode(response.body)['output_url']));
      // // print(jsonDecode(response.body)['output_url']);
      // var response3 = responseTest.bodyBytes;

      // End Test


      var response3 = response.bodyBytes;
      if (response.statusCode == 200) {
        if (response.body == "Failed") {
          showError();
          return;
        }
        received = response3;
        setState(() {
          isUploading = false;
        });
        Get.offAndToNamed(BeforeAfterScreen.routeName,
            arguments: {
              'selectedImage': CustomImage(
                imageBytes: temp!,
              ),
              'receivedImage': CustomImage(
                imageBytes: received!,
              ),
              'imageName': '$imageName'
            });
      } else {
        showError();
        return;
      }
    }).catchError((e) {
      if (token.isCancelled) {
        token = http.CancellationToken();
        return;
      }
      showError();
    }).then((value) {
      if (mounted) {
        setState(() {
          isUploading = false;
        });
      }
    });
  }

  Future<void> getImage(ImageSource imageSource) async {
    await ImagePicker()
        .pickImage(source: imageSource)
        .then((image) async{
          if(image == null){
            return;
          }
      selectedImage = File(image.path);
      temp = await selectedImage!.readAsBytes();
      setState((){});
      onUploadImage();
    });
  }

  @override
  void dispose() {
    super.dispose();
  }

  _onBack(BuildContext context) {
      return showDialog(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Exit'),
            content:
            const Text('Do you want to exit?'),
            actions: <Widget>[
              TextButton(
                  onPressed: () {
                    Navigator.of(context).pop(false);
                  },
                  child:const Text('No')),
              TextButton(
                  onPressed: () {
                    Navigator.of(context).pop(true);
                    // Navigator.of(context).pop(true);
                  },
                  child:const Text('Yes'))
            ],
          ));
  }


  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: ()async {
        return await _onBack(context);
      },
      child: Scaffold(
        appBar: AppBar(
          elevation: 20,
          actions: isUploading
              ? <Widget>[
            TextButton(
                onPressed: () {
                  token.cancel();
                  setState(() {
                    isUploading = false;
                  });
                  return;
                },
                child: const Text(
                  'Cancel',
                  style: TextStyle(color: Colors.white),
                ))
          ]
              : null,
          backgroundColor: Theme.of(context).primaryColor,
          title: const Text('Colorize'),
        ),
        body: isUploading ? const Center(child: Uploading(message: 'Uploading and processing your image ... \n                    Please be patient',)) : Padding(
              padding: const EdgeInsets.symmetric(vertical: 20.0,horizontal: 20.0),
              child: Column(
                children: <Widget>[
                    SizedBox(
                      height: MediaQuery.of(context).size.height / 4,
                      width: MediaQuery.of(context).size.width,
                      child: GridView(
                        padding: const EdgeInsets.all(10.0),
                        gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 1,
                          crossAxisSpacing: 20,
                          mainAxisSpacing: 10,
                        ),
                        children: <Widget>[
                          CustomCard(onTap: (){getImage(ImageSource.gallery);},title: 'Gallery',
                          placeHolderPath: 'assets/images/ph_gallery.png',imgPath: 'assets/images/gallery.png',
                          ),
                          CustomCard(onTap: (){getImage(ImageSource.camera);},title: 'Take photo',
                            placeHolderPath: 'assets/images/ph_camera.jpg',imgPath: 'assets/images/camera.jpg',
                          ),
                        ],
                      ),
                    ),
                  Center(child:CustomCard(onTap: (){Get.offNamedUntil(ImageUploadByNetwork.routeName,(_)=>false);},title: 'Colorize image by URL',
                    placeHolderPath: 'assets/images/ph_www.jpg',imgPath: 'assets/images/www.jpg',
                    ),
                  ),
                ],
              ),
            ),
      ),
    );
  }
}

class CustomTextButton extends StatelessWidget{
  final Function onTap;
  final Icon icon;
  const CustomTextButton({super.key, required this.onTap , required this.icon});
  @override
  Widget build(BuildContext context){
    return TextButton(
          onPressed: (){
            onTap();
          },
          style: ButtonStyle(
            shape: MaterialStatePropertyAll(RoundedRectangleBorder(borderRadius: BorderRadius.circular(30.0))),
            splashFactory: NoSplash.splashFactory,
            padding: MaterialStateProperty.all(EdgeInsets.zero)
          ),
          child: CircleAvatar(
            radius: 30,
            backgroundColor: Theme.of(context).colorScheme.secondary,
            child: icon,
          ),
        );
  }
}

class CustomCard extends StatelessWidget {
  final Function onTap;
  final String title;
  final String placeHolderPath;
  final String imgPath;
  const CustomCard({Key? key, required this.onTap, required this.title, required this.placeHolderPath, required this.imgPath}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 200,
      width: MediaQuery.of(context).size.width / 1.2,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: GridTile(
          footer: GridTileBar(
            backgroundColor: Colors.black87,
            title: Text(
              title,
              textAlign: TextAlign.center,
            ),
          ),
          child: GestureDetector(
            onTap: () {
              onTap();
            },
            child: FadeInImage(
              fadeOutDuration: const Duration(seconds: 2),
              fadeInDuration: const Duration(seconds: 2),
              fit: BoxFit.fill,
              placeholder: AssetImage(placeHolderPath),
              image: Image.asset(imgPath).image,
            ),
          ),
        ),
      ),
    );
  }
}
