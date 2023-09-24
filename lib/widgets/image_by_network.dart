import 'dart:convert';
import 'dart:typed_data';
import 'package:colorization/models/uploading.dart';
import 'package:colorization/widgets/overview.dart';
import 'package:flutter/material.dart';
import 'package:cancellation_token_http/http.dart' as http;
import 'package:get/get.dart';
import '../models/custom_button.dart';
import '../models/custom_image.dart';
import 'before_after.dart';
class ImageUploadByNetwork extends StatefulWidget {
  static const routeName = "/image-by-network";
  const ImageUploadByNetwork({super.key});

  @override
  ImageUploadByNetworkState createState() => ImageUploadByNetworkState();
}
class ImageUploadByNetworkState extends State<ImageUploadByNetwork> {
  bool _isProcessing = false;
  bool _isLoading = true;
  final _form = GlobalKey<FormState>();
  final TextEditingController imageUrlController = TextEditingController();
  Uint8List? imageBytes;
  late http.CancellationToken token;

  Future<void> getImageFromUrl(String imageUrl) async {
    if (imageUrlController.text.isEmpty) {
      return;
    }
    token = http.CancellationToken();
    setState(() {
      _isLoading = true;
      _isProcessing = true;
    });
    try {
      var response = await http.get(Uri.parse(imageUrl), cancellationToken: token);
      if (response.statusCode == 200) {
        setState(() {
          imageBytes = response.bodyBytes;
          _isLoading = false;
        });
      } else {
        print('Failed to load image: ${response.statusCode}');
      }
    }catch(e){
      if (token.isCancelled) {
        token = http.CancellationToken();
        return;
      }
      return;
    }finally{
      setState(() {
        _isProcessing = false;
      });
    }
  }
  Future<void> uploadImage() async {
    if(imageBytes == null){
      return;
    }
    setState(() {
      _isProcessing = true;
    });
    // MuhannadT-44221.portmap.io:44221


    //Development

    // var urlTest = Uri.parse('https://api.deepai.org/api/colorizer');
    // var request = http.MultipartRequest('POST', urlTest);
    // Map<String, String> headers = {
    //   "Content-type": "multipart/form-data",
    //   "Connection": "keep-alive",
    //   'api-key': '5cf50c0b-a221-472b-b042-ec954515f860'
    // };

    // End Development

    var url = Uri.parse('http://MuhannadT-44221.portmap.io:44221');
    String imageName = Uri.parse(imageUrlController.text).pathSegments.last;
    var request = http.MultipartRequest('POST', url);
    request.files.add(http.MultipartFile.fromBytes(
      'image',
      imageBytes!,
      filename: imageName,
    ));
    Map<String, String> headers = {
      "Content-type": "multipart/form-data",
      "Connection": "keep-alive",
    };
    request.headers.addAll(headers);
    try{
      var response = await request.send();
      if (response.statusCode == 200) {

        // Development

        // var responseTest = await http.Response.fromStream(response);
        // var responseTest2 = await http.get(Uri.parse(jsonDecode(responseTest.body)['output_url']));
        // // print(jsonDecode(response.body)['output_url']);
        // var result = responseTest2.bodyBytes;

        // End Development

        var result = await response.stream.toBytes();
        setState(() {
          _isProcessing = false;
        });
        Get.offAndToNamed(BeforeAfterScreen.routeName,
            arguments: {
              'selectedImage': CustomImage(
                imageBytes: imageBytes!,
              ),
              'receivedImage': CustomImage(
                imageBytes: result,
              ),
              'imageName': '${imageName}_colored'
            });
      } else {
        return;
        // print('Failed to upload image: ${response.statusCode}');
      }
    }catch(e){
      // print('Failed to upload image.');
      return;
    }finally{
      if(mounted){
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }
  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async{
        Get.off(()=> const OverviewScreen());
        return true;
      },
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).primaryColor,
          leading: !_isProcessing ? IconButton(
              iconSize: 30.0,
              splashRadius: 25.0,
              onPressed: (){
                Get.off(()=> const OverviewScreen());
              },
              icon: const Icon(Icons.arrow_back, color: Colors.white,)
          ):null,
          actions: _isProcessing ? <Widget>[
                  TextButton(
                      onPressed: () {
                        token.cancel();
                        setState(() {
                          _isProcessing = false;
                        });
                        return;
                      },
                      child: const Text(
                        'Cancel',
                        style: TextStyle(color: Colors.white),
                      ))
                ]
              : null,
        ),
        body: _isProcessing ? Center(child: Uploading(message: _isLoading ? 'Loading Your Image':'Uploading and processing your image ... \n                    Please be patient')) : Column(
          children: [
            Align(
              alignment: Alignment.center,
              child: Container(
                margin: const EdgeInsets.only(top: 20),
                width: MediaQuery.of(context).size.width / 1.2,
                child: TextFormField(
                  key: _form,
                  decoration: InputDecoration(labelText: 'Enter image URL',border: OutlineInputBorder(borderRadius: BorderRadius.circular(20.0))),
                  textInputAction: TextInputAction.done,
                  keyboardType: TextInputType.url,
                  controller: imageUrlController,
                ),
              ),
            ),
            const SizedBox(height: 16),
            CustomButton(
              onTap: () async{
              await getImageFromUrl(imageUrlController.text);
            }, title: 'Load Image',),
            const SizedBox(height: 16),
            if (imageBytes != null)
              Image.memory(
                imageBytes!,
                height: 200,
                fit: BoxFit.cover,
              ),
            const SizedBox(height: 16),
            ElevatedButton(
              style: ButtonStyle(
                backgroundColor: MaterialStateProperty.all(Theme.of(context).colorScheme.secondary)
              ),
              onPressed: () async{
                await getImageFromUrl(imageUrlController.text);
              },
              child: const Text('Upload Image'),
            ),
          ],
        ),
      ),
    );
  }
}