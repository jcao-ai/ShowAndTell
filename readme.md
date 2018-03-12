![ShowAndTell](https://github.com/litleCarl/ShowAndTell/blob/master/DemoImages/showAndTell.png)

<p align="center">
<a href="https://developer.apple.com/swift"><img src="https://img.shields.io/badge/language-swift4-f48041.svg?style=flat"></a>
<a href="https://developer.apple.com/ios"><img src="https://img.shields.io/badge/platform-iOS%2011%2B-blue.svg?style=flat"></a>
<a href="https://github.com/wxxsw/GSMessages/tree/1.0.0"><img src="https://img.shields.io/badge/release-1.0.0-blue.svg"></a>
</p>

# Show and Tell: A Neural Image Caption Generator 
> 🎉🎉🎉 Keras part is public now
## Brief

***Pull requests and issues:*** 
@litleCarl

A CoreML implementation of the image-to-text model described in the paper:

"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning
Challenge."

Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.

*IEEE transactions on pattern analysis and machine intelligence (2016).*

Full text available at: http://arxiv.org/abs/1609.06647

## Demo
<img src="https://github.com/LitleCarl/ShowAndTell/blob/master/DemoImages/demo_5.png" width="150" ><img src="https://github.com/LitleCarl/ShowAndTell/blob/master/DemoImages/demo_2.png" width="150" ><img src="https://github.com/LitleCarl/ShowAndTell/blob/master/DemoImages/demo_3.png" width="150" ><img src="https://github.com/LitleCarl/ShowAndTell/blob/master/DemoImages/demo_4.png" width="150" ><img src="https://github.com/LitleCarl/ShowAndTell/blob/master/DemoImages/demo_6.png" width="150" >


## Usage

### Simple use
```Swift
let showAndTell = ShowAndTell()
let results = showAndTell.predict(image: uiimage2predict, beamSize: 3, maxWordNumber: 30)
```



```Swift
// Parameter explaination
//    image:         The image to be used to generate the caption.
//    beamSize:      Max caption count in result to be reserved in beam search.(Affect the performance greatly)
//    maxWordNumber: Max number of words in a sentence to be predicted.
class ShowAndTell {
  ...
  func predict(image: UIImage, beamSize: Int = 3, maxWordNumber:Int = 20) -> PriorityQueue<Caption>
  ...
}
```

## Benchmark (Tested on iPhone 7+, welcome PR for more devices)
<table>
<tr><th>maxWordNumber = 20 </th><th>maxWordNumber = 30</th></tr>
<tr><td>

beamSize | Time (ms)
---- | ---
1  | 480.12
2  | 845.78
3  | 1443.82
4  | 2001.30
5  | 2648.48
6  | 3158.53
7  | 4179.14
8  | 4861.66
9  | 6003.65
10 | 7087.97
11 | 8134.95
12 | 9627.79

</td><td>

beamSize | Time (ms)
---- | ---
1 | 451.12
2 | 1194.65
3 | 1965.27
4 | 2971.92
5 | 3798.28
6 | 4391.35
7 | 5714.87
8 | 6937.60
9 | 8482.03
10 | 10421.52
11 | 12460.80
12 | 13777.67
</td></tr> </table>

<img src="https://github.com/LitleCarl/ShowAndTell/blob/master/DemoImages/chart_of_beam_size" >

Line chart for Time vs Beam Size (When `maxWordNumber = 30`)

So it is recommeneded to set `beamSize=1` on mobile devices due to less gpu/cpu time usage for saving battery life.  

## Requirements
- iOS 11.0+
- Xcode 9.0+ (Swift 4.x)

## Original Model
This coreml model is exported from keras which is trained with MSCOCO dataset for about 40k steps. And presently it is not in the state of art yet. You may not use this in production.
I trained the dataset with only one GTX Force 1080Ti for about 48 hours and currently don't have more time to train on it.Hope for community to keep on it.


## Keras part
- Train 
  ``` cd keras_part;
      python ./train.py --weight_path WEIGHT_FILE_PATH_TO_CONTINUE_TRAINING  --TFRecord_pattern TFRECORD_FILE_PATTERN
   ```
    For example:
  ```
      python ./train.py --weight_path ./keras_weight/weights_full.h5  --TFRecord_pattern ./tfrecords/train-?????-of-00256
  ```
- Test 
  ``` cd keras_part;
      python ./inference.py --weight_path WEIGHT_FILE_PATH  --image_path TEST_IMAGE_PATH --max_sentence_length 20
  ```
    For example:
  ```
      python ./inference.py --weight_path ./keras_weight/weights_full.h5  --image_path ./test.jpg --max_sentence_length 20
  ```
- Convert to CoreML Model 
  ``` cd keras_part;
      python ./convert_coreml.py --export_lstm False
  ```
  ```export_lstm ``` determine whether to export the inception part or lstm part model.(The whole model is split into 2 parts. One for image encoding, one for decoding words)
### Pretained Weight
Pretained Keras weight file will be uploaded to google driver in short time.
### Training dataset
We use MS-COCO dataset, you can fetch raw data and build them into tfrecords according to the origin tensorflow [im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)


## TODO 
- Train on the dataset to 100k steps. (currently 40k)
- ~~Open source origin model based on Keras which is trained with.~~
- More language support (Chinese).

## Thanks for third party lib in demo
- [GSMessages](https://github.com/wxxsw/GSMessages)
- [Swift-PriorityQueue](https://github.com/Bouke/Swift-PriorityQueue/)

## Contact
- 曹佳鑫 （tsao）An iOS developer with experience in deep learning living in Shanghai.
- Pull requests and issues are welcome.
- Mail: cjx5813@foxmail.com

## License

ShowAndTell is available under the MIT license. See the LICENSE file for more info.
