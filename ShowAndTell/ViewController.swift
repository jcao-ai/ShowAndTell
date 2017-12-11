//
//  ViewController.swift
//  TestML
//
//  Created by Tsao on 2017/12/5.
//  Copyright © 2017年 Tsao. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
    @IBOutlet weak var textView: UITextView!

    @IBOutlet weak var imageView: UIImageView!
   
    let showAndTell = ShowAndTell()
    var currentImage: UIImage = UIImage(named: "COCO_train2014_000000005340.jpg")! {
        didSet {
            self.imageView.image = currentImage
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    @IBAction func switchImage() {
        var imgs = [ "COCO_train2014_000000005303.jpg",
          "COCO_train2014_000000005336.jpg",
          "COCO_train2014_000000005359.jpg",
          "COCO_train2014_000000005377.jpg",
          "COCO_train2014_000000005434.jpg",
          "COCO_train2014_000000005472.jpg",
          "COCO_train2014_000000005312.jpg",
          "COCO_train2014_000000005339.jpg",
          "COCO_train2014_000000005360.jpg",
          "COCO_train2014_000000005383.jpg",
          "COCO_train2014_000000005435.jpg",
          "COCO_train2014_000000005482.jpg",
          "COCO_train2014_000000005313.jpg",
          "COCO_train2014_000000005340.jpg",
          "COCO_train2014_000000005362.jpg",
          "COCO_train2014_000000005396.jpg",
          "COCO_train2014_000000005453.jpg",
          "COCO_train2014_000000005483.jpg",
          "COCO_train2014_000000005324.jpg",
          "COCO_train2014_000000005344.jpg",
          "COCO_train2014_000000005368.jpg",
          "COCO_train2014_000000005424.jpg",
          "COCO_train2014_000000005459.jpg",
          "COCO_train2014_000000005500.jpg",
          "COCO_train2014_000000005326.jpg",
          "COCO_train2014_000000005345.jpg",
          "COCO_train2014_000000005373.jpg",
          "COCO_train2014_000000005425.jpg",
          "COCO_train2014_000000005469.jpg",
          "COCO_train2014_000000005505.jpg",
          "COCO_train2014_000000005335.jpg",
          "COCO_train2014_000000005355.jpg",
          "COCO_train2014_000000005376.jpg",
          "COCO_train2014_000000005430.jpg",
          "COCO_train2014_000000005471.jpg" ]
        let random = Int(arc4random_uniform(UInt32(imgs.count)))
        self.currentImage = UIImage(named:imgs[random])!
    }
    
    @IBAction func predict() {
        textView.text = nil
        let startTime = Date()
        let results = showAndTell.predict(image: self.currentImage, beamSize: 7, maxWordNumber: 30)
        GSMessage.showMessageAddedTo("Time elapsed：\(Date().timeIntervalSince(startTime) * 1000)ms", type: .info, options: nil, inView: self.view, inViewController: self)
        textView.text = results.sorted(by: {$0.score > $1.score}).map({
            var x = $0.readAbleSentence.suffix($0.readAbleSentence.count - 1)
            if $0.sentence.last == Caption.endID {
                _ = x.removeLast()
            }
            return String.init(format: "Probability:%.3f‱ \n \(x.joined(separator: " ").capitalizingFirstLetter())", pow(2, $0.score) * 10000.0)
        }).joined(separator: "\n\n")
    }
    
    @IBAction func takePhoto(_ sender: Any) {
        self.getCameraOn(self, canEdit: false)
    }
    
    func getCameraOn(_ onVC: UIViewController, canEdit: Bool) {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self as! UIImagePickerControllerDelegate & UINavigationControllerDelegate
            imagePicker.sourceType = .camera;
            imagePicker.allowsEditing = false
            self.present(imagePicker, animated: true, completion: nil)
        }
    }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        let image = info[UIImagePickerControllerOriginalImage] as! UIImage
        self.currentImage = image
        picker.dismiss(animated: true, completion: nil)
    }
}

extension String {
    func substring(_ from: Int) -> String {
        let start = index(startIndex, offsetBy: from)
        return String(self[start ..< endIndex])
    }
    
    func capitalizingFirstLetter() -> String {
        return prefix(1).uppercased() + dropFirst()
    }
    
    mutating func capitalizeFirstLetter() {
        self = self.capitalizingFirstLetter()
    }
}
