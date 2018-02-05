//
//  ShowAndTell.swift
//  TestML
//
//  Created by Tsao on 2017/12/11.
//  Copyright © 2017年 Tsao. All rights reserved.
//

import Foundation
import UIKit
import CoreML

// Initialize CoreML models once
let inception = keras_inception_lstm_good().fritz()
let inceptionWOLabel = keras_inception_step_good_without_label().fritz()

class ShowAndTell {
    let wordDict: [String]
    required init() {
        if let path = Bundle.main.path(forResource: "word_counts", ofType: "txt") {
            do {
                let data = try String(contentsOfFile: path, encoding: .utf8)
                let myStrings = data.components(separatedBy: .newlines)
                self.wordDict = myStrings.map({
                    String($0.split(separator: " ").first ?? "")
                })
            } catch {
                fatalError("Error found when processing word_counts.txt.")
            }
        } else {
            fatalError("File: word_counts.txt not found in bundle.")
        }
    }
    
    //    image:         The image to be used to generate the caption.
    //    beamSize:      Max caption count in result to be reserved in beam search.(Affect the performance greatly)
    //    maxWordNumber: Max number of words in a sentence to be predicted.
    func predict(image: UIImage, beamSize: Int = 3, maxWordNumber:Int = 20) -> PriorityQueue<Caption> {
        let img = image.pixelBuffer(width: 299, height: 299)!
        // 获取图像感知状态
        let result = try! inception.prediction(image: img, lstm_1_h_in: nil, lstm_1_c_in: nil)
        
        let initialCaption = Caption(state: (result.lstm_1_h_out, result.lstm_1_c_out))
        
        let completedCaption = PriorityQueue<Caption>.init(maxLength: beamSize, compare: {$0 > $1})
        let unCompletedCaption = PriorityQueue<Caption>.init(maxLength: beamSize, compare: {$0 > $1})
        unCompletedCaption.push(newElement: initialCaption)
        
        // 最大maxWordNumber个单词, 进行迭代获取下个单词
        for _ in 0..<maxWordNumber {
            // 如果未完成的语句没有了直接退出循环
            if unCompletedCaption.count < 1 {
                break
            }
            
            var newCaptions = [Caption]()
            
            for (_, unCompleted) in unCompletedCaption.enumerated() {
                newCaptions = newCaptions + unCompleted.nextStepWith(beamSize: beamSize, vocabDict: self.wordDict)
            }
            
            assert(unCompletedCaption.count == 0, "unCompletedCaption.count should be zero here.")
            
            newCaptions.filter({$0.isCompleted}).forEach({completedCaption.push(newElement: $0)})
            newCaptions.filter({!$0.isCompleted}).forEach({unCompletedCaption.push(newElement: $0)})
        }
        if unCompletedCaption.count > 0 {
            unCompletedCaption.forEach({completedCaption.push(newElement: $0)})
        }
        return completedCaption
    }
}

struct Word : Comparable{
    let index: Int
    let proba: Double
    
    static func ==(lhs: Word, rhs: Word) -> Bool {
        return lhs.index == rhs.index
    }
    static func <(lhs: Word, rhs: Word) -> Bool {
        return lhs.proba < rhs.proba
    }
}

class Caption: Comparable {
    static func ==(lhs: Caption, rhs: Caption) -> Bool {
        return lhs.sentence == rhs.sentence
    }
    
    static let startID = 2
    static let endID = 3
    
    var isCompleted: Bool {
        get {
            return self.sentence.last == Caption.endID
        }
    }
    
    // 语句id
    var sentence: [Int]
    var readAbleSentence: [String]
    
    // 分数
    var score: Double
    
    // h_state, c_state
    var state: (MLMultiArray, MLMultiArray)
    
    required init(state:(MLMultiArray, MLMultiArray), sentence: [Int] = [Caption.startID], score: Double = 0, readableSentence: [String] = ["<S>"]) {
        self.sentence = sentence
        self.readAbleSentence = readableSentence
        self.score = score
        self.state = state
    }
    
    func lastWordMatrix() -> MLMultiArray {
        let lastWord = self.sentence.last!
        let matrix = try! MLMultiArray(shape: [1], dataType: .float32)
        matrix[0] = lastWord as NSNumber
        return matrix
    }
    
    func nextStepWith(beamSize: Int, vocabDict: [String]) -> [Caption] {
        let result = try! inceptionWOLabel.prediction(input1: self.lastWordMatrix(), lstm_1_h_in: self.state.0, lstm_1_c_in: self.state.1)
        
        let topK = PriorityQueue<Word>(maxLength: beamSize, compare: {$0 > $1})
        
        var leftProbability: Double = 1.0
        let wrappedMatrix = MultiArray<Double>(result.output1)
        for i in 0..<wrappedMatrix.count {
            let value = wrappedMatrix[i]
            topK.push(newElement: Word(index: i, proba: value))
            leftProbability -= value
            
            if let minProbaInTopK = topK.heap.min(), topK.count == beamSize && leftProbability <= minProbaInTopK.proba {
                break
            }
        }
        
        return topK.map({
            return  Caption(state: (result.lstm_1_h_out, result.lstm_1_c_out), sentence: self.sentence + [$0.index], score: log2($0.proba) + self.score, readableSentence: self.readAbleSentence + [vocabDict[$0.index]])
        })
    }
    
    static func <(lhs: Caption, rhs: Caption) -> Bool {
        return (lhs.score / Double(lhs.sentence.count)) < (rhs.score / Double(rhs.sentence.count))
    }
    
}
