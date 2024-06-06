# OnnxBitAnalysis
Perform a search of invariant bits across multiple onnx nn models with a Python script: studyBits.py

------MAIN PROGRAM------

Arguments usage:

argv 1 -> onnx file -> model to be analysed

argv 2 -> string -> protection -> "fixed" for fixed protection, any other string will result in the application of variable protection

argv 3 -> string -> "normal" for  FP32 data, "quant" for quantised models in INT8 data

argv 4 -> string -> grouping strategy -> "filter" to perform a groping strategy across 2d convolutional filter, "id" groups across I dimension and "od" groups across O dimension

argv 5 -> string -> "transformer" for transformer models, "cnn" for convolutional neural network models -> In this last case, we only compute convolutional weights and convolutional 

argv 6 -> number -> bit positions to be analysed -> typically for INT8 would be 8 bits and for FP32, 32 bits. However, if the analysis is for the critical bits, INT8 would be 4 (bit positions 6-4). For FP32, 9, the exponent (positions 30-23).

The sign bit is added later. Bits start counting from the most significant bit (sign bit) -> bit number 0. 

Example of usage for a resnet model in fp32, with variable protection (VP) and a convolutional filter grouping strategy and the exponent bits to be analysed: "python3 studyBits.py resnet50.onnx var normal filter cnn 9" 

Quantise analysis has only been explored with a convolutional filter grouping strategy.
