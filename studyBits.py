from statistics import mean, median, mode
from tokenize import Double, Number
import numpy as np
import onnx
from onnx import numpy_helper
import json
from datetime import datetime
import json
from typing import Iterable
import struct
from struct import *
import sys
from collections import Counter
from bitarray import bitarray
import onnx
import time
def binaryConvert(num):
    # si es integer
    if isinstance(num, int):
        return "{0:08b}".format(num)
    # f for float, d for double and e for half or fp16
    else:
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
 
def makingMask(filter, protection):
    global quantized
    mask1 = []
    mask0 = []
    mask = []
    res = []
    if len(filter) == 0:
        print("Problem, lenght is 0")

    # the input array in converted to binary
    arrBinFlat = convert(filter.flatten())
    bitsEquals = 0
    bitsCount = []
    global globalbitcount
    global xbits
    start = 1
    end = xbits
    followed = False
    sign = []
    
    for i in range(start,end):
    
        count = 0
        for num in arrBinFlat:
            count += int(num[i])

        if protection and followed:
            break

        if count == len(arrBinFlat):
            mask1.append('1')
            mask0.append('1')
            mask.append('1')
            bitsEquals += 1
        elif count == 0:
            mask1.append('0')
            mask0.append('0')
            mask.append('0')
            bitsEquals += 1
        else:
            followed = True
            mask1.append('0')
            mask0.append('1')
            mask.append('X')

        # ----To count the number of equal bits-----
        bitsCount.append(count)
        
        
    # # ------To calculate if the sign is invariant--------
    # count = 0
    # for numero in arrBinFlat:
    #     count += int(numero[0])
    #     sign.append(numero[0])
    # bitsCount.insert(0, count)

    # if not sign or [sign[0]]*len(sign) == sign:
    #     bitsEquals += 1
    #     # print("sign add")
    #     if quantized:
    #         ints[0] += 1
    #     else:
    #         floats[0] += 1
    # --------------------------------------------------------

    # ---to calculate stats of invariant bits---
    bitsCoverts = bitsEquals
    percentCovert = (bitsEquals/(end-start))*100
    res.append(bitsCoverts)
    res.append(percentCovert)
    # print(mask)
    for i in range(start,len(mask)+1):
        value = mask[i-1]
        if value != 'X':
            if quantized:
                ints[i] += 1
            else:
                floats[i] += 1

    return res 

# only for fp32 cnn models
def ocProtection(convType):
    global totalElems 
    global contTensors
    global groups
    global xbits

    for pict in range(xbits+1):
        floats.append(0)
    for pict in range(xbits+1):
        ints.append(0)
    for n in nodes:
        if convType == n.op_type:
            for input in initializers:
                if n.input[1] == input.name:
                    a = onnx.numpy_helper.to_array(input).copy()
                    listPercentTensorTotal.clear()
                    totalElems += a.size
                    res = makingMask(a, protect)
                    listBitsTotal.append(res[0])
                    listPercentTensorTotal.append(res[1])
                    listPercentBitsTotal.append(res[1])
                    groups += 1 

                    if n.name == "": 
                        my_dictionary[contTensors] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0
                    else:
                        my_dictionary[n.input[1]] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0       
            contTensors += 1
    return

# only for fp32 cnn models
def icProtection(convType):
    global totalElems 
    global contTensors
    global groups
    global xbits

    for pict in range(xbits+1):
        floats.append(0)
    for pict in range(xbits+1):
        ints.append(0)
    for n in nodes:
        if convType == n.op_type:
            for input in initializers:
                if n.input[1] == input.name:
                    a = onnx.numpy_helper.to_array(input).copy()
                    listPercentTensorTotal.clear()
                    totalElems += a.size
                    # print(input.name, a.shape)
                    for oc in range(len(a)):
                        res = makingMask(a[oc],protect)
                        listBitsTotal.append(res[0])
                        listPercentTensorTotal.append(res[1])
                        listPercentBitsTotal.append(res[1])
                        groups += 1 

                    if n.name == "": 
                        my_dictionary[contTensors] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0
                    else:
                        my_dictionary[n.name] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0          
            contTensors += 1
    return

def filterProtectionVisualTransformers():
    global totalElems1x1
    global totalElems3x3
    global totalElems 
    global contTensors
    global tensors3x3
    global tensors1x1
    global extraTensors
    global tensorsMore3x3
    global cont
    global toComplete
    global extra
    global ite
    global globalbitcount 
    global groups
    global xbits


    for pict in range(xbits+1):
        floats.append(0)
    for pict in range(xbits+1):
        ints.append(0)
    for n in nodes:
            listPercentTensorTotal.clear()
            for nodeInput in n.input:
                for input in initializers:
                    if nodeInput == input.name:
                        a = onnx.numpy_helper.to_array(input).copy()
                        # if the size of the tensor is less than 3 elements, we dont compute it
                        if a.size > 3:
                            totalElems += a.size
                            cont = 0
                            toComplete = 0
                            ite = 0
                            # case with conv 12x128x3x3
                            if a.ndim > 3:
                                totalElems3x3 += a.size
                                tensors3x3 += 1   
                                for oc in range(len(a)):
                                    for ic in range(len(a[oc])):
                                        res = makingMask(a[oc][ic], protect)
                                        listBitsTotal.append(res[0])
                                        listPercentTensorTotal.append(res[1])
                                        listPercentBitsTotal.append(res[1])
                                        groups += 1 
                                if len(a[0][0]) != 3:
                                    shapeMore3x3.append(a.shape)
                                    tensorsMore3x3 += 1
                                    print("GREATER THAN 3X3")
                            # case with 2 dimensions, for example: 1024 x 4096
                            elif a.ndim == 2:
                                tensors1x1 += 1
                                # we tried to group weights in an exact number of sets of 9 weights, if this is not possible, in sets of 8 and if neither of them is possible, in sets of 9 weights and in another set the remaining weights.
                                if (len(a[1])%9) == 0:
                                    cont = 9
                                    toComplete = 0
                                    ite = int(len(a[0])/9)
                                    extra = 0
                                elif (len(a[1])%8) == 0:
                                    cont = 8
                                    toComplete = 0
                                    ite = int(len(a[0])/8)
                                    extra = 0
                                else:
                                    extraTensors += 1
                                    cont = 9 
                                    toComplete = len(a[0])%9
                                    ite = int(len(a[0])/9)
                                    extra = 1
                                totalElems1x1 += a.size 

                                for oc in range(len(a)):
                                    for i in range(ite):
                                        ini = cont * i
                                        fin = (cont * i) + cont
                                        res = makingMask(a[oc][ini:fin], protect)
                                        listBitsTotal.append(res[0])
                                        listPercentTensorTotal.append(res[1])
                                        listPercentBitsTotal.append(res[1])
                                        groups += 1 
                                    if toComplete:
                                        ini = fin
                                        fin = ini + toComplete
                                        res = makingMask(a[oc][ini:fin], protect)
                                        listBitsTotal.append(res[0])
                                        listPercentTensorTotal.append(res[1])
                                        listPercentBitsTotal.append(res[1])
                                        groups += 1
                            # case with three dimensions, for example: 1x1024x2048
                            elif a.ndim == 3:
                                tensors1x1 += 1
                                if (len(a[0][0])%9) == 0:
                                    cont = 9
                                    toComplete = 0
                                    ite = int(len(a[0][0])/9)
                                    extra = 0
                                elif (len(a[0][0])%8) == 0:
                                    cont = 8
                                    toComplete = 0
                                    ite = int(len(a[0][0])/8)
                                    extra = 0
                                else:
                                    extraTensors += 1
                                    cont = 9 
                                    toComplete = len(a[0][0])%9
                                    ite = int(len(a[0][0])/9)
                                    extra = 1
                                totalElems1x1 += a.size 

                                for oc in range(len(a)):
                                    for ic in range(len(a[oc])):
                                        for i in range(ite):
                                            ini = cont * i
                                            fin = (cont * i) + cont
                                            res = makingMask(a[oc][ic][ini:fin], protect)
                                            listBitsTotal.append(res[0])
                                            listPercentTensorTotal.append(res[1])
                                            listPercentBitsTotal.append(res[1])
                                            groups += 1 
                                        if toComplete:
                                            ini = fin
                                            fin = ini + toComplete
                                            res = makingMask(a[oc][ic][ini:fin], protect)
                                            listBitsTotal.append(res[0])
                                            listPercentTensorTotal.append(res[1])
                                            listPercentBitsTotal.append(res[1])
                                            groups += 1  
                            # cases with one dimension, for example: 1024
                            else:
                                tensors1x1 += 1
                                if (len(a)%9) == 0:
                                    cont = 9
                                    toComplete = 0
                                    ite = int(len(a)/9)
                                    extra = 0
                                elif (len(a)%8) == 0:
                                    cont = 8
                                    toComplete = 0
                                    ite = int(len(a)/8)
                                    extra = 0
                                else:
                                    extraTensors += 1
                                    cont = 9 
                                    toComplete = len(a)%9
                                    ite = int(len(a)/9)
                                    extra = 1
                                totalElems1x1 += a.size 

                                fin = 0
                                for i in range(ite):
                                    ini = cont * i
                                    fin = (cont * i) + cont
                                    res = makingMask(a[ini:fin], protect)
                                    listBitsTotal.append(res[0])
                                    listPercentTensorTotal.append(res[1])
                                    listPercentBitsTotal.append(res[1])
                                    groups += 1 
                                if toComplete:
                                    ini = fin
                                    fin = ini + toComplete
                                    res = makingMask(a[ini:fin], protect)
                                    listBitsTotal.append(res[0])
                                    listPercentTensorTotal.append(res[1])
                                    listPercentBitsTotal.append(res[1])
                                    groups += 1 
                        if n.name == "": 
                            my_dictionary[contTensors] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0 
                        else:
                            my_dictionary[n.name] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0             
            contTensors += 1
    return

        
def filterProtectionCnn(conv, weightID):
    global totalElems1x1
    global totalElems3x3
    global totalElems 
    global contTensors
    global tensors3x3
    global tensors1x1
    global extraTensors
    global tensorsMore3x3
    global cont
    global toComplete
    global extra
    global ite
    global globalbitcount 
    global groups
    global xbits


    #---------------------------------------MAKING MASKS-------------------------------------------------------------------

    for pict in range(xbits+1):
        floats.append(0)
    for pict in range(xbits+1):
        ints.append(0)
    for n in nodes:
        if conv == n.op_type:
            for input in initializers:
                if n.input[weightID] == input.name:
                    a = onnx.numpy_helper.to_array(input).copy()
                    totalElems += a.size
                    percentTensor = 0
                    cont = 0
                    toComplete = 0
                    ite = 0
                    if len(a[0][0]) > 1:
                        totalElems3x3 += a.size
                        tensors3x3 += 1     
                        for oc in range(len(a)):
                            for ic in range(len(a[oc])):
                                res = makingMask(a[oc][ic], protect)
                                listBitsTotal.append(res[0])
                                listPercentTensorTotal.append(res[1])
                                listPercentBitsTotal.append(res[1])
                                groups += 1
                        if len(a[0][0]) != 3:
                            shapeMore3x3.append(a.shape)
                            tensorsMore3x3 += 1
                            print("GREATER THAN 3X3") 
                    else:
                        totalElems1x1 += a.size
                        tensors1x1 += 1
                        if (len(a[0])%9) == 0:
                            cont = 9
                            toComplete = 0
                            ite = int(len(a[0])/9)
                            extra = 0
                        elif (len(a[0])%8) == 0:
                            cont = 8
                            toComplete = 0
                            ite = int(len(a[0])/8)
                            extra = 0
                        else:
                            cont = 9 
                            toComplete = len(a[0])%9
                            ite = int(len(a[0])/9)
                            extra = 1

                        for oc in range(len(a)):
                            for i in range(ite):
                                ini = cont * i
                                fin = (cont * i) + cont
                                res = makingMask(a[oc][ini:fin], protect)
                                listBitsTotal.append(res[0])
                                listPercentTensorTotal.append(res[1])
                                listPercentBitsTotal.append(res[1])
                                groups += 1 
                            if toComplete:
                                ini = fin
                                fin = ini + toComplete
                                res = makingMask(a[oc][ini:fin], protect)
                                listBitsTotal.append(res[0])
                                listPercentTensorTotal.append(res[1])
                                listPercentBitsTotal.append(res[1])
                                groups += 1 

                    if n.name == "": 
                        my_dictionary[contTensors] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0 
                    else:
                        my_dictionary[n.name] = round(mean(listPercentTensorTotal),3) if len(listPercentTensorTotal) > 0 else 0                  
            contTensors += 1
    return


# ------------------------------------------------------MAIN PROGRAM------------------------------------------------------------
# arguments usage:
# argv 1 -> onnx file -> model to be analysed
# argv 2 -> string -> protection -> "fixed" for fixed protection, any other string will result in the application of variable protection
# argv 3 -> string -> "normal" for  FP32 data, "quant" for quantised models in INT8 data
# argv 4 -> string -> grouping strategy -> "filter" to perform a groping strategy across 2d convolutional filter, "id" groups across I dimension and "od" groups across O dimension
# argv 5 -> string -> "transformer" for transformer models, "cnn" for convolutional neural network models -> in this last case we only comput convolutional weights and convolutional 
# argv 6 -> number -> bit positions to be analysed -> typically for INT8 would be 8 bits and for FP32, 32 bits. However, if the analysis is for the critical bits, for INT8 would be 4 (bit positions 6-4). For FP32, 9, the exponent (positions 30-23). 
# The sign bit is added later. Bits start counting from the most significant bit (sign bit) -> bit number 0. 
# example of usage, for a resnet model, in fp32, with variable protection (VP) and a convolutional filter grouping strategy and the exponent bits to be analysed: "python3 studyBits.py resnet50.onnx var normal filter cnn 9" 
# quantise analysis has only been explore with convolutional filter grouping strategy

# inputs:
print(f"model: {sys.argv[1]} / protection: {sys.argv[2]} / data type: {sys.argv[3]} / strategy: {sys.argv[4]} / model type: {sys.argv[5]} / bits analysed: {sys.argv[6]}")

loadmodel = sys.argv[1]
protect = 0
xbits = 0
conv = None
weightId = None
quantized = False
if "fixed" == sys.argv[2]:
    protect = 1
if sys.argv[3] == 'quant':
    conv = "QLinearConv"
    weightId = 3
    quantized = True
elif sys.argv[3] == 'normal':
    conv = "Conv"
    weightId = 1

xbits = int(sys.argv[6])
model = onnx.load(loadmodel)
print("LOAD", protect)
nodes = model.graph.node
initializers = model.graph.initializer
convert = np.vectorize(binaryConvert)

floats = []
ints = []
listBitsTotal = []
listPercentBitsTotal = []
listPercentTensorTotal = []
my_dictionary = {}
contTensors = 0
globalbitcount = 0
groups = 0
totalElems = 0

if "filter" == sys.argv[4]:
    percentBits1x1 = []
    shapeMore3x3 = []
    extraTensors = 0
    tensorsMore3x3 = 0
    tensors3x3 = 0
    tensors1x1 = 0
    totalElems1x1 = 0
    totalElems3x3 = 0
    cont = 0
    toComplete = 0
    extra = 0
    ite = 0

# -------------------------------------------LEVEL PROTECTION: FILTER, INPUT CHANNEL OR OUTPUT CHANEL---------------------------------------------
ort_start = time.time()
if "filter" == sys.argv[4]:
    if sys.argv[5] == "cnn":
        filterProtectionCnn(conv, weightId)
    elif sys.argv[5] == "transformer":
        filterProtectionVisualTransformers()
elif "id" == sys.argv[4]:
    icProtection(conv)
elif "od" == sys.argv[4]:
    ocProtection(conv)
ort_end = time.time() - ort_start

# ----------------------------------------------------------------------SHOW STATS----------------------------------------------------------------
#  we show stats per layer and stats of the whole model
# ------------------------------------------------------------------------------------------------------------------------------------------------
print("\n---------------------------------------Dictionary of bit-cover percentages per layer---------------------------------------")
auxprint = json.dumps(my_dictionary, indent=4).replace(',','')
print(auxprint.replace('.',','))

print("\n-------------------------------------------------INVARIANT BIT INFO------------------------------------------------------------------")
# --------Bit Info------------
print("Invariant Bits by position (starting from the most significant bit -> bit 0 is bit 31 in FP32 and vit bit 7 in INT8):")
if sys.argv[3] == 'quant':
    print(ints)
else:
    print(floats)

# Count the frequency of each element in the list
countes = Counter(listBitsTotal)
values = list(countes.keys())
counts = list(countes.values())

print("Groups by Invariant Bits:")
for i in range(len(values)):
    print(values[i], counts[i])

totalnums = 0
if sys.argv[3] == 'quant':
    for i in range(len(ints)):
        totalnums += ints[i]
else:
    for i in range(len(floats)):
        totalnums += floats[i]
        

totalpercentbits = 0
for i in range(len(values)):
    totalpercentbits += values[i] * counts[i]

print("Check if there are the same bits:")
print(totalnums, totalpercentbits, totalpercentbits==totalnums)

print("\n-------------------------------------------------INVARIANT BIT % INFO------------------------------------------------------------------")
totalnums2 = 0
if sys.argv[3] == 'quant':
    for i in range(len(ints)):
        ints[i] = round((ints[i]/totalnums)*100, 4)
        totalnums2 += ints[i]
    for i in range(len(values)):
        counts[i] = round((counts[i]/groups)*100, 4)

    print("Invariant Bits by position:")
    auxprint2 = str(ints).replace(',','')
    print(str(auxprint2).replace('.',','))
else:
    for i in range(len(floats)):
        floats[i] = round((floats[i]/totalnums)*100, 4)
        totalnums2 += floats[i]
    for i in range(len(values)):
        counts[i] = round((counts[i]/groups)*100, 4)

    print("Invariant Bits by position:")
    auxprint2 = str(floats).replace(',','')
    print(str(auxprint2).replace('.',','))

print("Groups by Invariant Bits:")
for i in range(len(values)):
    print(values[i], counts[i])


totalpercentbits = 0
for i in range(len(values)):
    totalpercentbits += counts[i]

print("Check if there is the same bits:")
print(round(totalnums2), round(totalpercentbits), round(totalpercentbits)==round(totalnums2))


print("\n------------------------------------------------INVARIANT GENERAL RESULTS------------------------------------------------------------")
resultBits = 0
resultTensor = 0
if len(listPercentBitsTotal) > 0:
    resultBits = mean(listPercentBitsTotal)

print(f"Avg Bits Covert per tensor {round(resultBits, 3)} %")
print("Fixed Protection") if protect else print("Variable Protection")
print(f"Time to compute all masks: {round(ort_end, 3)} seconds")



