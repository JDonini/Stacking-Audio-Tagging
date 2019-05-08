import sys
import getopt
import csv
import os

ifile = ''
ofile = ''

myopts, args = getopt.getopt(sys.argv[1:], "i:o:")

for o, a in myopts:
    if o == '-i':
        ifile = a
    elif o == '-o':
        ofile = a
    else:
        print("Usage: %s -i input -o output" % sys.argv[0])

fileIn = open(ifile)
fileOut = open(ofile, "w")

lastPos = fileIn.tell()

fileOut.write("@RELATION CSC722Assignment3\n\n\n")
count = 1

for k in range(len(fileIn.readline().strip().split(','))-1):
    fileOut.write("@ATTRIBUTE f"+str(count)+" real\n")
    count += 1

tempClass = []
for line in fileIn:
    tempClass.append(line.strip().split(',')[0])
fileOut.write("@ATTRIBUTE outputClass {"+','.join(tempClass)+" }\n\n\n")

fileOut.write("@DATA \n")

fileIn.seek(lastPos)

tempStr = []
for line in fileIn:
    for i in range(len(line.strip().split(','))):
        if i != 0:
            tempStr.append(line.strip().split(',')[i])
        if i == len(line.strip().split(','))-1:
            tempStr.append(line.strip().split(',')[0])
        i = i+1
    fileOut.write(','.join(tempStr)+"\n")
    tempStr = []
fileOut.close()
