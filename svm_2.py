import csv
import numpy
import math
import sys

def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    arr = []
    for row in reader:
        temp = []
        for val in row:
            temp.append(int(val))
        arr.append(temp)
    return arr    

def read_data():
    csv_path = sys.argv[1]
    with open(csv_path) as f_obj:
        inp = csv_reader(f_obj)
    inp = numpy.array(inp)
    size = len(inp)
    feature = len(inp[0])
    return inp, size, feature

def write_data(inp, size, feature):
    file = open(sys.argv[2],'w')
    for i in range(0,size):
        file.write(str(0)+" ")
        for j in range(0,feature):
            file.write(str(j)+":"+str(inp[i][j])+" ")
        file.write('\n')
    file.close()

  
inputx, size, feature = read_data()
write_data(inputx, size, feature)