#!/bin/bash

gcc -lm -O3 -fopenmp createGraph.c -o createGraph
./createGraph 700 700 12345678 > out.csv
python heatmap.py out.csv heatmap.png
