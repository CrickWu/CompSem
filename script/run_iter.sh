#!/bin/bash

python ./src/iter_solver.py --data ./data/60_word --gpu 0 --F_validation ./data/60_word/es-en/es-en.dict.tst.txt --F_training ./data/60_word/es-en/es-en.dict.trn.txt --epochs 100 --train_size 10 --normalize
