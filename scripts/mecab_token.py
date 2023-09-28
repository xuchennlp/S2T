#!/usr/bin/env python3
import MeCab
import sys

fin = sys.argv[1]
fout = sys.argv[2]

fw = open(fout, "w")

wakati = MeCab.Tagger("-Owakati")

with open(fin, "r") as fr:
    for line in fr.readlines():
        token_line = wakati.parse(line.strip()).split()
        fw.write(" ".join(token_line) + "\n")

fw.close()
