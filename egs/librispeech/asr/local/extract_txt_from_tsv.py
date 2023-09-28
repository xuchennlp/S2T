import sys
import csv

tsv_file = sys.argv[1]
out_file = sys.argv[2]
extract_item = sys.argv[3]

with open(tsv_file) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = [dict(e) for e in reader]

fw = open(out_file, "w", encoding="utf-8")
for s in samples:
    if extract_item in s:
        fw.write("%s\n" % s[extract_item])
    else:
        print("Error in sample: ", s, "when extract ", extract_item)
        exit()

