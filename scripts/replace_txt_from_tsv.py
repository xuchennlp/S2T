import sys
import csv
import pandas as pd

tsv_file = sys.argv[1]
out_file = sys.argv[2]
replace_file = sys.argv[3]
replace_item = sys.argv[4]

fr = open(replace_file, "r", encoding="utf-8")
replace_lines = fr.readlines()
idx = 0

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

    for s in samples:
        if replace_item in s:
            s[replace_item] = replace_lines[idx].strip()
            idx += 1
        else:
            print("Item %s Error in sample: " % replace_item)
            print(s)
            exit()
    df = pd.DataFrame.from_dict(samples)
    df.to_csv(
        out_file,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

