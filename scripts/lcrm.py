import string
import sys

punctuation_str = string.punctuation
punctuation_str = punctuation_str.replace("'", "")

user_input = sys.stdin.readlines()
for line in user_input:
    line = line.strip().lower()
    for w in punctuation_str:
        line = line .replace(w, "")
    line = " ".join(line.split(" "))

    print(line)
