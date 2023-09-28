import unicodedata
import jiwer
import jiwer.transforms as tr
import sys

ref_file = sys.argv[1]
hyp_file = sys.argv[2]


wer_standardize = tr.Compose(
    [
        tr.SubstituteRegexes({r"<<unk>>": r"@"}),
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.ExpandCommonEnglishContractions(),
        tr.RemoveKaldiNonWords(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.ReduceToListOfListOfWords(),
    ]
)
cer_standardize = tr.Compose(
    [
        tr.SubstituteRegexes({r"<<unk>>": r"@"}),
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.Strip(),
        tr.ReduceToListOfListOfChars(),
    ]
)


ref_lines = open(ref_file, "r").readlines()
hyp_lines = open(hyp_file, "r").readlines()

wer = jiwer.wer(ref_lines, hyp_lines,
                truth_transform=wer_standardize,
                hypothesis_transform=wer_standardize,
                )
cer = jiwer.cer(ref_lines, hyp_lines,
                truth_transform=cer_standardize,
                hypothesis_transform=cer_standardize,
                )

print("WER: %.4f" % wer)
print("CER: %.4f" % cer)
