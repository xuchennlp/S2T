#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple
import string
import sentencepiece as spm

from examples.speech_to_text.data_utils import (
    gen_vocab,
)
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["src_text", "tgt_text"]


class MTDataset(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    def __init__(self, root: str, src_lang, tgt_lang: str, split: str, tokenizer: bool = False) -> None:
        _root = Path(root) / "data" / split
        txt_root = _root / "txt" if (_root / "txt").is_dir() else _root
        assert _root.is_dir() and txt_root.is_dir(), (_root, txt_root)
        # Load source and target text
        self.data = []
        for _lang in [src_lang, tgt_lang]:
            txt_path = txt_root / f"{split}.{_lang}"
            if tokenizer:
                txt_path = txt_root / f"{split}.tok.{_lang}"

            with open(txt_path) as f:
                texts = [r.strip() for r in f]
                self.data.append(texts)
        self.data = list(zip(self.data[0], self.data[1]))

    def __getitem__(self, n: int) -> Tuple[str, str]:
        src_text, tgt_text = self.data[n]
        return src_text, tgt_text

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    splits = args.splits.split(",")
    src_train_text = []
    tgt_train_text = []
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    sent_num =[0]

    lang = f"{src_lang}-{tgt_lang}"
    cur_root = Path(args.data_root).absolute() / lang
    if args.output_root is None:
        output_root = cur_root
    else:
        output_root = Path(args.output_root).absolute()

    punctuation_str = string.punctuation
    punctuation_str = punctuation_str.replace("'", "")

    # Generate TSV manifest
    print("Generating manifest...")
    for split in splits:
        is_train_split = split.startswith("train")

        dataset = MTDataset(args.data_root, src_lang, tgt_lang, split, args.tokenizer)
        for src_text, tgt_text in tqdm(dataset):
            if args.lowercase_src:
                src_text = src_text.lower()
            if args.rm_punc_src:
                for w in punctuation_str:
                    src_text = src_text.replace(w, "")
                src_text = src_text.replace("  ", " ")

            manifest["src_text"].append(src_text)
            manifest["tgt_text"].append(tgt_text)

            if is_train_split and args.size != -1 and len(manifest["src_text"]) > args.size:
                break

        if is_train_split:
            src_train_text.extend(manifest["src_text"])
            tgt_train_text.extend(manifest["tgt_text"])
        
        sent_num.append(len(manifest["src_text"]))
    
    # Generate vocab and yaml
    print("Generating vocabulary...")
    tgt_v_size_str = "" if args.tgt_vocab_type == "char" else str(args.tgt_vocab_size)
    tgt_spm_filename_prefix = f"spm_{args.tgt_vocab_type}{tgt_v_size_str}"

    if args.share:
        if args.tgt_vocab_prefix is not None:
            tgt_spm_filename_prefix = args.tgt_vocab_prefix
        else:
            tgt_train_text.extend(src_train_text)
            tgt_spm_filename_prefix = tgt_spm_filename_prefix + "_share"
        src_spm_filename_prefix = tgt_spm_filename_prefix
    else:
        if args.tgt_vocab_prefix is not None:
            tgt_spm_filename_prefix = args.tgt_vocab_prefix
        else:
            tgt_spm_filename_prefix = tgt_spm_filename_prefix + "_" + tgt_lang

        if args.src_vocab_prefix is not None:
            src_spm_filename_prefix = args.src_vocab_prefix
        else:
            src_v_size_str = "" if args.src_vocab_type == "char" else str(args.src_vocab_size)
            src_spm_filename_prefix = f"spm_{args.src_vocab_type}{src_v_size_str}"
            src_spm_filename_prefix = src_spm_filename_prefix + "_" + src_lang

    src_spm_model = (output_root / (src_spm_filename_prefix + ".model")).as_posix()
    tgt_spm_model = (output_root / (tgt_spm_filename_prefix + ".model")).as_posix()

    if not os.path.exists(tgt_spm_model):
        with NamedTemporaryFile(mode="w") as f:
            for t in tgt_train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                output_root / tgt_spm_filename_prefix,
                args.tgt_vocab_type,
                args.tgt_vocab_size,
                normalization_rule_name="identity" if tgt_lang == "zh" else None
            )

    if not args.share and not os.path.exists(src_spm_model):
        with NamedTemporaryFile(mode="w") as f:
            for t in src_train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                output_root / src_spm_filename_prefix,
                args.src_vocab_type,
                args.src_vocab_size,
                normalization_rule_name="identity" if tgt_lang == "zh" else None
            )

    # Generate sentencepiece
    print("Applying sentencepiece...")
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.Load(tgt_spm_model)

    if args.share:
        src_sp = tgt_sp
    else:
        src_sp = spm.SentencePieceProcessor()
        src_sp.Load(src_spm_model)

    index = 0
    for split in splits:
        src_text = manifest["src_text"][sent_num[index]: sent_num[index + 1]]
        tgt_text = manifest["tgt_text"][sent_num[index]: sent_num[index + 1]]
        index += 1

        src_spm_name = (output_root / "data" / (split + "." + src_lang)).as_posix()
        tgt_spm_name = (output_root / "data" / (split + "." + tgt_lang)).as_posix()

        with open(src_spm_name, 'w') as f:
            for sentence in src_text:
                pieces = src_sp.EncodeAsPieces(sentence)
                result = " ".join(pieces)
                f.write(result + "\n")

        with open(tgt_spm_name, 'w') as f:
            for sentence in tgt_text:
                pieces = tgt_sp.EncodeAsPieces(sentence)
                result = " ".join(pieces)
                f.write(result + "\n")
        
    # Generate config YAML
    yaml_filename = f"config.yaml"
    if args.share:
        yaml_filename = f"config_share.yaml"

    conf = dict()
    conf["src_vocab_filename"] = src_spm_filename_prefix + ".txt"
    conf["tgt_vocab_filename"] = tgt_spm_filename_prefix + ".txt"
    conf["src_bpe_tokenizer"] = {
        "bpe": "sentencepiece",
        "sentencepiece_model": (output_root / (src_spm_filename_prefix + ".model")).as_posix(),
    }
    conf["tgt_bpe_tokenizer"] = {
        "bpe": "sentencepiece",
        "sentencepiece_model": (output_root / (tgt_spm_filename_prefix + ".model")).as_posix(),
    }

    import yaml
    with open(os.path.join(output_root, yaml_filename), "w") as f:
        yaml.dump(conf, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-root", "-o", default=None, type=str)
    parser.add_argument(
        "--src-vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument(
        "--tgt-vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--src-vocab-size", default=10000, type=int)
    parser.add_argument("--tgt-vocab-size", default=10000, type=int)
    parser.add_argument("--src-vocab-prefix", default=None, type=str, help="prefix of the specific source vocabulary")
    parser.add_argument("--tgt-vocab-prefix", default=None, type=str, help="prefix of the specific target vocabulary")
    parser.add_argument("--size", default=-1, type=int)
    parser.add_argument("--splits", default="train,dev,test", type=str)
    parser.add_argument("--lowercase-src", action="store_true", help="lowercase the source text")
    parser.add_argument("--rm-punc-src", action="store_true", help="remove the punctuation of the source text")
    parser.add_argument("--src-lang", required=True, type=str)
    parser.add_argument("--tgt-lang", required=True, type=str)
    parser.add_argument("--share", action="store_true", help="share the source and target vocabulary")
    parser.add_argument("--tokenizer", action="store_true", help="use tokenizer txt")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
