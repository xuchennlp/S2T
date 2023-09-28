#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
import csv
from tempfile import NamedTemporaryFile
import numpy as np

import pandas as pd
from examples.speech_to_text.data_utils import (
    cal_gcmvn_stats,
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "src_text"]


def process(args):
    data_root = Path(args.data_root).absolute()
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # Extract features
    feature_root = out_root.parent / "fbank80"
    feature_root.mkdir(exist_ok=True)
    zip_path = out_root.parent / "fbank80.zip"

    gen_cmvn_flag = False
    if args.cmvn_type == "global":
        cmvn_path = out_root / "gcmvn.npz"
        if not Path.exists(cmvn_path) or args.overwrite:
            gen_cmvn_flag = True

            print("And estimating cepstral mean and variance stats...")
            gcmvn_feature_list = []

    if args.overwrite or not Path.exists(zip_path):
        for split in SPLITS:
            print(f"Fetching split {split}...")

            is_train_split = split.startswith("train")
            dataset = LIBRISPEECH(data_root.as_posix(), url=split, download=True)
            print("Extracting log mel filter bank features...")
            for wav, sample_rate, _, spk_id, chapter_no, utt_no in tqdm(dataset):
                sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                features = extract_fbank_features(
                    wav, sample_rate, feature_root / f"{sample_id}.npy"
                )

                if (
                    is_train_split
                    and args.cmvn_type == "global"
                ):
                    if len(gcmvn_feature_list) < args.gcmvn_max_num:
                        gcmvn_feature_list.append(features)
        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
    else:
        if gen_cmvn_flag:
            for split in SPLITS:
                print(f"Fetching split {split} only for cmvn...")

                is_train_split = split.startswith("train")
                if not is_train_split:
                    continue

                dataset = LIBRISPEECH(data_root.as_posix(), url=split, download=True)
                print("Extracting log mel filter bank features...")
                for wav, sample_rate, _, spk_id, chapter_no, utt_no in tqdm(dataset):
                    sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                    features = extract_fbank_features(
                        wav, sample_rate, None
                    )

                    if (
                        is_train_split
                        and args.cmvn_type == "global"
                    ):
                        if len(gcmvn_feature_list) < args.gcmvn_max_num:
                            gcmvn_feature_list.append(features)
                        else:
                            break

    if gen_cmvn_flag and len(gcmvn_feature_list) > 0:
        # Estimate and save cmv
        stats = cal_gcmvn_stats(gcmvn_feature_list)
        with open(out_root / "gcmvn.npz", "wb") as f:
            np.savez(f, mean=stats["mean"], std=stats["std"])

    gen_manifest_flag = False
    for split in SPLITS:
        if not Path.exists(out_root / f"{split}.tsv"):
            gen_manifest_flag = True
            break

    train_text = []
    if args.overwrite or gen_manifest_flag:
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Generate TSV manifest
        print("Generating manifest...")
        for split in SPLITS:
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = LIBRISPEECH(data_root.as_posix(), url=split)
            for wav, sample_rate, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
                sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                manifest["id"].append(sample_id)
                manifest["audio"].append(zip_manifest[sample_id])
                duration_ms = int(wav.size(1) / sample_rate * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                manifest["tgt_text"].append(utt.lower())
                manifest["speaker"].append(spk_id)
            save_df_to_tsv(
                pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
            )
            if split.startswith("train"):
                train_text.extend(manifest["tgt_text"])


    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"

    gen_vocab_flag = True
    if args.asr_prefix is not None:
        gen_vocab_flag = False
        spm_filename_prefix = args.asr_prefix

    if gen_vocab_flag:
        with NamedTemporaryFile(mode="w") as f:
            if len(train_text) == 0:
                print("Loading the training text...")
                for split in SPLITS:
                    if split.startswith("train"):
                        csv_path = out_root / f"{split}.tsv"
                        with open(csv_path) as fc:
                            reader = csv.DictReader(
                                fc,
                                delimiter="\t",
                                quotechar=None,
                                doublequote=False,
                                lineterminator="\n",
                                quoting=csv.QUOTE_NONE,
                            )
                            for e in reader:
                                e = dict(e)
                                train_text.append(e["tgt_text"])
                                if "src_text" in e:
                                    train_text.append(e["src_text"])
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                out_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )

    # Generate config YAML
    gen_config_yaml(
        out_root, spm_filename_prefix + ".model", specaugment_policy="ld",
        asr_spm_filename=spm_filename_prefix + ".model",
        share_src_and_tgt=True,
        cmvn_type=args.cmvn_type,
        gcmvn_path=(out_root / "gcmvn.npz" if args.cmvn_type == "global" else None),
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--asr-prefix", type=str, default=None, help="prefix of the asr dict")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing files")
    parser.add_argument(
        "--cmvn-type",
        default="utterance",
        choices=["global", "utterance"],
        help="The type of cepstral mean and variance normalization",
    )
    parser.add_argument(
        "--gcmvn-max-num",
        default=150000,
        type=int,
        help=(
            "Maximum number of sentences to use to estimate" "global mean and variance"
        ),
    )
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
