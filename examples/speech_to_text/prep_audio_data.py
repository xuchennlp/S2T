#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
import string
import csv
import yaml
import copy

import numpy as np
import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
from torch.utils.data import Dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]


class AudioDataset(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    def __init__(
        self,
        root: str,
        src_lang,
        tgt_lang: str,
        split: str,
        speed_perturb: bool = False,
        size: int = -1,
        use_raw: bool = False,
        tokenizer: bool = False,
    ) -> None:
        _root = Path(root) / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert wav_root.is_dir() and txt_root.is_dir(), (_root, wav_root, txt_root)

        self.use_raw = use_raw
        self.speed_perturb = (
            [0.9, 1.0, 1.1] if speed_perturb and split.startswith("train") else None
        )
        self.size = size if split.startswith("train") else -1

        # Load audio segments
        yaml_file = txt_root / f"{split}.yaml"
        if yaml_file.is_file():
            self.mode = "yaml"
            with open(yaml_file) as f:
                segments = yaml.load(f, Loader=yaml.BaseLoader)
                total_length = len(segments)

                if 0 < self.size < total_length:
                    segments = segments[: self.size]
        else:
            self.mode = "easy"

            segments = dict()
            audio_file = txt_root / f"{split}.audio"
            assert audio_file.is_file(), audio_file
            with open(audio_file) as f:
                audios = [line.strip() for line in f.readlines()]
            total_length = len(audios)

            if 0 < self.size < total_length:
                audios = audios[: self.size]
            for idx, audio in enumerate(audios):
                segments[idx] = {"audio": audio}

        # Load source and target utterances
        self.have_src_utt = False
        self.have_tgt_utt = False
        for _lang in [src_lang, tgt_lang]:
            if _lang is None:
                continue
            txt_path = txt_root / f"{split}.{_lang}"
            if tokenizer:
                txt_path = txt_root / f"{split}.tok.{_lang}"

            if Path.exists(txt_path):
                if _lang == src_lang:
                    self.have_src_utt = True
                else:
                    self.have_tgt_utt = True
                with open(txt_path) as f:
                    utterances = [r.strip() for r in f]
                assert total_length == len(utterances), (total_length, len(utterances))

                if 0 < self.size < total_length:
                    utterances = utterances[: self.size]
                for idx, u in enumerate(utterances):
                    segments[idx][_lang] = u

        # split = split.replace("_gen", "")
        # Gather info
        self.data = dict()
        if self.mode == "easy":
            real_idx = 0
            for idx, v in segments.items():
                audio_name = f"{split}_{v['audio']}"
                v["audio"] = (wav_root / v["audio"].strip()).as_posix() + ".wav"
                if self.speed_perturb is not None:
                    for perturb in self.speed_perturb:
                        sp_item = copy.deepcopy(v)
                        sp_item["perturb"] = perturb
                        sp_item["id"] = f"{audio_name}_sp{perturb}"
                        self.data[real_idx] = sp_item
                        real_idx += 1
                else:
                    v["id"] = audio_name
                    self.data[real_idx] = v
                    real_idx += 1
                if 0 < self.size <= real_idx:
                    break

        elif self.mode == "yaml":
            idx = 0
            for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
                wav_path = wav_root / wav_filename
                sample_rate = torchaudio.info(wav_path.as_posix()).sample_rate
                seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
                for i, segment in enumerate(seg_group):
                    offset = int(float(segment["offset"]) * sample_rate)
                    n_frames = int(float(segment["duration"]) * sample_rate)
                    _id = f"{split}_{wav_path.stem}_{i}"
                    # _id = f"{wav_path.stem}_{i}"

                    item = dict()
                    item["audio"] = wav_path.as_posix()
                    item["offset"] = offset
                    item["n_frames"] = n_frames
                    item["sample_rate"] = sample_rate
                    item[src_lang] = segment[src_lang]
                    if tgt_lang is not None:
                        item[tgt_lang] = segment[tgt_lang]

                    if self.speed_perturb is not None:
                        for perturb in self.speed_perturb:
                            sp_item = copy.deepcopy(item)
                            sp_item["id"] = f"{_id}_sp{perturb}"
                            sp_item["perturb"] = perturb
                            self.data[idx] = sp_item
                            idx += 1
                    else:
                        item["id"] = _id
                        self.data[idx] = item
                        idx += 1
                    if 0 < self.size <= idx:
                        break

    def __getitem__(self, n: int):
        return self.data[n]

    def get(self, n: int, need_waveform: bool = False):
        item = self.data[n]
        audio = item["audio"]

        if item.get("n_frames", False) and item.get("sample_rate", False):
            n_frames = item["n_frames"]
            sample_rate = item["sample_rate"]
        else:
            info = torchaudio.info(audio)
            sample_rate = info.sample_rate
            n_frames = info.num_frames

        waveform = None
        if item.get("perturb", False):
            n_frames = n_frames / item["perturb"]

        if need_waveform:
            offset = item.get("offset", False)
            if offset is not False:
                waveform, sample_rate = torchaudio.load(
                    audio, frame_offset=offset, num_frames=item["n_frames"]
                )
            else:
                waveform, sample_rate = torchaudio.load(audio)

            if item.get("perturb", False):
                effects = [["speed", f"{item['perturb']}"], ["rate", f"{sample_rate}"]]
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, sample_rate, effects
                )

        return waveform, sample_rate, n_frames

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    task = args.task
    splits = args.splits.split(",")
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    assert (task == "st" and tgt_lang is not None) or (
        task == "asr" and src_lang is not None
    )

    cur_root = root
    if not cur_root.is_dir():
        logger.error(f"{cur_root.as_posix()} does not exist. Skipped.")

    if args.output_root is None:
        output_root = cur_root
    else:
        output_root = Path(args.output_root).absolute()

    # Extract features
    datasets = dict()
    use_raw = args.raw
    size = args.size
    if args.speed_perturb:
        zip_path = output_root.parent / "fbank80_sp.zip"
    else:
        zip_path = output_root.parent / "fbank80.zip"

    if use_raw:
        gen_feature_flag = False
    else:
        gen_feature_flag = False
        if not Path.exists(zip_path) or args.overwrite:
            gen_feature_flag = True

    if gen_feature_flag:
        if args.speed_perturb:
            feature_root = output_root.parent / "fbank80_sp"
        else:
            feature_root = output_root.parent / "fbank80"
        feature_root.mkdir(exist_ok=True)

        print("Extracting log mel filter bank features...")
        for split in splits:
            print(f"Fetching split {split}...")
            is_train_split = split.startswith("train")
            dataset = AudioDataset(
                root.as_posix(),
                src_lang,
                tgt_lang,
                split,
                args.speed_perturb,
                size,
                use_raw,
                args.tokenizer,
            )
            if split not in datasets:
                datasets[split] = dataset

            if is_train_split and args.cmvn_type == "global":
                print("And estimating cepstral mean and variance stats...")
                gcmvn_feature_list = []

            for idx in tqdm(range(len(dataset))):
                item = dataset[idx]

                utt_id = item["id"]
                features_path = (feature_root / f"{utt_id}.npy").as_posix()

                if os.path.exists(features_path):
                    continue

                waveform, sample_rate, _ = dataset.get(idx, need_waveform=True)
                if waveform.shape[1] == 0:
                    continue

                try:
                    features = extract_fbank_features(
                        waveform, sample_rate, Path(features_path)
                    )
                except AssertionError:
                    logger.warning("Extract file %s failed." % utt_id)

                if (
                    split == "train"
                    and args.cmvn_type == "global"
                    and not utt_id.startswith("sp")
                ):
                    if len(gcmvn_feature_list) < args.gcmvn_max_num:
                        gcmvn_feature_list.append(features)

            if is_train_split and args.cmvn_type == "global":
                # Estimate and save cmv
                stats = cal_gcmvn_stats(gcmvn_feature_list)
                with open(output_root / "gcmvn.npz", "wb") as f:
                    np.savez(f, mean=stats["mean"], std=stats["std"])

        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(feature_root, zip_path)

        # Clean up
        shutil.rmtree(feature_root)

    gen_manifest_flag = False
    for split in splits:
        if not Path.exists(output_root / f"{split}.tsv"):
            gen_manifest_flag = True
            break

    punctuation_str = string.punctuation
    punctuation_str = punctuation_str.replace("'", "")

    train_text = []
    if args.overwrite or gen_manifest_flag:
        if not use_raw:
            print("Fetching ZIP manifest...")
            zip_manifest = get_zip_manifest(zip_path)

        # Generate TSV manifest
        print("Generating manifest...")
        for split in splits:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}

            if split in datasets:
                dataset = datasets[split]
            else:
                dataset = AudioDataset(
                    root.as_posix(),
                    src_lang,
                    tgt_lang,
                    split,
                    args.speed_perturb,
                    size,
                    use_raw,
                    args.tokenizer,
                )
            if args.task == "st" and args.add_src and dataset.have_src_utt:
                manifest["src_text"] = []
            for idx in tqdm(range(len(dataset))):
                item = dataset[idx]
                _, sample_rate, n_frames = dataset.get(idx, need_waveform=False)
                utt_id = item["id"]

                if use_raw:
                    audio_path = item["audio"]

                    # add offset and frames info
                    if item.get("offset", False) is not False:
                        audio_path = f"{audio_path}:{item['offset']}:{n_frames}"
                    manifest["audio"].append(audio_path)
                else:
                    if utt_id in zip_manifest:
                        manifest["audio"].append(zip_manifest[utt_id])
                    else:
                        logger.warning("%s is not in the zip" % utt_id)
                        continue

                manifest["id"].append(utt_id)
                duration_ms = int(n_frames / sample_rate * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))

                if dataset.have_src_utt:
                    src_utt = item[src_lang]
                    if args.lowercase_src:
                        src_utt = src_utt.lower()
                    if args.rm_punc_src:
                        for w in punctuation_str:
                            src_utt = src_utt.replace(w, "")
                        src_utt = " ".join(src_utt.split(" "))
                else:
                    src_utt = None

                if dataset.have_tgt_utt:
                    tgt_utt = item[tgt_lang]
                else:
                    tgt_utt = None
                if task == "asr":
                    manifest["tgt_text"].append(src_utt)
                elif task == "st":
                    if args.add_src and src_utt is not None:
                        manifest["src_text"].append(src_utt)
                    manifest["tgt_text"].append(tgt_utt)

            if is_train_split:
                if args.task == "st" and args.add_src and args.share:
                    train_text.extend(manifest["src_text"])
                train_text.extend(manifest["tgt_text"])

            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=5)
            save_df_to_tsv(df, output_root / f"{split}.tsv")

    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{task}"
    asr_spm_filename = None
    gen_vocab_flag = True

    # if task == "st" and args.add_src:
    if args.add_src:
        if args.share:
            if args.st_spm_prefix is not None:
                gen_vocab_flag = False
                spm_filename_prefix = args.st_spm_prefix
            else:
                spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{task}_share"
            asr_spm_filename = spm_filename_prefix + ".model"
        else:
            if args.st_spm_prefix is not None:
                gen_vocab_flag = False
                spm_filename_prefix = args.st_spm_prefix
            assert args.asr_prefix is not None
            asr_spm_filename = args.asr_prefix + ".model"
    elif task == "asr":
        if args.asr_prefix is not None:
            gen_vocab_flag = False
            spm_filename_prefix = args.asr_prefix

    if gen_vocab_flag:
        if len(train_text) == 0:
            print("Loading the training text to build dictionary...")

            for split in splits:
                if split.startswith("train"):
                    csv_path = output_root / f"{split}.tsv"
                    with open(csv_path) as f:
                        reader = csv.DictReader(
                            f,
                            delimiter="\t",
                            quotechar=None,
                            doublequote=False,
                            lineterminator="\n",
                            quoting=csv.QUOTE_NONE,
                        )

                        # if task == "st" and args.add_src and args.share:
                        if args.add_src and args.share:
                            for e in reader:
                                src_utt = dict(e)["src_text"]
                                tgt_utt = dict(e)["tgt_text"]
                                if args.lowercase_src:
                                    src_utt = src_utt.lower()
                                if args.rm_punc_src:
                                    for w in punctuation_str:
                                        src_utt = src_utt.replace(w, "")
                                    src_utt = " ".join(src_utt.split(" "))
                                train_text.append(src_utt)
                                train_text.append(tgt_utt)
                        else:
                            tgt_text = [(dict(e))["tgt_text"] for e in reader]
                            train_text.extend(tgt_text)

        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                output_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )

    # Generate config YAML
    yaml_filename = f"config.yaml"
    if task == "st" and args.add_src and args.share:
        yaml_filename = f"config_share.yaml"

    gen_config_yaml(
        output_root,
        spm_filename_prefix + ".model",
        yaml_filename=yaml_filename,
        specaugment_policy="ld2",
        cmvn_type=args.cmvn_type,
        gcmvn_path=(output_root / "gcmvn.npz" if args.cmvn_type == "global" else None),
        asr_spm_filename=asr_spm_filename,
        share_src_and_tgt=True if task == "asr" else False,
    )


def main():
    parser = argparse.ArgumentParser()
    # general setting
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-root", "-o", default=None, type=str)
    parser.add_argument("--task", type=str, default="st", choices=["asr", "st"])
    parser.add_argument("--src-lang", type=str, required=True, help="source language")
    parser.add_argument("--tgt-lang", type=str, help="target language")
    parser.add_argument(
        "--splits", type=str, default="train,dev,test", help="dataset splits"
    )
    parser.add_argument("--size", default=-1, type=int, help="use part of the data")
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite the existing files"
    )
    # audio setting
    parser.add_argument(
        "--raw", default=False, action="store_true", help="use the raw audio"
    )
    parser.add_argument(
        "--speed-perturb",
        action="store_true",
        default=False,
        help="apply speed perturbation on wave file",
    )
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
    # text and dictionary settings
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["word", "bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="share the tokenizer and dictionary of the transcription and translation",
    )
    parser.add_argument(
        "--add-src", action="store_true", help="add the src text for st task"
    )
    parser.add_argument(
        "--asr-prefix", type=str, default=None, help="prefix of the asr dict"
    )
    parser.add_argument(
        "--st-spm-prefix", type=str, default=None, help="prefix of the existing st dict"
    )
    parser.add_argument(
        "--lowercase-src", action="store_true", help="lowercase the source text"
    )
    parser.add_argument(
        "--rm-punc-src",
        action="store_true",
        help="remove the punctuation of the source text",
    )
    parser.add_argument("--tokenizer", action="store_true", help="use tokenizer txt")

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
