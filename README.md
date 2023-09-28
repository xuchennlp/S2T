# Speech-to-Text (S2T) toolkit

## Overview

This repository is an extension of the [Fairseq toolkit](https://github.com/pytorch/fairseq) specialized for speech-to-text (S2T) generation tasks. This toolkit provides comprehensive support for Automatic Speech Recognition (ASR), Machine Translation (MT), and Speech Translation (ST).

## Features

- Complete recipes: Kaldi-style recipe support for ASR, MT, and ST tasks, ensuring a smooth workflow.
- Various configurations: An extensive collection of YAML configuration files to customize models for different tasks and scenarios.
- Easy reproduction: The comprehensive support of methods in our papers, including SATE, PDS, CTC-NAST, BIL-CTC, and more
- Multiple inference strategies: Greedy decoding, beam search, CTC decoding, CTC rescoring, and more
- More features can be found in the **run.sh** file.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/xuchennlp/S2T.git
    ```

2. Navigate to the project directory and install the required dependencies:

    ```bash
    cd S2T
    pip install -e .
    ```

    Our version: python 3.8, pytorch 1.11.0.

## Quick Start

1. Download your dataset and process it into the format of MUST-C dataset.
2. Run the shell script **run.sh** in the corresponding directory as follows:

   ```bash
   # Set ST_DIR environment variable as the parent directory of S2T directory
   export ST_DIR=/path/to/S2T/..
   cd egs/mustc/st/
   ./run.sh --stage 0 --stop_stage 2
   ```

- Stage 0 performs the data processing, including feature extraction of audios (Not required in MT), vocabulary generation, training and testing files generation.
- Stage 1 performs the model training, where multiple choices are supported.
- Stage 2 performs the model inference, where multiple strategies are supported.
- All details are available in **run.sh**.

## Reproduction of our methods

### SATE: Stacked Acoustic and Textual Encoding (ACL 2021)

**Paper**: [Stacked Acoustic-and-Textual Encoding: Integrating the Pre-trained Models into Speech Translation Encoders](https://aclanthology.org/2021.acl-long.204/)

**Highlights**: an simple and effective methods to utilize the pre-trained ASR and MT models to improve the end-to-end ST model; introducing the adapter to bridge the pre-trained encoders

Here is an example on the MUST-C ST dataset.

```bash
cd egs/mustc/st/
./run.sh --stage 0 --stop_stage 2 --train_config reproduction_sate
```

### PDS: Progressive Down-Sampling (ACL 2023 findings)

**Paper**: [Bridging the Granularity Gap for Acoustic Modeling](https://aclanthology.org/2023.findings-acl.688/)

**Highlights**: an effective method to facilitate the convergence of S2T tasks by increasing the modeling granularity of acoustic representations

Here is an example on the MUST-C ST dataset. This method also supports the ASR task.

```bash
cd egs/mustc/st/
./run.sh --stage 0 --stop_stage 2 --train_config reproduction_pds
```

### NAST: Non-Autoregressive Speech Translation (ACL 2023)

**Paper**: [CTC-based Non-autoregressive Speech Translation](https://aclanthology.org/2023.acl-long.744/)

**Highlights**: a non-autoregressive modeling method that only relies on the CTC inference and achieves the comparable results with the autoregressive methods

Here is an example on the MUST-C ST dataset.

```bash
cd egs/mustc/st/
# Non-autoregressive modeling
./run.sh --stage 0 --stop_stage 2 --train_config reproduction_nast
# Autoregressive modeling
./run.sh --stage 0 --stop_stage 2 --train_config reproduction_ctc_aug
```

### BiL-CTC: Bilingual CTC (Submitted to ICASSP 2024)

**Paper**: [Bridging the Gaps of Both Modality and Language: Synchronous Bilingual CTC for Speech Translation and Speech Recognition](https://arxiv.org/abs/2309.12234/)

**Highlights**: introducing both cross-modal and cross-lingual CTC for S2T tasks and developing an novel implementation strategy called Synchronous BiL-CTC that outperforms the traditional progressive strategy (the implementation in NAST)

Here is an example on the MUST-C ST dataset.

```bash
cd egs/mustc/st/
# Progressive BiL-CTC
./run.sh --stage 0 --stop_stage 2 --train_config reproduction_bil_ctc_progressive
# Synchronous BiL-CTC
./run.sh --stage 0 --stop_stage 2 --train_config reproduction_bil_ctc_synchronous
```

## Acknowledgments

- Fairseq community for the base toolkit
- ESPnet community for the base toolkit
- NiuTrans Team for their contributions and research

Finally, thank you to everyone who has helped me during my research career.
I sincerely hope that everyone can enjoy the pleasure of research

## Feedback

If you have any questions, feel free to contact xuchennlp[at]outlook.com.