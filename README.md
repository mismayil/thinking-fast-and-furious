# Thinking Fast and Furious: Driving with Language
Mahammad (Mete) Ismayilzada (337396), Chun-Tzu Chang (351986), Arina Rak (330939)
CS-503 Final Project Code

## Overview
This repo contains the code for our final project that tackles the [Driving with Language Challenge](https://huggingface.co/spaces/AGC2024/driving-with-language-2024) as part of the [Autonomous Grand Challenge](https://opendrivelab.com/challenge2024/) put out by OpenDriveLab in conjunction with CVPR 2024 Workshop.

## Repo structure
- `drivelm`: This folder contains a fork of the original challenge repo [DriveLM](https://github.com/OpenDriveLab/DriveLM). Minor modifications have been made for data preparation and submission.
- `experiments`: This folder contains all the code for our experiments. Here is the breakdown of subfolders and individual files:
    - `baseline`: This folder contains the code and data for baseline (llama2-adapter) experiments
    - `data`: This folder contains the training and test data. Note that due to the large size of the training and test files, they are omitted and instead a readme file describing the data downloading and preparating steps is included inside their respective folder.
    - `metrics`: This folder contains the metric results on the sample [test set](experiments/data/test/test_eval.json) provided to us by the challenge and organized by experiment setting (zero-shot vs. finetuned) and the name of the files describe the experiment settings (e.g. model, redcircle applied or not etc.). Content of these files can be reproduced using the [report_metrics.sh](experiments/report_metrics.sh) script and a corresponding results file from the [outputs](experiments/outputs) directory.
    - `outputs`: This folder contains the evaluation results on the sample [test set](experiments/data/test/test_eval.json) and file names describe the experiment settings.
    - `submissions`: This folder lists our final submissions (evaluated on the question-only [test set](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/v1_1_val_nus_q_only.json))
    - `notebooks`: This folder contains various raw experiment and analysis notebooks.
    - `finetune.py`: This script contains the code for fine-tuning the models. It accepts customizable arguments for different fine-tuning experiments. Run `python finetune.py -h` for further information.
    - `evaluate.py`: This script contains the code for evaluating models on the test sets. It accepts customizable arguments for different fine-tuning experiments. Run `python evaluate.py -h` for further information.
    - `metrics.py`: This file contains our custom metrics code reimplemented based on [challenge metrics](drivelm/challenge/evaluation.py)
    - `report_metrics.sh`: This script is used to report the metric results using the challenge's evaluation suite.

## Experiments
In order to reproduce the experiments, first install the required packages. We recommend using python>=3.10.
```
pip install -r requirements.txt
```

Once the environment is setup, each finetuning experiment can be run using `finetune.py` script with appropriate parameters for the respective experiment. By default, it uses data parallelism. Refer to [data](experiments/data/) folder on how to prepare the data for training and evaluation.

```
python finetune.py -h
```

In order to evaluate the model on the test set, run the `evaluate.py` script with appropriate arguments.

```
python evaluate.py -h
```

Once the evaluation is done, metrics can be reported using `report_metrics.sh` script. Note that this script requires OpenAI API Key as it uses ChatGPT for evaluation. In addition, the challenge uses the [language-evaluation] repo for other metrics, so it needs to be downloaded and installed as well.

```
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"
```

Then the metrics script can be run as below. It will output both the challenge evaluation results, and our metrics results.
```
OPENAI_API_KEY=<key> ./report_metrics.sh <path to predictions file>
```