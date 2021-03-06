# Copyright 2021 Jason Weddington - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: These data classes wrap parameters required for running the AdapterHub version of
# the HuggingFace GLUE training script and were extracted from the main script to integrate
# into a Jupyter Notebook training process. These classes are identical to the original script
# except for being prefaced with "Task" to avoid collisions in the namespace.
# These classes are used when adapting a pre-trained model to a certain language domain.
# More info on training adapters: https://docs.adapterhub.ml/training.html

# THIS A DERIVATIVE WORK AS DEFINED IN SECTION 4 OF THE APACHE 2.0 LICENSE, ORIGINAL LICENSE FOLLOWS
# Original Work:
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/text-classification/run_glue_alt.py

# Original License:

# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field
from typing import Optional, Dict
import random
import itertools
from shutil import copyfile

from transformers import TrainingArguments, MultiLingAdapterArguments

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class TaskDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."


@dataclass
class TaskModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


def getParams(dictionary, limit):
    paramsList = [
        dict(zip(dictionary, v)) for v in itertools.product(*dictionary.values())
    ]
    random.shuffle(paramsList)

    if limit is not False:
        paramsList = paramsList[0 : min(limit, len(paramsList))]

    return paramsList


def initParse(dictionary: Dict, output_prefix=""):
    model = TaskModelArguments(model_name_or_path=dictionary.get("model_name_or_path"))

    data = TaskDataTrainingArguments(
        task_name=dictionary.get("task_name"),
        max_seq_length=dictionary.get("max_seq_length"),
        pad_to_max_length=dictionary.get("pad_to_max_length"),
    )

    training = TrainingArguments(
        adam_beta1=dictionary.get("adam_beta1"),
        adam_beta2=dictionary.get("adam_beta2"),
        adam_epsilon=dictionary.get("adam_epsilon"),
        learning_rate=dictionary.get("learning_rate"),
        fp16=dictionary.get("fp16"),
        warmup_ratio=dictionary.get("warmup_ratio"),
        warmup_steps=dictionary.get("warmup_steps"),
        weight_decay=dictionary.get("weight_decay"),
        do_train=dictionary.get("do_train"),
        do_eval=dictionary.get("do_eval"),
        per_device_train_batch_size=dictionary.get("per_device_train_batch_size"),
        num_train_epochs=dictionary.get("num_train_epochs"),  # CHANGE ME
        overwrite_output_dir=dictionary.get("overwrite_output_dir"),
        output_dir=f"./adapter/task/{output_prefix}{dictionary.get('task_name')}",
    )

    adapter = MultiLingAdapterArguments(
        train_adapter=True,
        adapter_config="pfeiffer",
    )

    return model, data, training, adapter


def copy_adapter_config(task_name: str, model_dir: str):
    """Copy the adapter config into the downloaded local model location"""

    config_location = f"./adapter/task/{task_name}/{task_name}"

    copyfile(
        src=f"{config_location}/adapter_config.json",
        dst=f"{model_dir}/.git/adapter_config.json",
    )
    copyfile(
        src=f"{config_location}/pytorch_adapter.bin",
        dst=f"{model_dir}/.git/pytorch_adapter.bin",
    )
