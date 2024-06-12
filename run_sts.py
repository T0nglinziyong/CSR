import json
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import datasets
from datasets import Dataset
import numpy as np
import transformers
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PrinterCallback,
    Trainer,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers import default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from utils.progress_logger import LogCallback
from utils.sts.dataset_preprocessing import get_preprocessing_function, get_add_supervision_function_
from utils.sts.modeling_utils import DataCollatorWithPadding, get_model
from utils.sts.triplet_trainer import TripletTrainer
from utils.visualize_attention import visual_score

from make_test_submission import main as submission
from matplotlib import pyplot as plt
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import os   
import csv
'''
seed_value = 666
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)'''

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_DISABLED"] = "true"
#np.set_printoptions(precision=2, suppress=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
test_predict_file = "test_prediction.json"
test_predict_file_2 = "test_prediction_2.json"
email_address = "liujunhan@mails.tsinghua.edu.cn"
contrastive_objectives = {"triplet", "triplet_mse", "info", "info_mse"}
objective_set = contrastive_objectives.union({"mse"})

sentence1_key, sentence2_key, condition_key, similarity_key = (
    "sentence1",
    "sentence2",
    "condition",
    "label",
)

train_supervised_file = "data/supervision/gpt_prompt_train.json"
eval_supervised_file = "data/supervision/gpt_prompt_eval.json"
sentence1_keyword, sentence2_keyword, condition_keyword = (
    "sentence1_keyword",
    "sentence2_keyword",
    "condition_keyword",
)

train_key, eval_key, test_key = (
    "train",
    "val",
    "test"
)

def compute_metrics(output: EvalPrediction):
    preds = (
        output.predictions[0]
        if isinstance(output.predictions, tuple)
        else output.predictions
    )
    preds = np.squeeze(preds)
    return {
        "mse": ((preds - output.label_ids) ** 2).mean().item(),
        "pearsonr": pearsonr(preds, output.label_ids)[0],
        "spearmanr": spearmanr(preds, output.label_ids)[0],
    }


@dataclass
class TrainingArguments(HFTrainingArguments):
    log_time_interval: int = field(
        default=15,
        metadata={
            "help": (
                "Log at each `log_time_interval` seconds. "
                "Default will be to log every 15 seconds."
            )
        },
    )
    show_example: Optional[int] = field(
        default=None, metadata={}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

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
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
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
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    # Dataset specific arguments
    max_similarity: Optional[float] = field(
        default=None, metadata={"help": "Maximum similarity score."}
    )
    min_similarity: Optional[float] = field(
        default=None, metadata={"help": "Minimum similarity score."}
    )
    condition_only: Optional[bool] = field(
        default=False, metadata={"help": "Only use condition column."}
    )
    sentences_only: Optional[bool] = field(
        default=False, metadata={"help": "Only use sentences column."}
    )
    
    use_supervision: Optional[bool] = field(
        default=False, metadata={"help": "Use gpt prompt as supervision."}
    )

    def __post_init__(self):
        validation_extension = self.validation_file.split(".")[-1]
        if self.train_file is not None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            assert (
                train_extension == validation_extension
            ), "`train_file` and `validation_file` should have the same extension."
        if self.test_file is not None:
            test_extension = self.test_file.split(".")[-1]
            assert test_extension in [
                "csv",
                "json",
            ], "`test_file` should be a csv or a json file."
            assert (
                test_extension == validation_extension
            ), "`test_file` and `validation_file` should have the same extension."


@dataclass
class ModelArguments:
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
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    objective: Optional[str] = field(
        default="mse",
        metadata={
            "help": "Objective function for training. Options:\
            1) regression: Regression task (uses MSELoss).\
            2) classification: Classification task (uses CrossEntropyLoss).\
            3) triplet: Regression task (uses QuadrupletLoss).\
            4) triplet_mse: Regression task uses QuadrupletLoss with MSE loss."
        },
    )
    # What type of modeling
    encoding_type: Optional[str] = field(
        default="cross_encoder",
        metadata={
            "help": "What kind of model to choose. Options:\
            1) cross_encoder: Full encoder model.\
            2) bi_encoder: Bi-encoder model.\
            3) tri_encoder: Tri-encoder model."
        },
    )
    # Pooler for bi-encoder
    pooler_type: Optional[str] = field(
        default="cls",
        metadata={
            "help": "Pooler type: Options:\
            1) cls: Use [CLS] token.\
            2) avg: Mean pooling."
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={"help": "Freeze encoder weights."}
    )
    transform: Optional[bool] = field(
        default=False,
        metadata={"help": "Use a linear transformation on the encoder output"},
    )
    triencoder_head: Optional[str] = field(
        default="hadamard",
        metadata={
            "help": "Tri-encoder head type: Options:\
            1) hadamard: Hadamard product.\
            2) transformer: Transformer."
        },
    )

    routing_start: Optional[int] = field(
        default=19, metadata={"help": "The start of routing layer."}
    )

    routing_end: Optional[int] = field(
        default=24, metadata={"help": "The end of routing layer."}
    )

    router_type: Optional[int] = field(
        default=0,
        metadata={
            "help": "Router type: Options: \
            0-1) Use additional router model to get score\
            2) Average the weights of all words attended to by the attention_prob matrix.\
            3) Average the weights of [condition cls] token attended to by the attention_prob matrix.\
            "
        }
    )
    use_output: Optional[bool] = field(default=False, )
    use_attn: Optional[bool] = field(default=True, )

    mask_type: Optional[int] = field(
        default=0,
        metadata={
            "help": "Attention type for base model \
            Options: 0-6. See code for more details"
        }
    )

    mask_type_2: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention type for router model \
            Option: 0-6. Function is same as mask_type \
            None: use mask_type same as base model"
        }
    )

    use_condition: Optional[str] = field(
        default="True",
        metadata={
            "help": "use condition type: Options:\
            1) True: Do attention product.\
            2) False: Do n-to-1 linear .\
            3) None: return  1/ word_count "
        },
    )

    temperature: Optional[float] = field(default=1.0)

    layer_super: Optional[int] = field(default=23, )

    margin: Optional[float] = field(
        default=0.1, metadata={"help": "The margin of Ranking Loss(if used)."}
    )

    def __post_init__(self):
        assert self.temperature > 0
        flag = self.use_condition.lower()
        self.use_condition = True if flag == "true" else False if flag == "false" else None


def get_parser():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.log_level = "info"
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if model_args.objective in {"triplet", "triplet_mse"}:
        training_args.dataloader_drop_last = True
        training_args.per_device_eval_batch_size = 2
    logger.info("Training/evaluation parameters %s" % training_args)
    return model_args, data_args, training_args


def read_file(file_name):
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
    elif file_name.endswith(".json"):
        df = pd.read_json(file_name)
    else:
        raise Exception("invlid type of file name!")
    return df


def load_files(file_names, filter_function=None, process_function=None, keyword=None):
    if file_names is None:
        return None
    elif isinstance(file_names, str):
        if os.path.isdir(file_names):
            # 获取文件夹中所有的项（文件和子目录）
            items = os.listdir(file_names)
            # 过滤掉子目录，只保留文件
            file_names = [item for item in items if os.path.isfile(os.path.join(file_names, item) and 
                                                               (keyword is None or keyword in item))]
        elif os.path.isfile(file_names):
            file_names = [file_names]

    datasets = []

    for file_name in file_names:
        df = read_file(file_name)

        # to_dict方法将DataFrame转换为一个字典，其中每个键对应DataFrame中的一行，每个值是包含该行数据的字典。
        # 参数orient='records'指定了字典的格式，意味着字典的键将是行标签，而值是与这些行对应的记录列表
        data_dict = df.to_dict(orient='records') 
        datasets.append(Dataset.from_list(data_dict))
    
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(*datasets)

    if filter_function is not None:
        dataset.filter(filter_function)

    if process_function is not None:
        dataset = dataset.map(
            process_function,
            batched=True,
            #remove_columns=dataset.column_names,
        )
    return dataset


def load_dataset(data_args, training_args, filter_function=None, process_function=None):
    # 加载数据集
    train_dataset = load_files(data_args.train_file, filter_function, process_function, train_key)
    eval_dataset = load_files(data_args.validation_file, filter_function, process_function, eval_key)
    predict_dataset = load_files(data_args.test_file, filter_function, process_function, test_key)

    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if predict_dataset is None:
            raise ValueError("--do_predict requires a test dataset")
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    return train_dataset, eval_dataset, predict_dataset


def get_model_and_tokenizer(model_args):
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=1,
        # finetuning_task=None,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_cls = get_model(model_args)
    config.update(
        {
            "use_auth_token": model_args.use_auth_token,
            "model_revision": model_args.model_revision,
            "cache_dir": model_args.cache_dir,
            "model_name_or_path": model_args.model_name_or_path,
            "objective": model_args.objective,
            "pooler_type": model_args.pooler_type,
            "transform": model_args.transform,
            "triencoder_head": model_args.triencoder_head,
            "routing_start":model_args.routing_start,
            "routing_end":model_args.routing_end,
            "router_type":model_args.router_type,
            "use_output":model_args.use_output,
            "use_attn":model_args.use_attn,
            "mask_type":model_args.mask_type,
            "mask_type_2":model_args.mask_type_2,
            "use_condition":model_args.use_condition,
            "temperature":model_args.temperature,
            "layer_score":model_args.layer_super,
            "margin":model_args.margin,
        }
    )
    model = model_cls(config=config)
    #model.load_state_dict(torch.load('.//output//princeton-nlp__sup-simcse-roberta-large//enc_bi_encoder___lr_1e-5__wd_0.1__trans_False__obj_mse__tri_None__s_42//pytorch_model.bin'), strict=False)
    if model_args.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model, tokenizer


def get_trainer(model, tokenizer, model_args, data_args, training_args, train_dataset, eval_dataset):
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            pad_token_type_id=tokenizer.pad_token_type_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Initialize our Trainer
    trainer_cls = (
        TripletTrainer
        if model_args.objective in contrastive_objectives
        else Trainer
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_spearmanr",  # 根据验证集上的最小损失选择最佳模型
        #greater_is_better=True,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LogCallback)
    return trainer, data_collator


def show_examples_from_bi_encoder(trainer, train_dataset, tokenizer, num_example, path):
    logger.info("*** Showing Examples ***")

    sample_ids = random.sample(range(len(train_dataset)), num_example)
    samples = train_dataset.select(sample_ids)
    predictions = trainer.predict(samples.remove_columns("labels")).predictions

    split_posi = len(predictions[-1][0]) // 2
    fig_path = path + "/figures/"

    for id, input_ids_1, input_ids_2, input_ids_3, label,\
        predict, attention_1, attention_2  in \
        zip(range(len(samples['input_ids'])), samples["input_ids"], samples["input_ids_2"], samples["input_ids_3"], samples['labels'],
                predictions[0], predictions[-2], predictions[-1], #predictions[5], 
                ): 
        visual_score(input_ids_3,input_ids_1, attention_1, split_posi, tokenizer, 2*id, fig_path, label, predict)
        visual_score(input_ids_3,input_ids_2, attention_2, split_posi, tokenizer, 2*id+1, fig_path, label, predict)
        visual_score(input_ids_3,input_ids_1, attention_1, split_posi, tokenizer, 2*id, "./figures/", label, predict)
        visual_score(input_ids_3,input_ids_2, attention_2, split_posi, tokenizer, 2*id+1, "./figures/", label, predict)
        print(f'------------------sample {id}--------------------')
        try:
            print(f"condition: {tokenizer.decode(input_ids_3)}        key: {samples[id]['condition_key']}")
            print(f"sentence1: {tokenizer.decode(input_ids_1)}")
            print(f"sentence2: {tokenizer.decode(input_ids_2)}")
            print(f"key1: {samples[id]['sentence1_key']}")
            print(f"key2: {samples[id]['sentence2_key']}")

            print(f"label: {label}, prediction: {predict}")
        except:
            print(f"condition: {tokenizer.decode(input_ids_3)}")
            print(f"sentence1: {tokenizer.decode(input_ids_1)}")
            print(f"sentence2: {tokenizer.decode(input_ids_2)}")
            print(f"label: {label}, prediction: {predict}")


def main():
    torch.autograd.set_detect_anomaly(False)

    model_args, data_args, training_args = get_parser()

    #raw_datasets = get_data(model_args, data_args, training_args)

    model, tokenizer = get_model_and_tokenizer(model_args)
    
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    #set_seed(training_args.seed)

    
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    
    data_args.min_similarity, data_args.max_similarity = (1, 5)
 

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d."
            % (
                data_args.max_seq_length,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        )
    
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    

    preprocess_function = get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        condition_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        scale=(data_args.min_similarity, data_args.max_similarity)
        if model_args.objective in objective_set
        else None,
        condition_only=data_args.condition_only,
        sentences_only=data_args.sentences_only,
    )
    
    

    #train_dataset, eval_dataset, predict_dataset = get_preprocessed_data(raw_datasets, preprocess_function, training_args, data_args)
    
    train_dataset, eval_dataset, predict_dataset = load_dataset(data_args, training_args, None, process_function=preprocess_function)
    #eval_dataset2 = load_files("data/simple_example.csv", process_function=preprocess_function)
    #eval_dataset3 = load_files("data/hard_example.csv", process_function=preprocess_function)

    if data_args.use_supervision and model_args.encoding_type == 'bi_encoder':
        get_add_supervision_function = get_add_supervision_function_(
            tokenizer,
            sentence1_key,
            sentence2_key,
            condition_key,
            sentence1_keyword,
            sentence2_keyword,
            condition_keyword,
        )
        train_dataset = train_dataset.map(get_add_supervision_function(train_supervised_file) ,batched=False, with_indices=True)
        eval_dataset = eval_dataset.map(get_add_supervision_function(eval_supervised_file) ,batched=False, with_indices=True)


    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    trainer, data_collator = get_trainer(model, tokenizer, model_args, data_args, training_args, train_dataset, eval_dataset)
    
    if training_args.do_train: 
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation
    combined = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        combined.update(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined)
        if training_args.do_train:
            metrics = trainer.evaluate(
                eval_dataset=train_dataset, metric_key_prefix="train"
            )
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["train_samples"] = min(max_eval_samples, len(train_dataset))
            trainer.log_metrics("train", metrics)
            combined.update(metrics)
            trainer.save_metrics("train", combined)
    
    if training_args.show_example is not None and False:
        show_examples_from_bi_encoder(trainer, eval_dataset, tokenizer, training_args.show_example, training_args.output_dir) 

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        ).predictions[0]
        predictions = (
            np.squeeze(predictions)
            if model_args.objective in objective_set
            else np.argmax(predictions, axis=1)
        )
        predictions = dict(enumerate(predictions.tolist()))
        output_predict_file = os.path.join(training_args.output_dir, test_predict_file)
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w", encoding="utf-8") as outfile:
                json.dump(predictions, outfile)
            with open(test_predict_file, "w", encoding="utf-8") as outfile:
                json.dump(predictions, outfile)

        checkpoint_path = "./" + training_args.output_dir + "/checkpoint-709/pytorch_model.bin"
        trainer.model.load_state_dict(torch.load(checkpoint_path))
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict_2"
        ).predictions[0]
        predictions = (
            np.squeeze(predictions)
            if model_args.objective in objective_set
            else np.argmax(predictions, axis=1)
        )
        predictions = dict(enumerate(predictions.tolist()))
        output_predict_file = os.path.join(training_args.output_dir, test_predict_file_2)
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w", encoding="utf-8") as outfile:
                json.dump(predictions, outfile)
            with open(test_predict_file_2, "w", encoding="utf-8") as outfile:
                json.dump(predictions, outfile)



    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "CSTS"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
    #submission(email_address, test_predict_file)
