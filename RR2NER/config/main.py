#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Author  : HuYing
@Time    : 2022/12/26 22:03
@Description: 
"""
import argparse
from transformers import BertTokenizerFast, BertConfig
import logging
from utils.baseline import *

model_name_or_path_list = {"biobert": "/remote-home/TCCI26/bert/biobert_base",
                           "scibert": "/remote-home/TCCI26/bert/scibert_scivocab_uncased",
                           "pubmedbert": "/remote-home/TCCI26/bert/PubMedBERT_AbstractOnly"}


def parse_args(task_name="genia"):
    # Basic configuration
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=task_name, help="The name of the dataset task to train on.")
    parser.add_argument("--train_file", type=str, default="data/genia/train.json", help="A csv or a json file containing the training data.")
    parser.add_argument("--valid_file", type=str, default="data/genia/test.json", help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", type=str, default="data/genia/test.json", help="A csv or a json file containing the testing data.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to turn on the training model.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to turn on the evaluating model.")
    parser.add_argument("--do_test", type=bool, default=True, help="Whether to turn on the testing model.")
    parser.add_argument('--do_debug', type=bool, default=False)
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--model_name_or_path", type=str, default="chinese_rbt3_pytorch",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.", )
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate (after the potential warmup period) to use.", )
    parser.add_argument("--bert_learning_rate", type=float, default=5e-6, help="Initial learning rate (BERT) to use.", )
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument("--epochs", type=int, default=32, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], )
    parser.add_argument("--warmup_rate", type=float, default=0.1, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default="output", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="BERT Embedding hidden dropout prob.")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0, help="The upper norm of the parameter gradient for the group of networks.")

    parser.add_argument('--use_word_embedding', type=bool, default=True, help="Whether to increase the pre-training word vector")
    parser.add_argument('--word_embedding_file_path', type=str, default="config/glove/glove.6B.100d.txt", help="Word embedding file path")
    parser.add_argument('--bert_embedding_pooling_type', type=str, default="max", help="The integration of pre-training language models",
                        choices=["max", "mean"])
    parser.add_argument('--word_embedding_dim', type=int, default=100, help="The embedding dim of the glove word_embedding_file")
    parser.add_argument('--use_char_embedding', type=bool, default=True, help="Whether to increase the pre-training word vector")
    parser.add_argument('--use_pos_embedding', type=bool, default=True, help="Whether to increase the pre-training word vector")
    parser.add_argument('--char_embedding_dim', type=bool, default=50, help="Whether to increase the pre-training word vector")
    parser.add_argument('--pos_embedding_dim', type=bool, default=50, help="Whether to increase the pre-training word vector")
    parser.add_argument('--biaffine_size', type=int, default=128, help="Biaffine layer size.")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task_name == "genia" or args.task_name == "jnlpba":
        args.label_list = ['<pad>', "protein", "cell_type", "DNA", "cell_line", "RNA"]
    elif args.task_name == "ace04" or args.task_name == "ace05":
        args.label_list = ['<pad>', "PER", "ORG", "GPE", "LOC", "VEH", "WEA", "FAC"]
    elif args.task_name == "conll03":
        args.label_list = ['<pad>', "PER", "ORG", "LOC", "MISC"]
    elif args.task_name == "bc5cdr":
        args.label_list = ['<pad>', 'Entity']
    elif args.task_name == "ncbi-disease":
        args.label_list = ['<pad>', 'Disease']
    else:
        args.label_list = ['<pad>', "False", "True"]

    # 针对不同数据集使用不同的参数
    args = add_task_args(args)

    args.bert_config = BertConfig.from_pretrained(args.model_name_or_path)
    args.tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    return args


def add_task_args(config):
    try:
        with open(os.path.join("config", config.task_name + ".json"), "r", encoding="utf-8") as f:
            task_name_config = json.load(f)
            for k, v in task_name_config.items():
                if k in config.__dict__.keys() and task_name_config[k] != config.__dict__[k]:
                    tmp = config.__dict__[k]
                    config.__dict__[k] = task_name_config[k]
                    print("The parameter configuration {} ({}) has been updated to {}.".format(k, tmp, config.__dict__[k]))
                elif k not in config.__dict__.keys():
                    config.__dict__[k] = task_name_config[k]
                else:
                    continue
    except:
        print("No configuration file {}.json, use the default parameter configuration!".format(config.task_name))

    if config.__dict__["model_name_or_path"] in model_name_or_path_list:
        config.__dict__["model_name_or_path"] = model_name_or_path_list[config.__dict__["model_name_or_path"]]

    return config


def get_logger(dataset="ner"):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.exists("./log"):
        os.makedirs("./log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main():
    args = parse_args("genia")
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
