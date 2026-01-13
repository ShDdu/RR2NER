#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : main.py
@Author  : HuYing
@Time    : 2022/12/26 13:33
@Description: 
"""

from config.main import *
from model.main import *
from dataloader.main import *
from Trainer.train_dev_test import *


def main():
    train_dataset = None
    dev_dataset = None
    test_dataset = None
    if config.do_train:
        logger.info("Loading Train Data")
        train_dataset = BaselineDataset(config, logger, set_type="train")
    if config.do_eval:
        logger.info("Loading Evaluate Data")
        dev_dataset = BaselineDataset(config, logger, set_type="test")
    if config.do_test:
        logger.info("Loading Test Data")
        test_dataset = BaselineDataset(config, logger, set_type="test")

    model = RR2NER(config)
    trainer = Trainer(model, config)
    if config.do_train:
        trainer.train(train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

    ner_p, ner_r, ner_f = 0.0, 0.0, 0.0
    if config.do_test:
        ner_p_best, ner_r_best, ner_f_best = trainer.test(test_dataset, dataset + "/best_model.bin")

    return ner_p_best, ner_r_best, ner_f_best


if __name__ == '__main__':

    # "ace05", "ace04", "genia", "JNLPBA", "BC2GM-IOB", "BC4CHEMD", "BC5CDR-chem-IOB", "BC5CDR-disease-IOB", "BC5CDR-IOB",
    dataset_list = ["ace05", "ace04", "genia"]  # "jnlpba", "bc5cdr", "ncbi-disease", "genia", "ace05", "ace04"
    table = pt.PrettyTable(["Task_name", "Precision", "Recall", "F1"])
    logger = get_logger("RR2NER_" + "_".join(dataset_list))

    for dataset in dataset_list:
        # dataset = dataset.lower()
        config = parse_args(dataset)
        logger.info(config)
        config.logger = logger

        set_seed(config.seed)

        # if config.use_word_embedding:
        config.word2id, config.pretrained_word_embed = load_embeddings(config.word_embedding_file_path)
        # if config.use_char_embedding:
        config.char2id, config.pos2id = load_char_pos_vocab("config/vocab.json")

        if torch.cuda.is_available():
            torch.cuda.set_device(config.device_id)
            logger.info("CUDA USED !!!")
        logger.info("Task_name: {} ({}), training !!! ".format(dataset, config.epochs))

        ner_p, ner_r, ner_f = main()
        table.add_row(["{}: ".format(dataset), "{:3.4f}".format(ner_p), "{:3.4f}".format(ner_r), "{:3.4f}".format(ner_f)])
        table.add_row(["-" * 10] * 4)
        logger.info("\n{}".format(table))
