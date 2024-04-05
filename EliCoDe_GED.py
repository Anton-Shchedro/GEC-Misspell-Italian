from argparse import Namespace
import os
import sys
import torch

from transformers import (XLMRobertaTokenizerFast, XLMRobertaConfig)

from EliCoDe.src.token_classification.transfomer_log import TransformerNERLogger
from EliCoDe.src.token_classification.task import (set_seed, predict, load_model)
from EliCoDe.src.token_classification.data_utils import (TransformerNerDataProcessor,
                                             transformer_convert_data_to_features)
from EliCoDe.src.token_classification.model import (XLMRobertaNerModel)
from EliCoDe.src.common_utils.common_io import json_dump, json_load
from EliCoDe.src.common_utils.common_log import create_logger

global_args = Namespace()

MODEL_CLASSES = {
    'xlm-roberta': (XLMRobertaConfig, XLMRobertaNerModel, XLMRobertaTokenizerFast)
}

elicode_path = "./EliCoDe/"

global_args.model_type = 'xlm-roberta'
global_args.pretrained_model = elicode_path + 'data/output/it/roberta_model/'
# resume training on a NER model if set it will overwrite pretrained_model
global_args.resume_from_model = None
global_args.config_name = None
global_args.tokenizer_name = None
global_args.data_dir = elicode_path + 'data/output/it/'
global_args.new_model_dir = elicode_path + 'data/output/it/roberta_model/'
global_args.predict_output_file = elicode_path + 'data/output/it/roberta_preds.txt'
global_args.seed = 13
global_args.max_seq_length = 512
global_args.model_selection_scoring = 'strict-f_score-1'
global_args.do_predict = True
global_args.log_file = elicode_path + 'data/output/it/log_roberta_test.txt'
global_args.progress_bar = True
global_args.log_lvl = "i"
global_args.eval_batch_size = 8
global_args.train_batch_size = 8
global_args.train_steps = -1
global_args.learning_rate = 1e-5
global_args.min_lr = 1e-6
global_args.num_train_epochs = 10
global_args.gradient_accumulation_steps = 1
global_args.warmup_ratio = 0.1
global_args.weight_decay = 0.0
global_args.adam_epsilon = 1e-8
global_args.max_grad_norm = 1.0
global_args.max_num_checkpoints = 3
global_args.early_stop = -1
global_args.focal_loss_gamma = 2
global_args.mlp_dim = 128
global_args.mlp_layers = 0
global_args.mlp_hidden_dim = 0
global_args.adversarial_training_method = None
global_args.use_crf = False
global_args.data_has_offset_information = False
global_args.save_model_core = False
global_args.overwrite_model_dir = False
global_args.do_train = False
global_args.do_lower_case = False
global_args.do_warmup = False
global_args.focal_loss = False
global_args.use_biaffine = False
global_args.fp16 = False

logger = TransformerNERLogger(global_args.log_file, global_args.log_lvl).get_logger()
global_args.logger = logger
global_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not global_args.tokenizer_name:
    global_args.tokenizer_name = global_args.pretrained_model

if not global_args.config_name:
    global_args.config_name = global_args.pretrained_model


def get_test_examples(text, ner_data_processor):
    nsents, unique_labels = [], set()
    txt = text.strip()
    sents = txt.split("\n\n")
    for i, sent in enumerate(sents):
        nsent, offsets, labels = [], [], []
        words = sent.split("\n")
        for j, word in enumerate(words):
            word_info = word.split(" ")
            if word_info[0] == "" or word_info[0] == " " or word_info[0] == "\t":
                continue
            nsent.append(word_info[0])
            labels.append("O")
        nsents.append((nsent, offsets, labels))
    return ner_data_processor._create_examples(nsents, 'test')


def output_bio(bio_data, sep=" "):
    # modified version of EliCoDe.common_io.output_bio
    sents = ""
    for sent in bio_data:
        for word in sent:
            line = sep.join(word)
            sents = sents + line + "\n"
        sents = sents + "\n"
    return sents


# Some modification of task._output_bio() from EliCoDe to be able pass text directly (not write to file)
def _output_bio(args, tests, preds):
    new_sents = []
    assert len(tests) == len(preds), "Expect {} sents but get {} sents in prediction".format(len(tests), len(preds))
    for example, predicted_labels in zip(tests, preds):
        tokens = example.text
        assert len(tokens) == len(predicted_labels), "Not same length sentence\nExpect: {} {}\nBut: {} {}".format(
            len(tokens), tokens, len(predicted_labels), predicted_labels)
        offsets = example.offsets
        if offsets:
            new_sent = [(tk, ofs[0], ofs[1], ofs[2], ofs[3], lb) for tk, ofs, lb in
                        zip(tokens, offsets, predicted_labels)]
        else:
            new_sent = [(tk, lb) for tk, lb in zip(tokens, predicted_labels)]
        new_sents.append(new_sent)

    # changed this part!
    return output_bio(new_sents)


def predict_elicode(text, args):
    set_seed(args.seed)

    ner_data_processor = TransformerNerDataProcessor()
    ner_data_processor.set_data_dir(args.data_dir)
    ner_data_processor.set_logger(args.logger)
    label2idx = json_load(os.path.join(args.new_model_dir, "label2idx.json"))

    num_labels = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    args.num_labels = num_labels
    args.label2idx = label2idx
    args.idx2label = idx2label

    # get config, model and tokenizer
    model_config, model_model, model_tokenizer = MODEL_CLASSES[args.model_type]

    args.config = model_config.from_pretrained(args.new_model_dir, num_labels=num_labels)
    args.tokenizer = model_tokenizer.from_pretrained(
        args.new_model_dir, do_lower_case=False, add_prefix_space=True)

    model = load_model(args)
    model.to(args.device)

    test_example = get_test_examples(text, ner_data_processor)
    test_features = transformer_convert_data_to_features(args,
                                                         input_examples=test_example,
                                                         label2idx=label2idx,
                                                         tokenizer=args.tokenizer,
                                                         max_seq_len=args.max_seq_length)

    predictions = predict(args, model, test_features)
    return _output_bio(args, test_example, predictions)

def elicode_in(text):
    result = []
    words = []
    pred = predict_elicode(text, global_args)
    sents = pred.split("\n\n")
    errors = []

    for line in sents:
        if not (line == "" or line == " " or line == "\n"):
            words_bio = line.split("\n")
            try:
                for word in words_bio:
                    if word.split()[1] == "O":
                        words.append(word.split()[0])
                        result.append(True)
                    elif word.split()[1] == "B-Error":
                        words.append(word.split()[0])
                        result.append(False)
                    else:
                        errors.append(line)
            except:
                print(line)
                raise Exception()
    return words, result