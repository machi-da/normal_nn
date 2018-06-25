import argparse
import configparser
import os
import glob
import logging
import numpy as np
from logging import getLogger
import chainer
import dataset
import convert

import evaluate
from model import Multi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_dir = args.model_dir
    """LOAD CONFIG FILE"""
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    assert len(config_files) == 1, 'Put only one config file in the directory'
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)
    """LOGGER"""
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = model_dir + 'log.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('[Test start] logging to {}'.format(log_file))
    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    class_size = int(config['Parameter']['class_size'])
    dropout_ratio = float(config['Parameter']['dropout'])
    vocab_type = config['Parameter']['vocab_type']
    coefficient = float(config['Parameter']['coefficient'])
    """TEST DETAIL"""
    gpu_id = args.gpu
    batch_size = args.batch
    model_file = args.model
    """DATASET"""
    test_src_file = config['Dataset']['test_src_file']
    correct_txt_file = config['Dataset']['correct_txt_file']

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {0}'.format(test_data_size))
    if vocab_type == 'normal':
        src_vocab = dataset.VocabNormal()
        src_vocab.load(model_dir + 'src_vocab.normal.pkl')
        src_vocab.set_reverse_vocab()
        trg_vocab = dataset.VocabNormal()
        trg_vocab.load(model_dir + 'trg_vocab.normal.pkl')
        trg_vocab.set_reverse_vocab()

        sos = np.array([src_vocab.vocab['<s>']], dtype=np.int32)
        eos = np.array([src_vocab.vocab['</s>']], dtype=np.int32)

    elif vocab_type == 'subword':
        src_vocab = dataset.VocabSubword()
        src_vocab.load(model_dir + 'src_vocab.sub.model')
        trg_vocab = dataset.VocabSubword()
        trg_vocab.load(model_dir + 'trg_vocab.sub.model')

        sos = np.array([src_vocab.vocab.PieceToId('<s>')], dtype=np.int32)
        eos = np.array([src_vocab.vocab.PieceToId('</s>')], dtype=np.int32)

    src_vocab_size = len(src_vocab.vocab)
    trg_vocab_size = len(trg_vocab.vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    evaluater = evaluate.Evaluate(correct_txt_file)
    test_iter = dataset.Iterator(test_src_file, test_src_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False, include_label=False)
    """MODEL"""
    model = Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient)
    chainer.serializers.load_npz(model_file, model)
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TEST"""
    outputs = []
    labels = []
    for i, batch in enumerate(test_iter.generate(), start=1):
        batch = convert.convert(batch, gpu_id)
        output, label = model.predict(batch[0], sos, eos)
        for o, l in zip(output, label):
            outputs.append(trg_vocab.id2word(o))
            labels.append(l)
    rank_list = evaluater.rank(labels)
    single = evaluater.single(rank_list)
    multiple = evaluater.multiple(rank_list)
    logger.info('single: {} | {}'.format(single[0], single[1]))
    logger.info('multi : {} | {}'.format(multiple[0], multiple[1]))

    with open(model_file + '.hypo', 'w')as f:
        [f.write(o + '\n') for o in outputs]
    with open(model_file + '.attn', 'w')as f:
        [f.write('{}\n'.format(l)) for l in labels]


if __name__ == '__main__':
    main()