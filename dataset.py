import copy
import pickle
import numpy as np
import sentencepiece as spm
from collections import Counter
import random


def load(file_name, include_label=False):
    label_lit = []
    text = []
    with open(file_name)as f:
        data = f.readlines()
    for d in data:
        t = []
        if include_label:
            label, sentences = d.strip().split('\t')
            sentences = sentences.split('|||')
            l_index = [int(l)-1 for l in label.split(',')]
            label = np.array([1 if i in l_index else 0 for i in range(len(sentences))], dtype=np.int32)
            label_lit.append(label)
        else:
            sentences = d.strip().split('|||')
            label_lit.append(np.array([0 for _ in range(len(sentences))], dtype=np.int32))
        for sentence in sentences:
            t.append(sentence.split(' '))
        text.append(t)
    return label_lit, text


def data_size(file_name):
    with open(file_name, 'r')as f:
        size = sum([1 for _ in f.readlines()])
    return size


def load_pickle(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


class VocabNormal:
    def __init__(self):
        self.vocab = None
        self.reverse_vocab = None

    def build(self, file_name, include_label, initial_vocab, vocab_size, freq=0):
        self.vocab = self._build_vocab(file_name, include_label, initial_vocab, vocab_size, freq)

    def _build_vocab(self, file_name, include_label, initial_vocab, vocab_size, freq=0):
        _, documents = load(file_name, include_label=include_label)

        vocab = copy.copy(initial_vocab)
        word_count = Counter()
        words = []
        for i, doc in enumerate(documents):
            for sentence in doc:
                words.extend(sentence)
            # 10000文書ごとにCounterへ渡す
            if i % 10000 == 0:
                word_count += Counter(words)
                words = []
        else:
            word_count += Counter(words)

        for w, c in word_count.most_common():
            if len(vocab) >= vocab_size:
                break
            if c <= freq:
                break
            if w not in vocab:
                vocab[w] = len(vocab)
        return vocab

    def load(self, vocab_file):
        self.vocab = load_pickle(vocab_file)

    def word2id(self, sentence, sos=False, eos=False):
        vocab = self.vocab
        sentence_id = [vocab.get(word, self.vocab['<unk>']) for word in sentence]

        if sos:
            sentence_id.insert(0, self.vocab['<s>'])
        if eos:
            sentence_id.append(self.vocab['</s>'])

        return np.array(sentence_id, dtype=np.int32)

    def set_reverse_vocab(self):
        reverse_vocab = {}
        for k, v in self.vocab.items():
            reverse_vocab[v] = k
        self.reverse_vocab = reverse_vocab

    def id2word(self, sentence_id):
        sentence = [self.reverse_vocab.get(word, '<unk>') for word in sentence_id]
        sentence = ' '.join(sentence)
        return sentence


class VocabSubword:
    def __init__(self):
        self.vocab = None

    def build(self, file_name, model_name, vocab_size):
        self.vocab = self._build_vocab(file_name, model_name, vocab_size)

    def _build_vocab(self, text_file, model_name, vocab_size):
        args = '''
                --control_symbols=<eod> 
                --input={} 
                --model_prefix={} 
                --vocab_size={} 
                --hard_vocab_limit=false''' \
            .format(text_file, model_name, vocab_size)
        spm.SentencePieceTrainer.Train(args)
        sp = spm.SentencePieceProcessor()
        sp.Load(model_name + '.model')
        return sp

    def load(self, vocab_file):
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab_file)
        self.vocab = sp

    def word2id(self, sentence, sos=False, eos=False):
        sp = self.vocab
        sentence_id = sp.EncodeAsIds(' '.join(sentence))

        if sos:
            sentence_id.insert(0, sp.PieceToId('<s>'))
        if eos:
            sentence_id.append(sp.PieceToId('</s>'))
        return np.array(sentence_id, dtype=np.int32)

    def id2word(self, sentence_id):
        return self.vocab.DecodeIds(sentence_id.tolist())


class Iterator:
    def __init__(self, src_file, trg_file, src_vocab, trg_vocab, batch_size, sort=True, shuffle=True, include_label=True):
        self.label, self.src = load(src_file, include_label)
        _, self.trg = load(trg_file)

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.batch_size = batch_size

        self.sort = sort
        self.shuffle = shuffle

    def _convert(self, src, trg, label):
        src_id = [self.src_vocab.word2id(s) for s in src]
        trg_sos = self.trg_vocab.word2id(trg[0], sos=True)
        trg_eos = self.trg_vocab.word2id(trg[0], eos=True)
        return src_id, trg_sos, trg_eos, label

    def generate(self, batches_per_sort=10000):
        src, trg, label = self.src, self.trg, self.label
        batch_size = self.batch_size

        data = []
        for s, t, l in zip(src, trg, label):
            data.append(self._convert(s, t, l))

            if len(data) != batch_size * batches_per_sort:
                continue

            if self.sort:
                data = sorted(data, key=lambda x: len(x[0]), reverse=True)
            batches = [data[b * batch_size: (b + 1) * batch_size]
                       for b in range(batches_per_sort)]

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                yield batch

            data = []

        if len(data) != 0:
            if self.sort:
                data = sorted(data, key=lambda x: len(x[0]), reverse=True)
            batches = [data[b * batch_size: (b + 1) * batch_size]
                       for b in range(int(len(data) / batch_size) + 1)]

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                # 補足: len(data) == batch_sizeのとき、batchesの最後に空listができてしまうための対策
                if not batch:
                    continue
                yield batch




"""
def make_vocab_sp(text_file, model_name, vocab_size):
    args = '''
            --control_symbols=<eod> 
            --input={} 
            --model_prefix={} 
            --vocab_size={} 
            --hard_vocab_limit=false'''\
        .format(text_file, model_name, vocab_size)
    spm.SentencePieceTrainer.Train(args)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_name + '.model')
    return sp


class VocabSubword:
    def __init__(self):
        self.src_vocab = None
        self.trg_vocab = None

    def make_vocab(self, src_file, trg_file, save_dir, vocab_size):
        self.src_vocab = make_vocab_sp(src_file, save_dir + 'src_vocab.subword', vocab_size)
        self.trg_vocab = make_vocab_sp(trg_file, save_dir + 'trg_vocab.subword', vocab_size)

    def load_vocab(self, src_vocab_file, trg_vocab_file):
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(src_vocab_file)
        trg_sp.Load(trg_vocab_file)
        self.src_vocab = src_sp
        self.trg_vocab = trg_sp

    def convert2label(self, data):
        src_sp = self.src_vocab
        trg_sp = self.trg_vocab

        dataset_label = []
        for d in data:
            src, trg = d[0], d[1]
            src =     [self._convert2label(' '.join(sentence), src_sp, eos=src_sp.PieceToId('</s>')) for sentence in src]
            trg_sos = [self._convert2label(' '.join(sentence), trg_sp, sos=trg_sp.PieceToId('<s>')) for sentence in trg]
            trg_eos = [self._convert2label(' '.join(sentence), trg_sp, eos=trg_sp.PieceToId('</s>')) for sentence in trg]
            src[-1][-1] = src_sp.PieceToId('<eod>')
            trg_eos[-1][-1] = trg_sp.PieceToId('<eod>')
            dataset_label.append((src, trg_sos, trg_eos))
        return dataset_label

    def _convert2label(self, words, sp, sos=None, eos=None):
        word_labels = sp.EncodeAsIds(words)
        if sos is not None:
            word_labels.insert(0, sos)
        if eos is not None:
            word_labels.append(eos)
        return np.array(word_labels, dtype=np.int32)

    def label2word(self, sentence):
        return self.trg_vocab.DecodeIds(sentence)
"""


if __name__ == '__main__':
    initial_vocab = {'<unk>': 0, '<s>':1, '</s>': 2}

    src_file = '../quelabel'
    trg_file = '../anslabel'

    src_vocab = VocabNormal(src_file, True, initial_vocab, 100, 0)
    # print('src')
    # print(sorted(src_vocab.vocab.items(), key=lambda x:x[1]))
    trg_vocab = VocabNormal(trg_file, False, initial_vocab, 100, 0)
    # print('trg')
    # print(sorted(src_vocab.vocab.items(), key=lambda x: x[1]))

    iter = Iterator(src_file, trg_file, src_vocab, trg_vocab, 2, sort=False, reverse=False, shuffle=False)

    for i in iter.generate():
        print('a', len(i))
        for ii in i:
            print(ii)

