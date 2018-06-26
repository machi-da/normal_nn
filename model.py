import numpy as np
import chainer
from chainer import links as L
from chainer import functions as F


class WordEncoder(chainer.Chain):
    def __init__(self, n_vocab, embed, hidden, dropout):
        n_layers = 1
        super(WordEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed)
            self.Nlstm = L.NStepBiLSTM(n_layers, embed, hidden, dropout)
        self.hidden = hidden

    def __call__(self, hx, cx, xs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)
        """
        hy, cyの処理
        left-to-rightとright-to-leftの隠れ状態をsumしている
        shape:(2*layer数, batchの合計文数, hidden) -> (batchの合計文数, hidden)

        ysの処理
        ysはbatchの合計文数サイズのリスト
        shape:(単語数, 2*hidden) -> reshape:(単語数, 2, hidden) -> sum:(単語数, hidden)
        """
        hy = F.sum(hy, axis=0)
        cy = F.sum(cy, axis=0)
        ys = [F.sum(F.reshape(y, (-1, 2, self.hidden)), axis=1) for y in ys]

        return hy, cy, ys


class SentEncoder(chainer.Chain):
    def __init__(self, hidden, dropout):
        n_layers = 1
        super(SentEncoder, self).__init__()
        with self.init_scope():
            self.Nlstm = L.NStepBiLSTM(n_layers, hidden, hidden, dropout)
        self.hidden = hidden

    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.Nlstm(hx, cx, xs)
        """
        hy, cyの処理
        left-to-rightとright-to-leftの隠れ状態をsumしている
        shape:(2*layer数, batch, hidden) -> (batch, hidden)
        
        ysの処理
        ysはbatchサイズのリスト
        shape:(文数, 2*hidden) -> reshape:(文数, 2, hidden) -> sum:(文数, hidden)
        """
        hy = F.reshape(F.sum(hy, axis=0), (1, -1, self.hidden))
        cy = F.reshape(F.sum(cy, axis=0), (1, -1, self.hidden))
        ys = [F.sum(F.reshape(y, (-1, 2, self.hidden)), axis=1) for y in ys]

        return hy, cy, ys


class WordDecoder(chainer.Chain):
    def __init__(self, n_vocab, embed, hidden, dropout):
        n_layers = 1
        super(WordDecoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed)
            self.Nlstm = L.NStepLSTM(n_layers, embed, hidden, dropout)
            self.proj = L.Linear(hidden, n_vocab)
            # self.attention = Attention()
        self.dropout = dropout

    def __call__(self, hx, cx, xs, enc_hs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)
        ys_proj = [self.proj(F.dropout(y, self.dropout)) for y in ys]
        return hy, cy, ys_proj

'''
class Attention(chainer.Chain):
    def __init__(self):
        super(Attention, self).__init__()

    def __call__(self, dec_hs, enc_hs):
        """
        dec_hs: (単語数, hidden)
        enc_hs: (文数, hidden) encodeされた文ベクトル
        """

        print(dec_hs.shape, len(enc_hs), enc_hs[0].shape)
        exit()
        score = F.matmul(dec_hs, enc_hs, False, True)
        align = F.softmax(score, axis=1)
        attention = align.data
        cv = F.matmul(align, enc_hs)

        return cv, attention
'''


class LabelClassifier(chainer.Chain):
    def __init__(self, class_size, hidden, dropout):
        super(LabelClassifier, self).__init__()
        with self.init_scope():
            self.proj = L.Linear(2 * hidden, class_size)
        self.dropout = dropout

    def __call__(self, xs, doc_vec):
        # doc_vec:(1, batch_size, hidden_size)なのでdoc_vec[0]で(batch_size, hidden_size)にする
        doc_vec = doc_vec[0]

        # 各文ベクトルにdocumentベクトルをconcat
        # x:(batch_size, hidden_size), d:(,hidden_size)なので次元を合わせるためbroadcast_toでd:(batch_size, hidden_size)へ変換
        xs_proj = [self.proj(F.dropout(F.concat((x, F.broadcast_to(d, x.shape)), axis=1), self.dropout)) for x, d in zip(xs, doc_vec)]
        return xs_proj


class Multi(chainer.Chain):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout, coefficient):
        super(Multi, self).__init__()
        with self.init_scope():
            self.wordEnc = WordEncoder(src_vocab_size, embed_size, hidden_size, dropout)
            self.sentEnc = SentEncoder(hidden_size, dropout)
            self.wordDec = WordDecoder(trg_vocab_size, embed_size, hidden_size, dropout)
            self.labelClassifier = LabelClassifier(class_size, hidden_size, dropout)
        self.lossfun = F.softmax_cross_entropy
        self.coefficient = coefficient

    def __call__(self, sources, targets_sos, targets_eos, label_gold):
        coe = self.coefficient
        hs, cs, enc_ys = self.encode(sources)
        label_proj = self.labelClassifier(enc_ys, hs)

        concat_label_proj = F.concat(label_proj, axis=0)
        concat_label_gold = F.concat(label_gold, axis=0)
        loss_label = self.lossfun(concat_label_proj, concat_label_gold) / len(sources)

        return loss_label

    def encode(self, sources):
        sentences = []
        split_num = []
        sent_vectors = []

        for source in sources:
            split_num.append(len(source))
            sentences.extend(source)

        word_hy, _, _ = self.wordEnc(None, None, sentences)

        start = 0
        for num in split_num:
            sent_vectors.append(word_hy[start:start + num])
            start += num

        sent_hy, sent_cy, sent_ys = self.sentEnc(None, None, sent_vectors)

        return sent_hy, sent_cy, sent_vectors

    def predict(self, sources, sos, eos, limit=50):
        hs, cs, enc_ys = self.encode(sources)
        label_proj = self.labelClassifier(enc_ys, hs)

        result = []
        ys = [sos for _ in range(len(sources))]
        for i in range(limit):
            hs, cs, ys = self.wordDec(hs, cs, ys, enc_ys)
            ys = [self.xp.argmax(y.data, axis=1).astype(self.xp.int32) for y in ys]
            result.append(ys)
        result = self.xp.concatenate([self.xp.expand_dims(self.xp.array(x, dtype=self.xp.int32), 0) for x in result]).T
        result = self.xp.reshape(result, (len(sources), -1))

        output = []
        label = []
        for l in label_proj:
            l = F.softmax(l)
            r = l.data[:, 1]
            label.append(r)
        for r in result:
            index = np.argwhere(r == eos)
            if len(index) > 0:
                r = r[:index[0][0]]
            output.append(r)
        return output, label
