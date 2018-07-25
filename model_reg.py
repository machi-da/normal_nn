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
            self.W_c = L.Linear(2 * hidden, hidden)
            self.proj = L.Linear(hidden, n_vocab)
        self.dropout = dropout

    def __call__(self, hx, cx, xs, enc_hs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)

        ys_pad = F.pad_sequence(ys, length=None, padding=0.0)
        enc_hs = F.pad_sequence(enc_hs, length=None, padding=0.0)

        mask = self.xp.all(enc_hs.data == 0, axis=2, keepdims=True)
        mask_num = self.xp.full(mask.shape, -1024.0, dtype=self.xp.float32)

        alignment = []
        decode = []

        ys_pad = F.transpose(ys_pad, (1, 0, 2))
        for y in ys_pad:
            y = F.reshape(y, (*y.shape, 1))
            score = F.matmul(enc_hs, y)
            score = F.where(mask, mask_num, score)
            align = F.softmax(score, axis=1)
            context_vector = F.matmul(enc_hs, align, True, False)
            t = self.W_c(F.dropout(F.concat((y, context_vector), axis=1), self.dropout))
            ys_proj = self.proj(F.dropout(t, self.dropout))
            alignment.append(F.reshape(align, (len(xs), -1)))
            decode.append(ys_proj)

        decode = F.stack(decode, axis=1)
        alignment = F.stack(alignment, axis=1)
        return hy, cy, decode, alignment.data


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
        xs_proj = [self.proj(F.dropout(F.concat((x, F.broadcast_to(d, x.shape)), axis=1), self.dropout)) for x, d in
                   zip(xs, doc_vec)]
        return xs_proj


class MultiReg(chainer.Chain):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout, coefficient):
        super(MultiReg, self).__init__()
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

        concat_label_proj = F.concat(F.concat(label_proj, axis=0), axis=0)
        concat_label_gold = F.concat(label_gold, axis=0)
        loss_label = F.mean_squared_error(concat_label_proj, concat_label_gold)

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

    def predict(self, sources, sos, eos, limit=80):
        hs, cs, enc_ys = self.encode(sources)
        label_proj = self.labelClassifier(enc_ys, hs)

        alignments = []
        sentences = []
        # hs = F.transpose(hs, (1, 0, 2))
        # cs = F.transpose(cs, (1, 0, 2))
        # for h, c, y in zip(hs, cs, enc_ys):
        #     h = F.reshape(h, (1, *h.shape))
        #     c = F.reshape(c, (1, *c.shape))
        #     pre_word = sos
        #     sentence = [sos]
        #     attn_score = []
        #     for i in range(1, limit + 1):
        #         h, c, word_ys, alignment = self.wordDec(h, c, [pre_word], [y])
        #         word = self.xp.argmax(word_ys[0].data, axis=1)
        #         word = word.astype(self.xp.int32)
        #         pre_word = word
        #         if word == eos:
        #             break
        #         attn_score.append(alignment[0][0])
        #         sentence.append(word)
        #     else:
        #         attn_score = self.xp.sum(attn_score, axis=0) / i
        #     sentences.append(self.xp.hstack(sentence[1:]))
        #     alignments.append(attn_score)

        label = []
        for l in label_proj:
            l = F.softmax(l.T).data[0]
            label.append(l)

        return sentences, label