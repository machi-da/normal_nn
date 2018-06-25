import re

patterns = [
        [r'。+', '。'],
        [r'\.+', '.'],
        [r'・+', '・'],
        [r'\?+', '？'],
        [r'\!+', '!'],
        [r'-+', '-']
    ]
for p in patterns:
    p[0] = re.compile(p[0])


# 記号の繰り返しを除去
def remove_repetition(sentence):
    for p in patterns:
        sentence = re.sub(p[0], p[1], sentence)

    return sentence


open_symbol  = ['「', '『', '（', '(']
close_symbol = ['」', '』', '）', ')']
split_symbol = re.compile(r'。+|？+|！+|(!\?)+|\?+|!+')
maru_kakko = re.compile(r'\(.*?\)')
alphabet = re.compile(r'[a-zA-Z]')


def pair_bracket(o, c):
    o_symbol = open_symbol.index(o[1])
    c_symbol = close_symbol.index(c[1])
    if o_symbol == c_symbol:
        return range(o[0], c[0]+1)
    return []


def bracket_range(text):
    check = []
    except_range = []
    for i, w in enumerate(text):
        if w in open_symbol:
            check.append((i, w))
        elif w in close_symbol:
            if len(check) > 0:
                o = check.pop()
                c = (i, w)
                except_range.extend(pair_bracket(o, c))
    except_range = set(except_range)

    return except_range


# テキストから文分割したリストを返す
def sentence_split(text):
    except_range = bracket_range(text)

    split_ite = re.finditer(split_symbol, text)
    sentences = []

    # splitシンボルがなければそのまま返す
    if len(re.findall(split_symbol, text)) == 0:
        sentences.append(text)
        return sentences

    start = 0
    for ite in split_ite:
        if start > ite.end():
            continue
        # '!'の前がアルファベットの場合は分割しない
        if '!' in ite.group():
            if re.match(alphabet, text[ite.start() - 1]):
                continue
        # 括弧内のsplitシンボルは分割しない
        if ite.end()-1 in except_range:
            continue

        sentences.append(text[start:ite.end()])
        start = ite.end()

        # 丸括弧の場合、前の文に接続する
        if len(text) > start:
            if text[start] == '(':
                m = re.match(r'\(.*?\)', text[start:])
                if m:
                    sentences[-1] += text[start:start + m.end()]
                    start += m.end()
            elif text[start] == '（':
                m = re.match(r'（.*?）', text[start:])
                if m:
                    sentences[-1] += text[start:start + m.end()]
                    start += m.end()

    # start位置がtext位置と同じになったら終了
    if start == len(text):
        return sentences

    # まだ文字が残っていたら残りを追加する
    sentences.append(text[start:])

    return sentences

# a = '新しいタマゴッチがでるときいたんですが前と何がちがうんですか'
# print(sentence_split(a))