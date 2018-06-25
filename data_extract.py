import psycopg2
import sys
import MeCab
from multiprocessing import Pool
import preprocessing


def data_format(id, category, text):
    m = MeCab.Tagger('-Owakati')
    m.parse('')
    text = preprocessing.remove_repetition(text)
    sentences = preprocessing.sentence_split(text)

    split_sentences = []
    for sentence in sentences:
        s = m.parse(sentence).strip()
        split_sentences.append(s)
    data = '{}\t{}\t{}\t{}'.format(id, category, len(split_sentences), '|'.join(split_sentences))
    return data


def wrapper_data_format(l):
    data = data_format(l[0], l[1], l[2])
    return data


def write(res_lit):
    with open('category/que.tsv', 'a')as f:
        [f.write(r + '\n') for r in res_lit]


def main():
    args = sys.argv

    # db情報
    host = 'localhost'
    port = 5432
    database = 'chiebukuro'
    user = 'machida'
    passwd = 'matimati'

    connector = psycopg2.connect(host=host, port=port, database=database, user=user, password=passwd)
    connector.autocommit = True
    cursor = connector.cursor()

    start = 0
    limit = int(args[1])

    p = Pool()

    count = 0
    while True:
        sql = 'SELECT id, category, text FROM question WHERE id BETWEEN {} AND {}'.format(start, start+limit)

        cursor.execute(sql)
        res = cursor.fetchall()

        # res_lit = []
        # for r in res:
        #     res_lit.append(data_format(r[0], r[1], r[2]))

        res_lit = p.map(wrapper_data_format, res)
        write(res_lit)

        count += len(res)
        print('Finish :{}'.format(count))

        if len(res) == 0:
            print('Process all data')
            break

        start += limit + 1

main()