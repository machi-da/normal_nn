import copy


class Evaluate:
    def __init__(self, correct_txt_file):
        with open(correct_txt_file, 'r')as f:
            self.correct_data = f.readlines()

    def rank(self, attn_list):
        attn_data = copy.deepcopy(attn_list)
        rank_list = []
        for attn, d in zip(attn_data, self.correct_data):
            label = [int(num) - 1 for num in d.split('\t')[0].split(',')]
            rank = []
            for _ in range(len(attn)):
                index = attn.argmax()
                if index in label:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                attn[index] = -1
            rank_list.append(rank)

        return rank_list

    def single(self, rank_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        for r in rank_list:
            sent_num = len(r)
            # 正解ラベルの数: correct_num
            count_num = 0
            for rr in r:
                if rr[1]:
                    count_num += 1
            # 正解した数: correct
            correct = 0
            for i in range(count_num):
                if r[i][1]:
                    correct += 1
            if count_num == 1:
                score_dic[sent_num][1] += 1
                if correct:
                    score_dic[sent_num][0] += 1

        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        rate = [str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]
        rate.append(str(round(t_correct / t, 3)))
        count = ['{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()]
        count.append('{}/{}'.format(t_correct, t))
        return rate, count

    def multiple(self, rank_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        for r in rank_list:
            sent_num = len(r)
            count_num = 0
            for rr in r:
                if rr[1]:
                    count_num += 1
            correct = 0
            for i in range(count_num):
                if r[i][1]:
                    correct += 1

            score_dic[sent_num][0] += correct
            score_dic[sent_num][1] += count_num

        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        rate = [str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]
        rate.append(str(round(t_correct / t, 3)))
        count = ['{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()]
        count.append('{}/{}'.format(t_correct, t))
        return rate, count