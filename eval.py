import numpy as np

def accumulate_score(self, logit, onehot, observed):
    y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
                            for i, oh in enumerate(onehot)])
    
    y_pred, y_true = [],[]
    th = [[0.62, 0.38],\
            [0.55, 0.45],\
            [0.14, 0.36, 0.37, 0.17],\
            [0.22, 0.41, 0.22, 0.19],\
            [0.12, 0.13, 0.39, 0.26, 0.16, 0.12]]
    for b_idx, ob in enumerate(observed):
        pred, true = [],[]
        start = 0
        for a_idx, al in enumerate(self.attr_len):
            end = start + al
            if not sum(ob[start:end]):
                #p = np.argmax(logit[b_idx][start:end] - th[a_idx]) + start
                p = np.argmax(logit[b_idx][start:end], 0) + start
                t = sum(y_numbering[b_idx][start:end])
                if p == t:
                    self.attr_em[a_idx] += 1
                self.attr_cnt[a_idx] += 1
                pred.append(p)
                true.append(t)
            start += al
        if pred and true:
            y_pred.append(pred)
            y_true.append(true)

    self.num_users += len(y_true)
    
    for yp in y_pred:
        for p in yp:
            self.yp_counter[p] += 1
    for yp in y_true:
        for p in yp:
            self.yt_counter[p] += 1

    for y in zip(y_pred, y_true):
        self.y_counter[str(y[1])] += 1
        if np.array_equal(y[0], y[1]):
            self.y_em_counter[str(y[1])] += 1
            # count exact matchings for evaluating wP, wR, wF1
            self.em += 1
        # calculate and accumulate hamming loss
        self.hm_acc += hamming_loss(y[1], y[0])

def get_score(self):
    hm_loss = self.hm_acc / self.num_users
    wP = 0
    for y, cnt in self.y_counter.items():
        wP += self.y_em_counter[y] / cnt
    ## for debugging
    #if self.step == self.num_steps:
    #    for i in range(0, 18):
    #        print('{} : y-pred / y-true : {}, {}'
    #                .format(i, self.yp_counter[i], self.yt_counter[i]))
    #    print(len(self.y_em_counter), len(self.y_counter), wP / len(self.y_em_counter))
    ##
    wP /= len(self.y_counter)
    wR = self.em / self.num_users
    if wP == 0 and wR == 0:
        wP = wR = wF1 = 0
    else:
        wF1 = (2 * wP * wR) / (wP + wR)
    return hm_loss, wP, wR, wF1
