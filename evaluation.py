from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_fscore_support as f1_score

def get_score(pred, onehot, attr_len):
    print(onehot)
    print(attr_len)
    sys.exit()

    hm_loss = hamming_loss(label, pred)
    p, r, f1, _ = f1_score(label, pred, average='weighted')

    return hm_loss, p, r, f1

