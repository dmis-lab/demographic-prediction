
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='ocb',
                        help="[beiren, ocb, movieLens]")
    parser.add_argument('--data-path', type=str, default="./data/preprd/",
                        help="")
    parser.add_argument('--rand-seed', type=int, default=1)
    parser.add_argument('--data-shuffle', type=int, default=0)
    parser.add_argument('--data-sampling', type=int, default=0)
    parser.add_argument('--sample_type', type=str, default='full')

    # task settings
    parser.add_argument('--partial-training', type=int, default=1)
    parser.add_argument('--partial-eval', type=int, default=1)
    parser.add_argument('--task-type', type=str, default='new_user',
                        help="[partial50, new_user]")
    parser.add_argument('--tasks', type=int, nargs='+')

    # optimizations
    parser.add_argument('--opt', type=str, default='Adam',
                        help="Adam / RMSprop / SGD / Adagrad / Adadelta / Adamax")
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--use-negsample', type=int, default=0)

    # sharing & model structure
    parser.add_argument('--uniq-input', type=int, default=0)
    parser.add_argument('--share-emb', type=int, default=1)
    parser.add_argument('--share-attention', type=int, default=0)
    parser.add_argument('--attention-layer', type=int, default=1,
                        help="you can choose [1 or 2] when using TAN model")
    parser.add_argument('--learning-form', type=str, default='separated',
                        help="[separated, structured]")

    # embeddings
    parser.add_argument('--item-emb-size', type=int, default=100)

    # training parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--user-emb-dim', type=int, default=100)
    parser.add_argument('--num-negs', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=20000)
    parser.add_argument('--grad-max-norm', type=float, default=5)
    parser.add_argument('--num-batches', type=int, default=20)

    # model's parameters
    parser.add_argument('--model-type', type=str, default='TAN',
                        help="[POP, SVD_str, Average, RNN, TAN]")
    parser.add_argument('--loss-type', type=str, default='likelihood',
                        help="[classification, likelihood]")
    parser.add_argument('--rnn-type', type=str, default='LSTM')
    parser.add_argument('--rnn-size', type=int, default=70)
    parser.add_argument('--rnn-layer', type=int, default=2)
    parser.add_argument('--rnn-drop', type=float, default=0.2)

    # debugging and analysis
    parser.add_argument('--save-log', type=int, default=1)
    parser.add_argument('--save-output', type=int, default=0)
    parser.add_argument('--print-per-step', type=int, default=99999)
    parser.add_argument('--vis-per-step', type=int, default=100)
    parser.add_argument('--rand-search', type=int, default=0)
    parser.add_argument('--print-attr-score', type=int, default=0)

    # regularization
    parser.add_argument('--early-stop', type=str, default='va_wF1',
                        help="you can combinate two words which are from in each set: [mic, mac, w] and [P, R, F1]")
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=0.0)

    args = parser.parse_args()
    return args
