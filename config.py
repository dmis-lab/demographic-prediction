
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
    parser.add_argument('--num-workers', type=int, default=2)

    # task settings
    parser.add_argument('--task-type', type=str, default='new_user',
                        help="[partial50, new_user]")

    # optimizations
    parser.add_argument('--opt', type=str, default='Adam',
                        help="Adam / RMSprop / SGD / Adagrad / Adadelta / Adamax")
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)

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
    parser.add_argument('--model-type', type=str, default='ETNA',
                        help="[POP, JNE, SNE, ETN, ETNA]")
    parser.add_argument('--loss-type', type=str, default='likelihood',
                        help="[classification, likelihood]")

    # debugging and analysis
    parser.add_argument('--do-validation', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help="Whether not to use CUDA when available")
    parser.add_argument('--save-log', type=int, default=0)
    parser.add_argument('--print-per-step', type=int, default=300)
    parser.add_argument('--vis-per-step', type=int, default=100)
    parser.add_argument('--print-attr-score', type=int, default=0)

    # regularization
    parser.add_argument('--no-early-stop', action='store_false', default=True,
                        help="you can combinate two words which are from in each set: [mic, mac, w] and [P, R, F1]")
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=0.0)

    args = parser.parse_args()
    return args

