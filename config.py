import argparse


def config_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epoch",
                        type=int,
                        default=500,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="weight decay (L2 loss on parameters)",
    )
    parser.add_argument("--layer",
                        type=int,
                        default=2,
                        help="number of layers")
    parser.add_argument("--hidden",
                        type=int,
                        default=256,
                        help="hidden dimensions.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.3,
                        help="Dropout rate (1 - keep probability)")
    parser.add_argument(
        "--separate_loss",
        "-sr",
        type=bool,
        default=False,
        help="encourage the separation of the theta values",
    )
    parser.add_argument(
        "--supervised_loss",
        "-sv",
        type=bool,
        default=False,
        help="add inner layer supervised loss",
    )
    parser.add_argument("--patience", type=int, default=100, help="Patience")
    parser.add_argument("--device", type=int, default=4, help="device id")
    parser.add_argument("--dataset", default="chameleon", help="dateset")
    parser.add_argument(
        "--activation",
        type=str,
        default="all",
        choices=["all", "img", "real"],
        help="choose the activation function",
    )
    parser.add_argument("--sr_weight",
                        type=float,
                        default=0.5,
                        help="weight for the separation loss")
    parser.add_argument("--run", type=int, default=1, help="number of runs")
    parser.add_argument("--clip", type=float, default=10, help="clip")

    parser.add_argument(
        "--adj_norm",
        type=str,
        default="sys",
        choices=["sys", "row", "in_out_sys", "none"],
        help="adjacency matrix normalization method",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="default",
        choices=["random", "balance", "default", "balance_48"],
        help="split mode",
    )
    parser.add_argument(
        "--reciprocal",
        "--reci",
        type=bool,
        default=False,
        help="Use 1/C rather than C*(conj)",
    )
    parser.add_argument("--extra_dim",
                        type=int,
                        default=0,
                        help="extra dimension for the pesudo label")
    parser.add_argument("--select",
                        type=str,
                        default="soft",
                        help="select mode")
    parser.add_argument("--use_auc",
                        type=bool,
                        default=False,
                        help="use auc for the evaluation")

    parser.add_argument("--search_hyper_mode",
                        type=bool,
                        default=False,
                        help="search hyperparameters")
    parser.add_argument("--sweep_project_name",
                        type=str,
                        default=0,
                        help="sweep project name")

    parser.add_argument(
        "--cal_time",
        type=bool,
        default=True,
        help=
        "calculate the running time: training time and inference time per epoch"
    )

    args = parser.parse_args()
    if args.dataset in ["minesweeper", "tolokers", "quetions"]:
        args.use_auc = True
    else:
        args.use_auc = False
    return args
