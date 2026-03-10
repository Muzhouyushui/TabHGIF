import argparse
from paths import ACI_TEST, ACI_TRAIN

def get_args():
    parser = argparse.ArgumentParser()

    # Task control
    parser.add_argument("--cuda", action="store_true", default=True)

    parser.add_argument("--remove_ratio",   type=float, default=0.3,
                        help="Ratio of training nodes to unlearn")
    parser.add_argument("--dataset",        type=str,   default="adult",
                        help="Dataset name, used only for logging")

    # Data files
    # parser.add_argument("--train_csv",      type=str,
    #                     help="Path to training CSV")
    # parser.add_argument("--test_csv",       type=str,
    #                     help="Path to test CSV")
    #
    parser.add_argument("--train_csv", type=str,
                        default=ACI_TRAIN,
                        help="Path to training CSV")
    parser.add_argument("--test_csv", type=str,
                        default=ACI_TEST,
                        help="Path to test CSV")

    # # Data files for autodl
    # parser.add_argument("--train_csv", type=str,
    #                     help="Path to training CSV")
    # parser.add_argument("--test_csv", type=str,
    #                     help="Path to test CSV")

    parser.add_argument("--hidden_dim", type=int, default=90)

    # Hypergraph construction
    parser.add_argument("--max_nodes_per_hyperedge", type=int, default=50,
                        help="Maximum number of nodes allowed in a hyperedge during construction")

    parser.add_argument("--mediators",       action="store_true",
                        help="Whether HGCN uses the mediators strategy")

    # Model structure
    parser.add_argument("--depth",          type=int,   default=3,
                        help="Number of HyperGCN layers / depth")
    parser.add_argument("--dropout",        type=float, default=0.00 ,
                        help="Dropout probability in the model")
    parser.add_argument("--fast",           action="store_true",
                        help="Enable HyperGCN fast mode")

    # Training hyperparameters
    parser.add_argument("--epochs",         type=int,   default=100,
                        help="Total number of training epochs")
    parser.add_argument("--log_every",      type=int,   default=10,
                        help="Print logs every N epochs during training")
    parser.add_argument("--lr",             type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight_decay",   type=float, default=5e-4,
                        help="Weight decay coefficient for Adam")
    parser.add_argument("--milestones",     nargs="+",   type=int, default=[100,150],
                        help="List of milestones for the learning-rate scheduler")
    parser.add_argument("--gamma",          type=float, default=0.1,
                        help="Gamma for the learning-rate scheduler")

    # GIF/IF hyperparameters
    parser.add_argument("--if_iters",       type=int,   default=80,
                        help="Number of GIF iterations")
    parser.add_argument("--if_damp",        type=float, default=0.01,
                        help="Damping coefficient for GIF")
    parser.add_argument("--if_scale",       type=float, default=1e6,
                        help="Scaling coefficient for GIF")
    parser.add_argument("--neighbor_K",     type=int,   default=5,
                        help="Neighbor discovery threshold: a node is considered a neighbor if it shares at least K hyperedges with any deleted node")

    # Continuous columns
    parser.add_argument(
        "--categate_cols",
        nargs="+",
        default=   [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "sex", "native-country"
        ],
        help="Columns treated as continuous values in delete_feature_column_hgcn"
    )

    # Columns for feature-level unlearning
    parser.add_argument(
        "--columns_to_unlearn",
        nargs="+",
        default=["age"],
        help="List of column names for column-level deletion"
    )
    # [
    #     "age", "workclass", "fnlwgt", "education", "education-num",
    #     "marital-status", "occupation", "relationship", "race",
    #     "sex", "capital-gain", "capital-loss", "hours-per-week",
    #     "native-country", "income"
    # ]
    return parser.parse_args()