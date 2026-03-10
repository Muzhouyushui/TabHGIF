# # config_args.py
# import argparse
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_nodes_per_hyperedge", type=int, default=700,
#                         help="Maximum nodes allowed in a hyperedge before clustering")
#     # Other arguments...
#     return parser.parse_args()

# config_args.py
import argparse
from paths import ACI_TEST, ACI_TRAIN

def get_args():
    parser = argparse.ArgumentParser(description="Hypergraph experiments")

    # ---------- General ----------
    # parser.add_argument("--model_name", type=str, default="HyperGCN",
    #                     choices=["HyperGCN", "HGNN"],
    #                     help="Which model to train")
    parser.add_argument("--depth", type=int, default=2,
                        help="Number of layers / propagation depth")
    parser.add_argument("--hidden_dim", type=int, default=90)
    parser.add_argument("--dropout", type=float, default=0.01)

    parser.add_argument("--d", type=int, default=None,
                        help="Input feature dimension; should be assigned after loading the data in the script")
    parser.add_argument("--n_class", "--c", dest="n_class",
                        type=int, default=2,
                        help="Number of classes for the classification task (can use --n_class or --c)")
    parser.add_argument(
               "--dataset", type=str, default="adult",
               help="Dataset name, used for HyperGCN hidden-dimension generation (e.g., if 'citeseer', then power += 2)"
    )

    parser.add_argument(
        '--log_every', type=int, default=10,
        help='Print training status every N epochs (default: 10)'
    )

    # ---------- HyperGCN-specific ----------
    parser.add_argument("--fast", action="store_true",
                        help="Fast-HyperGCN (only approximate once)")
    parser.add_argument("--mediators", action="store_true",
                        help="Use mediator connections in Laplacian")

    # ---------- Data / Training ----------
    parser.add_argument("--max_nodes_per_hyperedge", type=int, default=100)
    parser.add_argument("--max_nodes_per_hyperedge_test", type=int, default=100)
    parser.add_argument("--max_nodes_per_hyperedge_train", type=int, default=100)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", default=True)

    # Data files
    # parser.add_argument("--train_csv", type=str,
    #                     help="Path to training CSV")
    # parser.add_argument("--test_csv", type=str,
    #                     help="Path to test CSV")
    parser.add_argument("--train_csv", type=str,
                        default=ACI_TRAIN,
                        help="Path to training CSV")
    parser.add_argument("--test_csv", type=str,
                        default=ACI_TEST,
                        help="Path to test CSV")

    # Data files for autodl
    # parser.add_argument("--train_csv", type=str,
    #                     help="Path to training CSV")
    # parser.add_argument("--test_csv", type=str,
    #                     help="Path to test CSV")

    parser.add_argument(
        "--cat_cols",
        nargs="+",
        default=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ],
        help="List of categorical column names used to construct hyperedges"
    )

    # Milestones for the learning-rate scheduler
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[100, 150],
        help="List of milestones for MultiStepLR (epoch indices)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Decay factor for MultiStepLR"
    )
    parser.add_argument(
        "--continuous_cols",
        nargs="+",
        default=[
            "age","fnlwgt","education-num",
            "capital-gain","capital-loss","hours-per-week"
        ],
        help="List of column names treated as continuous values in delete_feature_column"
    )

    # GIF/IF hyperparameters
    parser.add_argument("--if_iters", type=int, default=80,
                        help="Number of GIF iterations")
    parser.add_argument("--if_damp", type=float, default=0.01,
                        help="Damping coefficient for GIF")
    parser.add_argument("--if_scale", type=float, default=10,
                        help="Scaling coefficient for GIF")

    parser.add_argument(
        "--columns_to_unlearn",
        nargs="+",
        default=["age","education-num"],
        help="List of column names for feature-level unlearning"
    )
    parser.add_argument(
        '--neighbor_k',
        type=int,
        default=12,
        help='Threshold K: a node is considered a neighbor if it shares at least K hyperedges with any deleted node'
    )
    parser.add_argument(
        "--del_cols",
        nargs="+",
        # default=["age","education-num","capital-loss","hours-per-week"],
        default=["age",],

        help="List of column names for feature-level unlearning"
    )

    # Additional hyperparameters for the Full model
    parser.add_argument('--full-epochs', type=int, default=100)
    parser.add_argument('--full-lr', type=float, default=5e-3)
    parser.add_argument('--full-hidden', type=int, default=64)
    parser.add_argument('--full-dropout', type=float, default=0.5)

    # MIA-specific hyperparameters (can also directly reuse shadow settings)
    parser.add_argument('--shadow-epochs', type=int, default=100)
    parser.add_argument('--shadow-lr', type=float, default=5e-3)
    parser.add_argument('--shadow-hidden', type=int, default=64)
    parser.add_argument('--shadow-dropout', type=float, default=0.5)
    parser.add_argument('--shadow-test-ratio', type=float, default=0.3)
    parser.add_argument('--attack-epochs', type=int, default=50)
    parser.add_argument('--attack-lr', type=float, default=1e-2)
    parser.add_argument('--attack-test-ratio', type=float, default=0.3)
    return parser.parse_args()