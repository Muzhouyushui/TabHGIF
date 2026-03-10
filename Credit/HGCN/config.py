import argparse
from paths import CREDIT_DATA

def get_args():
    parser = argparse.ArgumentParser(description="HyperGCN Column-level Unlearning on the Bank dataset")

    # Device and task control
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA instead of CPU")
    # parser.add_argument("--remove_ratio", type=float, default=0.3,
    #                     help="Ratio for randomly deleting columns (as a proportion of all feature columns)")
    parser.add_argument("--dataset", type=str, default="bank",
                        help="Dataset name, used only for recording/logging")

    # Data files (Bank Marketing dataset; semicolon-separated)
    # parser.add_argument("--train_csv", type=str,
    #                     help="Path to the full Bank Marketing CSV file (; separated)")
    # Data files (Bank Marketing dataset; semicolon-separated)
    parser.add_argument("--train_csv", type=str,
                        default=CREDIT_DATA,
                        help="Path to the full Bank Marketing CSV file (; separated)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Optional path to an independent test CSV file (if not provided, split using split-ratio)")
    parser.add_argument("--split_ratio", type=float, default=0.2,
                        help="If test_csv is not provided, split the test set from train_csv using this ratio")
    # —— Hyperedge construction configuration —— #
    parser.add_argument(
        "--max_nodes_per_hyperedge",
        type=int,
        default=30,
        help="Maximum number of nodes allowed in a single hyperedge during hyperedge construction"
    )
    # Model structure parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="HyperGCN hidden layer dimension")
    parser.add_argument("--depth", type=int, default=3, help="HyperGCN network depth")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout probability")
    parser.add_argument("--fast", action="store_true", help="HyperGCN fast mode")
    parser.add_argument("--mediators", action="store_true", help="Whether to use the Mediators strategy")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--log_every", type=int, default=100, help="Logging interval (in epochs)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 150],
                        help="LR scheduler milestones")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler decay factor")

    # GIF/IF hyperparameters
    parser.add_argument("--gif_iters", type=int, default=80, help="Number of GIF iterations")
    parser.add_argument("--gif_damp", type=float, default=0.01, help="GIF damping coefficient")
    parser.add_argument("--gif_scale", type=float, default=1e7, help="GIF scaling factor")

    # Column categories for the Bank dataset
    # parser.add_argument("--data-csv", type=str,
    #                     help="Credit Approval data CSV file path")
    parser.add_argument("--data-csv", type=str,
                        default=CREDIT_DATA,
                        help="Credit Approval data CSV file path")
    parser.add_argument("--cat-cols", nargs="+", type=str,
                        default=['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'],
                        help="List of categorical feature column names")
    parser.add_argument("--cont-cols", nargs="+", type=str,
                        default=['A2', 'A3', 'A8', 'A11', 'A14', 'A15'],
                        help="List of continuous feature column names")

    # Columns to perform column-level Unlearning on
    parser.add_argument(
        "--columns_to_unlearn",
        nargs="+",
        default=["A5"],  # Can be changed to other columns as needed
        help="Which columns to zero out and rebuild hyperedges for (column-level Unlearning)"
    )

    return parser.parse_args()