# ft_config.py
import argparse
from paths import ACI_TEST, ACI_TRAIN

def get_args():
    parser = argparse.ArgumentParser("FT baselines for HGCN row deletion (feature-zero) + MIA")

    # ===== Data =====
    parser.add_argument("--train_csv", type=str,
                        default=ACI_TRAIN,
                        help="训练数据路径（adult.data）")
    parser.add_argument("--test_csv", type=str,
                        default=ACI_TEST,
                        help="测试数据路径（adult.test）")

    # Adult 字段（和你原 HGCN 管线一致即可）
    parser.add_argument("--label_col", type=str, default="income", help="标签列名")
    parser.add_argument("--categate_cols", type=str, nargs="+",
                        default=[
                            "workclass", "education", "marital-status", "occupation",
                            "relationship", "race", "sex", "native-country"
                        ],
                        help="离散列名（用于 one-hot + 超边）")

    # 如果你原始实验过滤 '?'，这里打开
    parser.add_argument("--filter_missing_q", action="store_true",
                        help="过滤包含 '?' 的样本行（Adult缺失值）")

    # ===== Hyperedge config =====
    parser.add_argument("--max_nodes_per_hyperedge", type=int, default=10000,
                        help="每条超边最多节点数，过大则随机截断")
    parser.add_argument("--mediators", action="store_true",
                        help="laplacian() 的 mediators 选项（和你 HyperGCN 实现一致）")

    # ===== Model (HGCN) =====
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cuda", action="store_true", default=True)

    # ===== Train (Full) =====
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--gamma", type=float, default=0.1)

    # ===== Deletion =====
    parser.add_argument("--remove_ratio", type=float, default=0.30,
                        help="从训练集中随机删除的比例")
    parser.add_argument("--seed", type=int, default=1)

    # ===== Finetune baselines =====
    parser.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200],
                        help="FT-K/FT-head 的 K steps 列表")
    parser.add_argument("--ft_lr", type=float, default=1e-3)
    parser.add_argument("--ft_wd", type=float, default=0.0)

    # ===== MIA (use your MIA_HGCN.py) =====
    parser.add_argument("--run_mia", action="store_true",default=True,
                        help="是否对每个模型跑 MIA")
    parser.add_argument("--shadow_test_ratio", type=float, default=0.3)
    parser.add_argument("--num_shadows", type=int, default=5)
    parser.add_argument("--num_attack_samples", type=int, default=None)
    parser.add_argument("--shadow_lr", type=float, default=5e-3)
    parser.add_argument("--shadow_epochs", type=int, default=100)
    parser.add_argument("--attack_test_split", type=float, default=0.3)
    parser.add_argument("--attack_lr", type=float, default=1e-2)
    parser.add_argument("--attack_epochs", type=int, default=50)

    # ===== Multi-run + save =====
    parser.add_argument("--runs", type=int, default=1, help="重复 runs 次（每次 seed+run_id）")
    parser.add_argument("--out_csv", type=str, default="ft_row_zero_results.csv")

    return parser.parse_args()
