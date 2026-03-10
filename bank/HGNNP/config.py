# # config_args.py
# import argparse
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_nodes_per_hyperedge", type=int, default=700,
#                         help="Maximum nodes allowed in a hyperedge before clustering")
#     # 其余参数……
#     return parser.parse_args()
# config_args.py
import argparse
from paths import BANK_DATA

def get_args():
    parser = argparse.ArgumentParser(
        description="HGNNP 行级 & 列级 Unlearning for Bank Marketing 数据集"
    )

    # 数据路径
    # parser.add_argument(
    #     "--train-csv", type=str,
    #     help="训练集 CSV 文件路径（Bank Marketing）"
    # )
    parser.add_argument(
        "--train-csv", type=str,
        default=BANK_DATA,
        help="训练集 CSV 文件路径（Bank Marketing）"
    )
    parser.add_argument(
        "--test-csv", type=str, default=None,
        help="测试集 CSV 文件路径；若不提供则使用 train-csv 并按 split-ratio 拆分"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.2,
        help="当不提供 --test-csv 时，从 train-csv 中拆出的测试集比例"
    )

    # 超图构建
    parser.add_argument(
        "--max-nodes-per-hyperedge", type=int, default=50,
        help="训练/测试集超边最大节点数（Bank）"
    )

    # 模型结构 & 训练超参
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="HGNNP 隐藏层维度"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="HGNNP dropout 比例"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="学习率"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4,
        help="权重衰减"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="训练轮数"
    )
    parser.add_argument(
        "--milestones", nargs='+', type=int, default=[100, 150],
        help="MultiStepLR 的里程碑 epochs 列表"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1,
        help="学习率衰减系数"
    )

    # GIF Unlearning 超参
    parser.add_argument(
        "--gif-iters", type=int, default=20,
        help="GIF 迭代次数"
    )
    parser.add_argument(
        "--gif-damp", type=float, default=0.01,
        help="GIF 阻尼系数"
    )
    parser.add_argument(
        "--gif-scale", type=float, default=1e7,
        help="GIF 缩放比例"
    )

    # 行级 Unlearning
    parser.add_argument(
        "--remove-ratio", type=float, default=0.3,
        help="行级 Unlearning 时随机删除节点的比例"
    )
    parser.add_argument(
        "--neighbor-k", type=int, default=12,
        help="行级 Unlearning GIF 时，选择至少 K 条超边为邻居"
    )

    # 列级 Unlearning & 日志
    parser.add_argument(
        "--log-every", type=int, default=10,
        help="训练过程中每隔多少 epoch 打印一次日志"
    )
    parser.add_argument(
        "--columns-to-unlearn", nargs='+', type=str, default=["age"],
        help="列级 Unlearning 时要删除的列名列表（默认只删除 age）"
    )
    # MIA 攻击超参（可选）
    parser.add_argument(
        "--shadow-epochs", type=int, default=100,
        help="Shadow 模型训练轮数"
    )
    parser.add_argument(
        "--shadow-lr", type=float, default=0.005,
        help="Shadow 模型学习率"
    )
    parser.add_argument(
        "--attack-epochs", type=int, default=50,
        help="Membership inference 攻击模型训练轮数"
    )
    parser.add_argument(
        "--attack-lr", type=float, default=0.01,
        help="Membership inference 攻击模型学习率"
    )
    parser.add_argument(
        "--attack-test-ratio", type=float, default=0.3,
        help="Membership inference 攻击模型测试集比例"
    )

    args = parser.parse_args()
    return args