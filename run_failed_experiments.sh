#!/bin/bash
# 重新运行之前失败的实验（修复路径、__init__.py、argparse 后重跑）

PROJ="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJ:$PROJ/HGNNs_Model"
LOG_DIR=$PROJ/results
mkdir -p $LOG_DIR

run_exp() {
    local script=$1
    local logfile=$2
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] 开始: $script"
    echo "========================================"
    python $script > "$LOG_DIR/$logfile" 2>&1
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] 完成: $script ✓"
    else
        echo "[$(date '+%H:%M:%S')] 失败: $script ✗  (查看 $LOG_DIR/$logfile)"
    fi
}

cd $PROJ

echo "########################################"
echo "# 重跑失败实验 - ACI"
echo "########################################"

# ACI - HGNN rowgroup（adult.data 路径修复后）
run_exp ACI_run/HGNN/HGNN_Unlearning_Value.py        aci_hgnn_rowgroup.log

# ACI - HGNNP（__init__.py 修复后）
run_exp ACI_run/HGNNP/HGNNP_Unlearning_row_nei.py    aci_hgnnp_row.log
run_exp ACI_run/HGNNP/HGNNP_Unlearning_col.py        aci_hgnnp_col.log
run_exp ACI_run/HGNNP/HGNNP_Unlearning_row_Value.py  aci_hgnnp_rowgroup.log

# ACI - HGAT（__init__.py 修复后）
run_exp ACI_run/HGAT/HGAT_Unlearning_row_nei.py      aci_hgat_row.log
run_exp ACI_run/HGAT/HGAT_Unlearning_col.py          aci_hgat_col.log
run_exp ACI_run/HGAT/HGAT_Unlearning_row_Value.py    aci_hgat_rowgroup.log

echo "########################################"
echo "# 重跑失败实验 - Bank"
echo "########################################"

# Bank - HGCN col（argparse 冲突修复后）
run_exp bank/HGCN/HGCN_Unlearning_col_bank.py        bank_hgcn_col.log
# Bank - HGCN rowgroup（bank-full.csv 路径修复后）
run_exp bank/HGCN/HGCN_Unlearning_row_bank_Value.py  bank_hgcn_rowgroup.log

# Bank - HGNN（bank-full.csv 路径修复后）
run_exp bank/HGNN/HGNN_Unlearning_col_bank.py        bank_hgnn_col.log
run_exp bank/HGNN/HGNN_Unlearning_Value.py           bank_hgnn_rowgroup.log

# Bank - HGAT（__init__.py 修复后）
run_exp bank/HGAT/HGAT_Unlearning_row_nei.py         bank_hgat_row.log
run_exp bank/HGAT/HGAT_Unlearning_col.py             bank_hgat_col.log
run_exp bank/HGAT/HGAT_Unlearning_row_Value.py       bank_hgat_rowgroup.log

echo "########################################"
echo "# 重跑失败实验 - Credit"
echo "########################################"

# Credit - HGCN（crx.data 路径修复后）
run_exp Credit/HGCN/HGCN_Unlearning_row_Credit.py    credit_hgcn_row.log
run_exp Credit/HGCN/HGCN_Unlearning_col_Credit.py    credit_hgcn_col.log
run_exp Credit/HGCN/HGCN_Unlearning_Value.py         credit_hgcn_rowgroup.log

# Credit - HGNN（crx.data 路径修复后）
run_exp Credit/HGNN/HGNN_Unlearning_row_Credit.py    credit_hgnn_row.log
run_exp Credit/HGNN/HGNN_Unlearning_col_Credit.py    credit_hgnn_col.log
run_exp Credit/HGNN/HGNN_Unlearning_value_Credit.py  credit_hgnn_rowgroup.log

# Credit - HGNNP col（crx.data 路径修复后）
run_exp Credit/HGNNP/HGNNP_Unlearning_col_Credit.py  credit_hgnnp_col.log

# Credit - HGAT（路径修复后）
run_exp Credit/HGAT/HGAT_Unlearning_row_Credit.py    credit_hgat_row.log
run_exp Credit/HGAT/HGAT_Unlearning_col_credit.py    credit_hgat_col.log
run_exp Credit/HGAT/HGAT_Unlearning_Value.py         credit_hgat_rowgroup.log

echo "########################################"
echo "# 重跑失败实验 - RELOAD 基线"
echo "########################################"

# RELOAD（路径修复后）
run_exp baseline_Tabnet/Baseline_RELOAD_ACI_Value.py  reload_aci_rowgroup.log
run_exp baseline_Tabnet/Baseline_RELOAD_Bank.py       reload_bank_row.log
run_exp baseline_Tabnet/Baseline_RELOAD_Bank_Value.py reload_bank_rowgroup.log
run_exp baseline_Tabnet/Baseline_RELOAD_Credit_value.py reload_credit_rowgroup.log

echo "########################################"
echo "# 重跑失败实验 - Fine-Tuning 基线"
echo "########################################"

# FT ACI HGNNP（__init__.py 修复后）
run_exp Baseline_FT_ACI/HGNNP_baseline_FT/run_ft_hgnnp_row_zero.py ft_aci_hgnnp_row.log
run_exp Baseline_FT_ACI/HGNNP_baseline_FT/run_ft_hgnnp_col_zero.py ft_aci_hgnnp_col.log

# FT ACI HGAT（__init__.py 修复后）
run_exp Baseline_FT_ACI/HGAT_baseline_FT/run_ft_hgat_row_zero.py ft_aci_hgat_row.log
run_exp Baseline_FT_ACI/HGAT_baseline_FT/run_ft_hgat_col_zero.py ft_aci_hgat_col.log

# FT Bank HGAT row（__init__.py 修复后）
run_exp bank/FT_bank/FT_HGAT/row.py                  ft_bank_hgat_row.log

# FT Credit HGNNP row（import 修复后）
run_exp Credit/FT_baseline_Credit/HGNNP/row.py        ft_credit_hgnnp_row.log

echo "########################################"
echo "# 重跑完成！"
echo "########################################"
echo "日志保存在: $LOG_DIR"
