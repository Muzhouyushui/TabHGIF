#!/bin/bash
# TabHGIF 全部实验运行脚本
# 运行方式: bash run_all_experiments.sh 2>&1 | tee results/run_all.log

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
echo "# Table III/IV/VII: 主体实验 - ACI"
echo "########################################"

# ACI - HGCN
run_exp ACI_run/HGCN/HGCN_Unlearning_row_nei.py     aci_hgcn_row.log
run_exp ACI_run/HGCN/HGCN_Unlearning_col.py          aci_hgcn_col.log
run_exp ACI_run/HGCN/HGCN_Unlearning_row_Value.py    aci_hgcn_rowgroup.log

# ACI - HGNN
run_exp ACI_run/HGNN/HGNN_Unlearning_row_nei.py      aci_hgnn_row.log
run_exp ACI_run/HGNN/HGNN_Unlearning_col.py          aci_hgnn_col.log
run_exp ACI_run/HGNN/HGNN_Unlearning_Value.py        aci_hgnn_rowgroup.log

# ACI - HGNNP
run_exp ACI_run/HGNNP/HGNNP_Unlearning_row_nei.py    aci_hgnnp_row.log
run_exp ACI_run/HGNNP/HGNNP_Unlearning_col.py        aci_hgnnp_col.log
run_exp ACI_run/HGNNP/HGNNP_Unlearning_row_Value.py  aci_hgnnp_rowgroup.log

# ACI - HGAT
run_exp ACI_run/HGAT/HGAT_Unlearning_row_nei.py      aci_hgat_row.log
run_exp ACI_run/HGAT/HGAT_Unlearning_col.py          aci_hgat_col.log
run_exp ACI_run/HGAT/HGAT_Unlearning_row_Value.py    aci_hgat_rowgroup.log

echo "########################################"
echo "# Table III/VII: 主体实验 - Bank"
echo "########################################"

# Bank - HGCN
run_exp bank/HGCN/HGCN_Unlearning_row_bank.py        bank_hgcn_row.log
run_exp bank/HGCN/HGCN_Unlearning_col_bank.py        bank_hgcn_col.log
run_exp bank/HGCN/HGCN_Unlearning_row_bank_Value.py  bank_hgcn_rowgroup.log

# Bank - HGNN
run_exp bank/HGNN/HGNN_Unlearning_row_bank.py        bank_hgnn_row.log
run_exp bank/HGNN/HGNN_Unlearning_col_bank.py        bank_hgnn_col.log
run_exp bank/HGNN/HGNN_Unlearning_Value.py           bank_hgnn_rowgroup.log

# Bank - HGNNP
run_exp bank/HGNNP/HGNNP_Unlearning_row_bank.py      bank_hgnnp_row.log
run_exp bank/HGNNP/HGNNP_Unlearning_col_bank.py      bank_hgnnp_col.log
run_exp bank/HGNNP/HGNNP_Unlearning_row_bank_Value.py bank_hgnnp_rowgroup.log

# Bank - HGAT
run_exp bank/HGAT/HGAT_Unlearning_row_nei.py         bank_hgat_row.log
run_exp bank/HGAT/HGAT_Unlearning_col.py             bank_hgat_col.log
run_exp bank/HGAT/HGAT_Unlearning_row_Value.py       bank_hgat_rowgroup.log

echo "########################################"
echo "# Table III/VII: 主体实验 - Credit"
echo "########################################"

# Credit - HGCN
run_exp Credit/HGCN/HGCN_Unlearning_row_Credit.py    credit_hgcn_row.log
run_exp Credit/HGCN/HGCN_Unlearning_col_Credit.py    credit_hgcn_col.log
run_exp Credit/HGCN/HGCN_Unlearning_Value.py         credit_hgcn_rowgroup.log

# Credit - HGNN
run_exp Credit/HGNN/HGNN_Unlearning_row_Credit.py    credit_hgnn_row.log
run_exp Credit/HGNN/HGNN_Unlearning_col_Credit.py    credit_hgnn_col.log
run_exp Credit/HGNN/HGNN_Unlearning_value_Credit.py  credit_hgnn_rowgroup.log

# Credit - HGNNP
run_exp Credit/HGNNP/HGNNP_Unlearning_row_Credit.py  credit_hgnnp_row.log
run_exp Credit/HGNNP/HGNNP_Unlearning_col_Credit.py  credit_hgnnp_col.log
run_exp Credit/HGNNP/HGNNP_Unlearning_value_Credit.py credit_hgnnp_rowgroup.log

# Credit - HGAT
run_exp Credit/HGAT/HGAT_Unlearning_row_Credit.py    credit_hgat_row.log
run_exp Credit/HGAT/HGAT_Unlearning_col_credit.py    credit_hgat_col.log
run_exp Credit/HGAT/HGAT_Unlearning_Value.py         credit_hgat_rowgroup.log

echo "########################################"
echo "# Table III/VII: RELOAD 基线"
echo "########################################"

run_exp baseline_Tabnet/Baseline_RELOAD_ACI.py        reload_aci_row.log
run_exp baseline_Tabnet/Baseline_RELOAD_ACI_COL.py    reload_aci_col.log
run_exp baseline_Tabnet/Baseline_RELOAD_ACI_Value.py  reload_aci_rowgroup.log
run_exp baseline_Tabnet/Baseline_RELOAD_Bank.py       reload_bank_row.log
run_exp baseline_Tabnet/Baseline_RELOAD_Bank_col.py   reload_bank_col.log
run_exp baseline_Tabnet/Baseline_RELOAD_Bank_Value.py reload_bank_rowgroup.log
run_exp baseline_Tabnet/Baseline_RELOAD_Credit.py     reload_credit_row.log
run_exp baseline_Tabnet/Baseline_RELOAD_Credit_col.py reload_credit_col.log
run_exp baseline_Tabnet/Baseline_RELOAD_Credit_value.py reload_credit_rowgroup.log

echo "########################################"
echo "# Appendix C: Fine-Tuning 基线 - ACI"
echo "########################################"

run_exp Baseline_FT_ACI/HGCN_baseline_FT/run_ft_row_zero.py     ft_aci_hgcn_row.log
run_exp Baseline_FT_ACI/HGCN_baseline_FT/run_ft_hgcn_col_zero.py ft_aci_hgcn_col.log
run_exp Baseline_FT_ACI/HGNN_baseline_FT/run_ft_hgnn_row_zero.py ft_aci_hgnn_row.log
run_exp Baseline_FT_ACI/HGNN_baseline_FT/run_ft_hgnn_col_zero.py ft_aci_hgnn_col.log
run_exp Baseline_FT_ACI/HGNNP_baseline_FT/run_ft_hgnnp_row_zero.py ft_aci_hgnnp_row.log
run_exp Baseline_FT_ACI/HGNNP_baseline_FT/run_ft_hgnnp_col_zero.py ft_aci_hgnnp_col.log
run_exp Baseline_FT_ACI/HGAT_baseline_FT/run_ft_hgat_row_zero.py ft_aci_hgat_row.log
run_exp Baseline_FT_ACI/HGAT_baseline_FT/run_ft_hgat_col_zero.py ft_aci_hgat_col.log

echo "########################################"
echo "# Appendix C: Fine-Tuning 基线 - Bank"
echo "########################################"

run_exp bank/FT_bank/FT_HGCN/FT_HGCN_row.py              ft_bank_hgcn_row.log
run_exp bank/FT_bank/FT_HGCN/run_ft_hgcn_bank_col_zero.py ft_bank_hgcn_col.log
run_exp bank/FT_bank/FT_HGNN/run_ft_hgcn_row_zero_bank.py ft_bank_hgnn_row.log
run_exp bank/FT_bank/FT_HGNN/FT_HGNN_col.py              ft_bank_hgnn_col.log
run_exp bank/FT_bank/FT_HGNNP/run_ft_hgnnp_row_bank.py   ft_bank_hgnnp_row.log
run_exp bank/FT_bank/FT_HGNNP/run_ft_hgnnp_col.py        ft_bank_hgnnp_col.log
run_exp bank/FT_bank/FT_HGAT/row.py                      ft_bank_hgat_row.log
run_exp bank/FT_bank/FT_HGAT/col.py                      ft_bank_hgat_col.log

echo "########################################"
echo "# Appendix C: Fine-Tuning 基线 - Credit"
echo "########################################"

run_exp Credit/FT_baseline_Credit/HGCN/row.py   ft_credit_hgcn_row.log
run_exp Credit/FT_baseline_Credit/HGCN/col.py   ft_credit_hgcn_col.log
run_exp Credit/FT_baseline_Credit/HGNN/row.py   ft_credit_hgnn_row.log
run_exp Credit/FT_baseline_Credit/HGNN/col.py   ft_credit_hgnn_col.log
run_exp Credit/FT_baseline_Credit/HGNNP/row.py  ft_credit_hgnnp_row.log
run_exp Credit/FT_baseline_Credit/HGNNP/col.py  ft_credit_hgnnp_col.log
run_exp Credit/FT_baseline_Credit/HGAT/row.py   ft_credit_hgat_row.log
run_exp Credit/FT_baseline_Credit/HGAT/col.py   ft_credit_hgat_col.log

echo "########################################"
echo "# 全部实验完成！"
echo "########################################"
echo "日志保存在: $LOG_DIR"
echo "失败的实验:"
grep -l "✗" $LOG_DIR/run_all.log 2>/dev/null || grep "失败:" $LOG_DIR/run_all.log 2>/dev/null || echo "  (无)"
