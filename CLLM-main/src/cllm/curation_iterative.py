import sys
import os
sys.path.append(os.path.abspath('../ src'))

from cllm.utils import *
from cllm.curation import *
def iterative_data_centric_curation(
    X_train_orig,
    y_train_orig,
    X_check,
    y_check,
    curation_metric="aleatoric",
    nest=100,
    curation_ythresh=0.2,
    curation_xthresh=0.2,
    max_iterations=10
):
    """
    迭代版本的 data_centric_curation，自动处理 ambig_train 数据直到为空
    
    Args:
      X_train_orig: 原始训练数据特征
      y_train_orig: 原始训练数据标签
      X_check: 需要进行数据优化的目标数据集特征
      y_check: 目标数据集的真实标签
      curation_metric: 用于数据优化的评估指标类型
      retrain: 是否在X_check数据上重新训练模型
      nest: XGBoost分类器中使用的树的数量
      curation_ythresh: 置信度阈值参数
      curation_xthresh: 不确定性度量的阈值参数
      max_iterations: 最大迭代次数，防止无限循环
      
    Returns:
      所有轮次处理后的最终结果
    """
    
    from xgboost import XGBClassifier
    import numpy as np

    iteration = 0
    
    current_X_train = X_train_orig
    current_y_train = y_train_orig
    
     # 迭代处理数据直到没有模糊数据或达到最大迭代次数
    while iteration < max_iterations:
        print(f"第 {iteration + 1} 轮数据优化开始...")
        
        # 使用原始函数进行数据分类
        # train xgboost on X_train_orig, y_train_orig
        xgb = XGBClassifier(n_estimators=nest)
        xgb.fit(current_X_train, current_y_train)
        

        Curator_xgb = Curator(X=X_check, y=y_check)

        for i in range(1, nest):
            # *** Characterize with Curator [LINE 2] ***
            Curator_xgb.on_epoch_end(clf=xgb, iteration=i)

        if curation_metric == "aleatoric":
            curation_xmetric = Curator_xgb.aleatoric
        elif curation_metric == "epistemic":
            curation_xmetric = Curator_xgb.variability
        elif curation_metric == "entropy":
            curation_xmetric = Curator_xgb.entropy
        elif curation_metric == "mi":
            curation_xmetric = Curator_xgb.mi

        confidence = Curator_xgb.confidence
        # confidence is an array of size [N,1] where N is the number of training data points
        if curation_xthresh == 0:
            print("Using adaptive threshold")
            curation_xthresh = 0.75 * (np.max(curation_xmetric) - np.min(curation_xmetric))
        
        curation_ythresh = curation_ythresh

        curated_train, ambig_train, unlearnable_train = get_groups(
            confidence=confidence,
            aleatoric_uncertainty=curation_xmetric,
            curation_xthresh=curation_xthresh,
            curation_ythresh=curation_ythresh,
        )
        
        
        print(f"  精选数据: {len(curated_train)} 个")
        print(f"  模糊数据: {len(ambig_train)} 个")
        print(f"  不可学习数据: {len(unlearnable_train)} 个")
        
        # 如果没有模糊数据，结束循环
        if len(ambig_train) == 0:
            print("没有更多模糊数据，迭代结束")
            break
        
        # 模拟人工判断过程 - 这里需要您根据实际情况实现人工判断逻辑
        # 在实际应用中，这里应该是一个人工标注的过程
        print("模拟人工判断模糊数据...")
        # 简单示例：假设人工判断后，所有模糊数据都是正确的
        # 实际应用中，您需要在这里集成的模糊数据添加人工标注的逻辑
        human_labeled_train = human_label(X_check[ambig_train], y_check[ambig_train])
        human_labeled_X = X_check[human_labeled_train]
        human_labeled_y = y_check[human_labeled_train]
         # 将人工判断的数据和精选数据合并用于下一轮训练
        next_round_curated_X = X_check[curated_train]
        next_round_curated_y = y_check[curated_train]
         # 合并数据用于下一轮训练
        current_X_train = np.vstack([current_X_train, next_round_curated_X, human_labeled_X])
        current_y_train = np.hstack([current_y_train, next_round_curated_y, human_labeled_y])


        iteration += 1
        
        if iteration >= max_iterations:
            print(f"达到最大迭代次数 {max_iterations}，停止迭代")
            
    all_curated = np.union1d(all_curated, human_labeled_train)
    return all_curated

# 使用示例:
# curated_final = iterative_data_centric_curation(
#     X_train_orig, y_train_orig, X_check, y_check
# )