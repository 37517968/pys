# -*- coding: utf-8 -*-

import os
import sys
# sys.path.append("/home/stu/pys/ExampleDrivenErrorDetection-master/model")
# from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
# from ml.datasets.hospital.HospitalHoloCleanIndices import HospitalHoloCleanIndices
# from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run

model_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, model_path)

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
from ml.datasets.hospital.HospitalHoloCleanIndices import HospitalHoloClean
from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run

data = HospitalHoloClean()


if __name__ == "__main__":
    result = run(dataSet=data,
    classifier_model=XGBoostClassifier,
	ngrams=1,
    iterations=4,
	use_metadata=True, # 引入元数据（metadata）特征。
	use_word2vec=False, # 能捕捉语义相似性的词嵌入特征。
	use_boostclean_metadata=False, # 是否使用 BoostClean 系列算法专用的元数据增强
	w2v_size=100 # Word2Vec向量大小，默认为100
)
