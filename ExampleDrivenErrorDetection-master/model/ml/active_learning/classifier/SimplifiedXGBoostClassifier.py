from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

class SimplifiedXGBoostClassifier(object):
    name = 'SimplifiedXGBoost'
    
    def __init__(self, X_train=None, X_test=None, balance=False, feature_names=None):
        self.params = {}
        self.model = None
        self.feature_names = feature_names
        
        self.name = SimplifiedXGBoostClassifier.name
        self.balance = balance
        
        # 如果提供了训练数据，转换为DMatrix格式
        if X_train is not None:
            self.X_train = xgb.DMatrix(X_train, feature_names=feature_names)
        else:
            self.X_train = None
            
        # 如果提供了测试数据，转换为DMatrix格式
        if X_test is not None:
            self.X_test = xgb.DMatrix(X_test, feature_names=feature_names)
        else:
            self.X_test = None
    
    def set_default_params(self):
        """设置默认参数"""
        self.params = {
            'learning_rate': 0.1,
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 0,
            'objective': 'binary:logistic',
            'n_jobs': 4,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8
        }
    
    def run_cross_validation(self, X, y, folds=5):
        """运行交叉验证以找到最佳参数"""
        if not self.params:
            self.set_default_params()
            
        cv_params = {
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'max_depth': [3, 5, 7]
        }
        
        ind_params = self.params.copy()
        
        if self.balance:
            ratio = float(np.sum(y == False)) / np.sum(y == True)
            print("weight ratio: " + str(ratio))
            ind_params['scale_pos_weight'] = ratio
        
        optimized_GBM = GridSearchCV(
            xgb.XGBClassifier(**ind_params),
            cv_params,
            scoring='f1', 
            cv=folds, 
            n_jobs=1, 
            verbose=0
        )
        
        print(X.shape)
        optimized_GBM.fit(X, y)
        
        # 更新参数为最佳参数
        self.params.update(optimized_GBM.best_params_)
        
        return self.params
    
    def train(self, X, y, use_cv=False, cv_folds=5):
        """训练模型"""
        if not self.params:
            self.set_default_params()
        
        # 如果需要，运行交叉验证
        if use_cv:
            self.run_cross_validation(X, y, cv_folds)
        
        # 处理类别不平衡
        if self.balance:
            ratio = float(np.sum(y == False)) / np.sum(y == True)
            print("weight ratio: " + str(ratio))
            self.params['scale_pos_weight'] = ratio
        
        # 转换为DMatrix格式
        if not isinstance(X, xgb.DMatrix):
            xgdmat = xgb.DMatrix(X, y, feature_names=self.feature_names)
        else:
            xgdmat = X
        
        # 训练模型
        self.model = xgb.train(
            self.params, 
            xgdmat, 
            num_boost_round=3000, 
            verbose_eval=False
        )
        
        return self.model
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba > 0.5)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # 转换为DMatrix格式
        if not isinstance(X, xgb.DMatrix):
            if self.feature_names:
                xgdmat = xgb.DMatrix(X, feature_names=self.feature_names)
            else:
                xgdmat = xgb.DMatrix(X)
        else:
            xgdmat = X
        
        # 预测概率
        return self.model.predict(xgdmat)
    
    def evaluate(self, X, y):
        """评估模型性能"""
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        y_pred = self.predict(X)
        
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
    
    def save_model(self, filepath):
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance = self.model.get_score(importance_type='gain')
        
        # 如果有特征名称，按特征名称排序
        if self.feature_names:
            # 确保所有特征都有重要性分数
            full_importance = {}
            for feat in self.feature_names:
                if feat in importance:
                    full_importance[feat] = importance[feat]
                else:
                    full_importance[feat] = 0.0
            
            # 按重要性排序
            sorted_importance = sorted(full_importance.items(), key=lambda x: x[1], reverse=True)
            return sorted_importance
        else:
            # 按重要性排序
            return sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    def explain_prediction(self, X, top_features=5):
        """解释单个预测"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # 转换为DMatrix格式
        if not isinstance(X, xgb.DMatrix):
            if self.feature_names:
                xgdmat = xgb.DMatrix(X, feature_names=self.feature_names)
            else:
                xgdmat = xgb.DMatrix(X)
        else:
            xgdmat = X
        
        # 获取预测贡献
        contribution = self.model.predict(xgdmat, pred_contribs=True)
        
        # 如果是单个样本
        if len(contribution.shape) == 2 and contribution.shape[0] == 1:
            contribution = contribution[0]
            
            # 获取特征重要性
            feature_contributions = []
            for i, feat_name in enumerate(self.feature_names):
                feature_contributions.append((feat_name, contribution[i]))
            
            # 按绝对贡献排序
            sorted_contributions = sorted(
                feature_contributions, 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            return sorted_contributions[:top_features]
        else:
            raise ValueError("This method supports single instance explanation only.")