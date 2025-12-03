import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import logging
import datetime
import os

warnings.filterwarnings('ignore')


class ModelLogger:
    """模型训练和预测的日志记录器"""

    def __init__(self, log_file="model_training.log"):
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        """设置日志配置"""
        # 创建logs目录
        if not os.path.exists('logs'):
            os.makedirs('logs')

        log_path = os.path.join('logs', self.log_file)

        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_data_info(self, X, y, data_type="训练"):
        """记录数据信息"""
        unique_labels, counts = np.unique(y, return_counts=True)
        self.logger.info(f"=== {data_type}数据信息 ===")
        self.logger.info(f"样本数: {len(X)}, 特征数: {X.shape[1]}")
        self.logger.info(f"类别数: {len(unique_labels)}")
        self.logger.info(f"类别分布: {dict(zip(unique_labels, counts))}")

        # 数据质量检查
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            self.logger.warning(f"发现 {nan_count} 个NaN值")

        # 特征统计
        self.logger.info(f"特征值范围: {X.min():.6f} ~ {X.max():.6f}")
        self.logger.info(f"特征均值: {X.mean():.6f}, 标准差: {X.std():.6f}")

    def log_training_start(self, optimize_hyperparams):
        """记录训练开始"""
        self.logger.info("=" * 50)
        self.logger.info("开始模型训练")
        self.logger.info(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"超参数优化: {'是' if optimize_hyperparams else '否'}")

    def log_preprocessing(self, original_shape, processed_shape):
        """记录预处理信息"""
        self.logger.info(f"预处理: {original_shape} -> {processed_shape}")
        reduction_ratio = (1 - processed_shape[1] / original_shape[1]) * 100
        self.logger.info(f"维度降低: {reduction_ratio:.1f}%")

    def log_hyperparameter_optimization(self, param_type, best_params, best_score=None, best_scores=None):
        """记录超参数优化结果
        
        Parameters:
        -----------
        param_type : str
            参数类型
        best_params : dict
            最佳参数
        best_score : float, optional
            单个最佳分数（向后兼容）
        best_scores : dict, optional
            多指标评分结果
        """
        # 根据规范，只记录优化完成，不显示具体参数和评分结果
        self.logger.info(f"{param_type}优化完成")
        
        # 注释掉原来的详细信息记录
        # self.logger.info(f"  最佳参数: {best_params}")
        # 
        # if best_scores is not None:
        #     # 记录多指标评分
        #     self.logger.info(f"  多指标评分结果:")
        #     for metric, score in best_scores.items():
        #         self.logger.info(f"    {metric}: {score:.4f}")
        # elif best_score is not None:
        #     # 向后兼容：记录单个最佳分数
        #     self.logger.info(f"  最佳分数: {best_score:.4f}")

    def log_training_complete(self, training_time):
        """记录训练完成"""
        self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")

    def log_prediction_start(self, n_samples):
        """记录预测开始"""
        self.logger.info(f"开始预测 {n_samples} 个样本")

    def log_prediction_results(self, predictions, probabilities):
        """记录预测结果"""
        max_probs = probabilities.max(axis=1)
        self.logger.info(f"预测完成:")
        self.logger.info(f"  平均置信度: {max_probs.mean():.3f}")
        self.logger.info(f"  最高置信度: {max_probs.max():.3f}")
        self.logger.info(f"  最低置信度: {max_probs.min():.3f}")

        # 预测分布
        unique_preds, counts = np.unique(predictions, return_counts=True)
        pred_dist = dict(zip(unique_preds, counts))
        self.logger.info(f"  预测分布: {pred_dist}")

    def log_evaluation_results(self, results):
        """记录评估结果"""
        self.logger.info("=== 模型评估结果 ===")
        for metric, value in results.items():
            if isinstance(value, dict):
                self.logger.info(f"{metric}:")
                for k, v in value.items():
                    self.logger.info(f"  {k}: {v:.4f}")
            else:
                self.logger.info(f"{metric}: {value:.4f}")

    def log_error(self, error_msg, exception=None):
        """记录错误信息"""
        self.logger.error(f"错误: {error_msg}")
        if exception:
            self.logger.error(f"异常详情: {str(exception)}")

    def log_warning(self, warning_msg):
        """记录警告信息"""
        self.logger.warning(warning_msg)


class SmallSampleRegionClassifier:
    def __init__(self, enable_logging=True):
        self.scaler = StandardScaler()
        self.pca = None  # 将在preprocess_data中根据样本数动态初始化
        self.label_encoder = LabelEncoder()
        self.models = self._init_models()
        self.ensemble = None
        self.use_simple_knn = True

        # 初始化日志记录器
        if enable_logging:
            self.logger = ModelLogger()
        else:
            self.logger = None
        self.feature_importance = None
        self.component_names = None
        
        # 保存符合条件的模型列表
        self.qualified_models = []  # 存储满足条件的模型及其性能指标

    def _init_models(self):
        return {
            'knn1': KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
            'knn_cos': KNeighborsClassifier(n_neighbors=1, metric='cosine'),
            'knn3': KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        }

    def _extract_advanced_features(self, X):
        # 基础统计特征 - 替换均值和标准差为绝对偏差中位数
        # 计算每个样本的中位数
        medians = np.median(X, axis=1)
        
        # 计算绝对偏差中位数 (MAD)
        mad = []
        for i in range(X.shape[0]):
            abs_dev = np.abs(X[i, :] - medians[i])
            mad_value = np.median(abs_dev)
            mad.append(mad_value)
        mad = np.array(mad)
        
        stats = np.column_stack([
            np.max(X, axis=1),
            np.min(X, axis=1),
            np.median(X, axis=1),
            mad  # 绝对偏差中位数
        ])

        # 特征间比例组合 - 使用前6个特征的C(6,2)组合
        ratios = []
        for i in range(min(6, X.shape[1])):
            for j in range(i + 1, min(6, X.shape[1])):
                ratio = X[:, i] / (X[:, j] + 1e-8)
                ratios.append(ratio)
        ratio_features = np.column_stack(ratios) if ratios else np.empty((X.shape[0], 0))

        # 不再计算分布特征
        
        return np.hstack([X, stats, ratio_features])

    def preprocess_data(self, X, y=None, fit=True):
        X_clean = np.nan_to_num(X, nan=0.0)

        if fit:
            X_features = self._extract_advanced_features(X_clean)
            X_scaled = self.scaler.fit_transform(X_features)
            
            # 优先使用最优主成分数（如果存在）
            if hasattr(self, 'n_components') and self.n_components is not None:
                n_components = self.n_components
                # 根据规范，不显示最优PCA主成分数
                # print(f"使用最优PCA主成分数: {n_components}")
            else:
                # 根据样本数量动态设置PCA主成分数量为样本数的1/5
                n_samples = X.shape[0]
                n_components = max(1, int(n_samples / 5))  # 确保至少有1个主成分
                print(f"根据样本数{n_samples}，自动设置PCA主成分数为{n_components}")
            
            # 重新初始化PCA
            self.pca = PCA(n_components=n_components, whiten=True)
            
            X_pca = self.pca.fit_transform(X_scaled)
            if self.logger:
                self.logger.log_preprocessing(X_features.shape, X_pca.shape)
            
            if y is not None:
                y_encoded = self.label_encoder.fit_transform(y)
                return X_pca, y_encoded
            return X_pca
        else:
            X_features = self._extract_advanced_features(X_clean)
            X_scaled = self.scaler.transform(X_features)
            X_pca = self.pca.transform(X_scaled)
            return X_pca

    def train(self, X, y, optimize_hyperparams=True):
        """
        训练模型，可选择是否优化超参数

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        optimize_hyperparams : bool, default=True
            是否优化超参数
        """
        import time
        start_time = time.time()

        # 记录训练开始
        if self.logger:
            self.logger.log_training_start(optimize_hyperparams)
            self.logger.log_data_info(X, y, "训练")

        n_samples = len(X)
        n_unique_labels = len(np.unique(y))

        # 样本数检查和警告
        print(f"训练数据信息：样本数={n_samples}, 类别数={n_unique_labels}")

        if n_samples < 10:
            warning_msg = "样本数过少(< 10)，模型性能可能不稳定"
            print(f"⚠️  警告：{warning_msg}")
            if self.logger:
                self.logger.log_warning(warning_msg)

        if n_samples == n_unique_labels:
            warning_msg = "每个类别只有1个样本，这是极小样本学习问题"
            print(f"⚠️  警告：{warning_msg}")
            print("   建议：考虑收集更多数据或使用其他机器学习方法")
            if self.logger:
                self.logger.log_warning(warning_msg)

        if n_samples < 50:
            info_msg = "样本数较少，将使用留一法交叉验证(LOOCV)进行评估"
            print(f"ℹ️  提示：{info_msg}")
            if self.logger:
                self.logger.logger.info(info_msg)

        # 如果选择优化超参数，先进行参数优化
        if optimize_hyperparams:
            print("\n=== 开始超参数优化 ===")

            # 优化PCA组件数
            print("1. 优化PCA组件数...")
            # 使用基于评分指标的优化方法，默认同时使用f1_macro和accuracy
            pca_results = self.optimize_pca_components(X, y, scoring_metrics=['f1_macro'])
            # 保存最优主成分数作为类属性
            self.n_components = pca_results['best_n_components']
            # 保存所有测试结果
            self.pca_test_results = pca_results
            
            # 步骤：筛选符合条件的模型（F1 > 0.9, LOOCV > 0.9, Variance > 0.8）
            print("\n=== 筛选符合条件的模型 ===")
            self.qualified_models = []
            threshold_f1 = 0.9
            threshold_loocv = 0.9
            threshold_variance = 0.8
            
            for n_comp, result in pca_results['all_results'].items():
                f1_score = result['avg_scores'].get('f1_macro', 0)
                loocv_acc = result.get('loocv_accuracy', 0)
                variance_exp = result.get('variance_explained', 0)
                
                # 检查是否满足所有条件
                if f1_score > threshold_f1 and loocv_acc > threshold_loocv and variance_exp > threshold_variance:
                    self.qualified_models.append({
                        'n_components': n_comp,
                        'f1_macro': f1_score,
                        'loocv_accuracy': loocv_acc,
                        'variance_explained': variance_exp
                    })
                    print(f"  ✓ n_components={n_comp}: F1={f1_score:.3f}, LOOCV={loocv_acc:.3f}, Variance={variance_exp:.3f}")
            
            if self.qualified_models:
                print(f"\n找到 {len(self.qualified_models)} 个符合条件的模型")
            else:
                print(f"\n警告：没有找到符合条件的模型，将使用最优模型")
                self.qualified_models = [{
                    'n_components': pca_results['best_n_components'],
                    'f1_macro': pca_results.get('best_score', 0),
                    'loocv_accuracy': 0,
                    'variance_explained': 0
                }]
            
            if self.logger:
                self.logger.log_hyperparameter_optimization(
                    "PCA",
                    {"n_components": pca_results['best_n_components']},
                    best_scores=pca_results.get('best_scores', {})
                )

            # 为每个符合条件的主成数组合进行K值优化
            print("2. 为每个主成数组合优化K值...")
            self.k_optimization_results = {}  # 保存每个主成数组合的K值优化结果
            for model_config in self.qualified_models:
                n_comp = model_config['n_components']
                print(f"  优化主成数组合 {n_comp} 的K值...")
                # 临时设置PCA主成数进行K值优化
                original_n_components = self.n_components
                self.n_components = n_comp
                k_results = self.optimize_k_values(X, y, scoring_metrics=['f1_macro'])
                self.k_optimization_results[n_comp] = k_results
                # 恢复原来的PCA主成数
                self.n_components = original_n_components
        else:
            print("\n=== 跳过超参数优化，使用默认参数 ===")

        # 使用符合条件的模型进行数据预处理和训练
        original_shape = X.shape
        
        # 为每个符合条件的模型训练一个PCA+KNN组合
        print(f"\n=== 训练 {len(self.qualified_models)} 个符合条件的模型 ===")
        self.trained_models = []  # 保存训练完的模型
        
        for idx, model_config in enumerate(self.qualified_models):
            n_comp = model_config['n_components']
            print(f"\n训练模型 {idx + 1}/{len(self.qualified_models)}: n_components={n_comp}")
            
            # 重新设置PCA主成分数
            self.n_components = n_comp
            
            # 执行数据预处理
            X_processed, y_processed = self.preprocess_data(X, y, fit=True)
            print(f"  预处理后特征维度: {X_processed.shape}")
            
            # 获取该主成数组合的最优K值（如果存在优化结果）
            best_k_values = {'euclidean': 1, 'manhattan': 1, 'cosine': 1}
            if hasattr(self, 'k_optimization_results') and n_comp in self.k_optimization_results:
                k_results = self.k_optimization_results[n_comp]
                default_scoring = 'f1_macro' if 'f1_macro' in ['f1_macro'] else ['f1_macro'][0]
                best_k_values['euclidean'] = k_results['euclidean']['best_k_dict'].get(default_scoring, 1)
                best_k_values['manhattan'] = k_results['manhattan']['best_k_dict'].get(default_scoring, 1)
                best_k_values['cosine'] = k_results['cosine']['best_k_dict'].get(default_scoring, 1)
                print(f"  使用优化后的K值: euclidean={best_k_values['euclidean']}, manhattan={best_k_values['manhattan']}, cosine={best_k_values['cosine']}")
            else:
                print(f"  使用默认K值: K=1")
            
            # 为这个配置训练三个KNN模型，使用优化后的K值
            knn_models = {
                'euclidean': KNeighborsClassifier(n_neighbors=best_k_values['euclidean'], metric='euclidean'),
                'manhattan': KNeighborsClassifier(n_neighbors=best_k_values['manhattan'], metric='manhattan'),
                'cosine': KNeighborsClassifier(n_neighbors=best_k_values['cosine'], metric='cosine')
            }
            
            for metric_name, knn_model in knn_models.items():
                knn_model.fit(X_processed, y_processed)
            
            # 保存这个模型配置
            self.trained_models.append({
                'n_components': n_comp,
                'scaler': self.scaler,
                'pca': self.pca,
                'models': knn_models,
                'X_train': X_processed,
                'y_train': y_processed,
                'performance': {
                    'f1_macro': model_config['f1_macro'],
                    'loocv_accuracy': model_config['loocv_accuracy'],
                    'variance_explained': model_config['variance_explained']
                }
            })
        
        print(f"\n✅ 模型训练完成，共训练 {len(self.trained_models)} 个模型")
        
        # 显示符合条件的模型信息
        print("\n=== 符合条件的模型 ===")
        for model in self.qualified_models:
            print(f"  n_components={model['n_components']}: F1={model['f1_macro']:.3f}, LOOCV={model['loocv_accuracy']:.3f}, Variance={model['variance_explained']:.3f}")
        
        # 为了兼容性，设置默认的self属性
        if self.trained_models:
            last_model = self.trained_models[-1]
            self.knn_euclidean = last_model['models']['euclidean']
            self.knn_manhattan = last_model['models']['manhattan']
            self.knn_cosine = last_model['models']['cosine']
            self.X_train = last_model['X_train']
            self.y_train = last_model['y_train']
            self.y_train_encoded = last_model['y_train']
            self.y_train_original = y
        else:
            # 备用方案
            X_processed, y_processed = self.preprocess_data(X, y, fit=True)
            self.knn_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
            self.knn_manhattan = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
            self.knn_cosine = KNeighborsClassifier(n_neighbors=1, metric='cosine')
            self.knn_euclidean.fit(X_processed, y_processed)
            self.knn_manhattan.fit(X_processed, y_processed)
            self.knn_cosine.fit(X_processed, y_processed)
            self.X_train = X_processed
            self.y_train = y_processed
            self.y_train_encoded = y_processed
            self.y_train_original = y
        
        if self.logger:
            self.logger.log_preprocessing(original_shape, self.X_train.shape)

        # 记录训练完成
        training_time = time.time() - start_time
        print("✅ 模型训练完成")

        if self.logger:
            self.logger.log_training_complete(training_time)

        return self
    def _create_default_k_results(self):
        """创建默认的K值优化结果"""
        return {
            'euclidean': {'best_k': 1, 'best_score': 0.0,
                          'all_results': {1: {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}}},
            'manhattan': {'best_k': 1, 'best_score': 0.0,
                          'all_results': {1: {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}}},
            'cosine': {'best_k': 1, 'best_score': 0.0,
                       'all_results': {1: {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}}}
        }

    def optimize_k_values(self, X, y, k_range=None, cv_folds=3, scoring_metrics=['f1_macro']):
        """
        使用交叉验证优化K值（使用Pipeline避免数据泄露）

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        k_range : list, optional
            K值搜索范围，默认为1到min(5, n_samples-1)
        cv_folds : int, default=3
            交叉验证折数
        scoring_metrics : list, default=['f1_macro', 'accuracy']
            评估指标列表

        Returns:
        --------
        dict : 包含优化结果的字典
        """
        X_clean = np.nan_to_num(X, nan=0.0)
        X_features = self._extract_advanced_features(X_clean)
        y_encoded = self.label_encoder.fit_transform(y)
        n_samples = len(X_features)

        if k_range is None:
            # 对于16地区×3样本的数据集，每类只有3个样本，K值应该限制为1-2
            n_unique_labels = len(np.unique(y))
            samples_per_class = n_samples // n_unique_labels
            if samples_per_class <= 3:
                # 每类样本数较少时，K值限制为1-2
                max_k = min(2, n_samples - 1)
            else:
                # 原有逻辑
                max_k = min(5, n_samples - 1)
            k_range = list(range(1, max_k + 1))

        # 过滤掉超过样本数的K值
        k_range = [k for k in k_range if k < n_samples]

        if not k_range:
            print(f"警告：样本数({n_samples})过少，无法进行K值优化，使用默认K=1")
            return self._create_default_k_results()

        # 检查每个类别的样本数
        n_unique_labels = len(np.unique(y))
        samples_per_class = n_samples // n_unique_labels

        # 对于极小样本或每类样本数很少，强制使用LOOCV
        if n_samples <= 20 or samples_per_class < cv_folds:
            cv = LeaveOneOut()
            print(f"样本数({n_samples})较少或每类样本数({samples_per_class})不足，使用留一法交叉验证(LOOCV)")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        optimization_results = {}

        # 测试不同距离度量的KNN
        metrics = ['euclidean', 'manhattan', 'cosine']

        for metric in metrics:
            metric_results = {}
            # 为每个评价指标保存最佳K值和分数
            best_k_dict = {scoring: 1 for scoring in scoring_metrics}
            best_score_dict = {scoring: 0 for scoring in scoring_metrics}

            for k in k_range:
                    if k >= n_samples:
                        continue

                    # 创建包含预处理的Pipeline，使用当前设置的PCA主成数
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=self.n_components, whiten=True)),  # 使用当前的主成数组合
                        ('knn', KNeighborsClassifier(n_neighbors=k, metric=metric))
                    ])
                    
                    k_results = {}
                    
                    # 使用cross_validate来同时计算多个评分指标
                    from sklearn.model_selection import cross_validate
                    cv_results = cross_validate(pipeline, X_features, y_encoded, cv=cv, scoring=scoring_metrics)
                    
                    # 提取每个指标的分数
                    for scoring in scoring_metrics:
                        # cross_validate返回的键格式为'test_' + metric
                        test_key = f'test_{scoring}'
                        if test_key in cv_results:
                            scores = cv_results[test_key]
                            mean_score = scores.mean()
                            std_score = scores.std()
                            
                            k_results[scoring] = {
                                'mean_score': mean_score,
                                'std_score': std_score,
                                'scores': scores
                            }
                            
                            # 更新该指标的最佳K值
                            if mean_score > best_score_dict[scoring]:
                                best_score_dict[scoring] = mean_score
                                best_k_dict[scoring] = k

                    metric_results[k] = k_results

            # 更新模型的K值 - 默认使用f1_macro的结果
            default_scoring = 'f1_macro' if 'f1_macro' in scoring_metrics else scoring_metrics[0]
            optimization_results[metric] = {
                'best_k': best_k_dict[default_scoring],  # 向后兼容
                'best_score': best_score_dict[default_scoring],  # 向后兼容
                'best_k_dict': best_k_dict,  # 每个指标的最佳K值
                'best_score_dict': best_score_dict,  # 每个指标的最佳分数
                'all_results': metric_results
            }

        # 更新模型的K值 - 默认使用f1_macro的结果        default_scoring = 'f1_macro' if 'f1_macro' in scoring_metrics else scoring_metrics[0]
        
        print(f"K值优化结果 (默认使用{default_scoring}指标):")
        for metric in metrics:
            print(f"  {metric}:")
            for scoring in scoring_metrics:
                print(f"    {scoring}: K={optimization_results[metric]['best_k_dict'][scoring]}, 分数={optimization_results[metric]['best_score_dict'][scoring]:.3f}")
        
        self.optimized_models = {
            'knn_euclidean': KNeighborsClassifier(n_neighbors=optimization_results['euclidean']['best_k_dict'][default_scoring],
                                                  metric='euclidean'),
            'knn_manhattan': KNeighborsClassifier(n_neighbors=optimization_results['manhattan']['best_k_dict'][default_scoring],
                                                  metric='manhattan'),
            'knn_cosine': KNeighborsClassifier(n_neighbors=optimization_results['cosine']['best_k_dict'][default_scoring], metric='cosine')
        }
        
        # 移除直接拟合scaler和pca的代码，避免数据泄露
        # 让scaler和pca在正式训练时通过Pipeline拟合

        return optimization_results

    def evaluate_with_loocv(self, X, y):
        """
        使用留一法交叉验证评估模型（避免数据泄露）
        """
        X_clean = np.nan_to_num(X, nan=0.0)
        X_features = self._extract_advanced_features(X_clean)
        y_encoded = self.label_encoder.fit_transform(y)

        loo = LeaveOneOut()
        results = {}

        # 创建包含预处理的Pipeline
        pipelines = {}
        for name, model in self.models.items():
            pipelines[name] = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', self.pca),
                ('classifier', model)
            ])

        # 创建集成模型的Pipeline
        ensemble_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', self.pca),
            ('ensemble', self.ensemble)
        ])

        for name, pipeline in pipelines.items():
            scores = cross_val_score(pipeline, X_features, y_encoded,
                                     cv=loo, scoring='f1_macro')
            results[name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }

        ensemble_scores = cross_val_score(ensemble_pipeline, X_features, y_encoded,
                                          cv=loo, scoring='f1_macro')
        results['ensemble'] = {
            'mean_f1': ensemble_scores.mean(),
            'std_f1': ensemble_scores.std(),
            'scores': ensemble_scores
        }

        return results

    def evaluate_with_cv(self, X, y, cv_folds=3,
                         scoring_metrics=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']):
        """
        使用k-fold交叉验证评估模型

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        cv_folds : int, default=3
            交叉验证折数
        scoring_metrics : list, default=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']
            评估指标列表

        Returns:
        --------
        dict : 包含评估结果的字典
        """
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        n_samples = len(X_processed)

        # 检查每个类别的样本数
        n_unique_labels = len(np.unique(y))
        samples_per_class = n_samples // n_unique_labels

        # 自动选择合适的交叉验证策略
        if n_samples <= 20 or samples_per_class < cv_folds:
            cv = LeaveOneOut()
            print(f"模型评估：样本数({n_samples})较少或每类样本数({samples_per_class})不足，使用留一法交叉验证")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            print(f"模型评估：使用{cv_folds}折分层交叉验证")

        results = {}

        for name, model in self.models.items():
            model_results = {}

            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X_processed, y_processed, cv=cv, scoring=metric)
                    model_results[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores
                    }
                except Exception as e:
                    print(f"评估指标{metric}在模型{name}上失败: {e}")
                    model_results[metric] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'scores': []
                    }

            results[name] = model_results

        return results



    def optimize_pca_components(self, X, y, n_components_range=None, cv_folds=3, scoring_metrics=['f1_macro']):
        """
        优化PCA组件数（使用Pipeline避免数据泄露）
        
        现在确保同时保存和返回f1_macro和accuracy指标，以便在UI中显示
        for i in range(best_n_comp):
            # 仅分析主成分中原始特征的部分
        优化PCA组件数（使用Pipeline避免数据泄露）

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        n_components_range : list, optional
            PCA组件数搜索范围
        cv_folds : int, default=3
            交叉验证折数
        scoring_metrics : list, default=['f1_macro', 'accuracy']
            评估指标列表

        Returns:
        --------
        dict : 包含优化结果的字典
        """
        X_clean = np.nan_to_num(X, nan=0.0)
        X_features = self._extract_advanced_features(X_clean)
        
        # 注意：这里不再对整个数据集进行缩放，缩放将在Pipeline中进行

        n_samples, n_features = X_features.shape

        if n_components_range is None:
            # 对于16地区×3样本的数据集（48样本），PCA组件数范围调整为2-15
            n_unique_labels = len(np.unique(y))
            if n_unique_labels >= 10:  # 对于多类别问题（如16类）
                max_components = min(min(n_samples, n_features) - 1, 15)
                n_components_range = list(range(2, max_components + 1))
            else:
                # 原有逻辑：对于类别较少的问题
                max_components = min(min(n_samples, n_features) - 1, 10)
                n_components_range = list(range(2, max_components + 1))

        # 过滤掉超过样本数或特征数的组件数
        n_components_range = [n for n in n_components_range if n < min(n_samples, n_features)]

        if not n_components_range:
            # 根据规范，不显示具体组件数信息
            # print(f"警告：样本数({n_samples})或特征数({n_features})过少，无法进行PCA优化，使用默认组件数")
            return {'best_n_components': min(5, min(n_samples, n_features) - 1), 'best_score': 0.0, 'all_results': {}}

        # 对于极小样本或每类样本数较少的情况，强制使用LOOCV
        n_unique_labels = len(np.unique(y))
        samples_per_class = n_samples // n_unique_labels

        if n_samples <= 20 or samples_per_class <= 5:
            cv = LeaveOneOut()
            # 根据规范，简化PCA优化信息显示
            print("PCA优化：使用留一法交叉验证(LOOCV)")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            # 根据规范，简化PCA优化信息显示
            print("PCA优化：使用分层交叉验证")

        y_encoded = self.label_encoder.fit_transform(y)

        pca_results = {}
        best_n_components_dict = {metric: min(n_components_range) if n_components_range else 2 for metric in scoring_metrics}
        best_scores_dict = {metric: 0 for metric in scoring_metrics}

        # 恢复显示PCA优化测试组件数范围的信息
        print(f"PCA优化：测试组件数范围 {n_components_range}")

        for n_comp in n_components_range:
            try:
                # 测试不同的KNN模型，使用Pipeline确保每个CV折中独立进行PCA降维
                models_to_test = {
                    'euclidean': Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=n_comp, whiten=True)),
                        ('knn', KNeighborsClassifier(n_neighbors=1, metric='euclidean'))
                    ]),
                    'manhattan': Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=n_comp, whiten=True)),
                        ('knn', KNeighborsClassifier(n_neighbors=1, metric='manhattan'))
                    ]),
                    'cosine': Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=n_comp, whiten=True)),
                        ('knn', KNeighborsClassifier(n_neighbors=1, metric='cosine'))
                    ])
                }

                component_scores = {}
                # 为每个评价指标保存分数
                metric_scores = {metric: 0 for metric in scoring_metrics}
                
                for metric_name, pipeline in models_to_test.items():
                    component_scores[metric_name] = {}
                    
                    # 分别计算每个指标，确保准确性
                    for metric in scoring_metrics:
                        # 使用cross_val_score计算单个指标，避免潜在的多指标计算问题
                        scores = cross_val_score(pipeline, X_features, y_encoded, cv=cv, scoring=metric)
                        component_scores[metric_name][metric] = {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            'scores': scores
                        }
                        # 累加每个模型的该指标分数
                        metric_scores[metric] += scores.mean()
                
                # 计算每个指标的平均分数
                avg_scores = {metric: score / len(models_to_test) for metric, score in metric_scores.items()}

                # 计算方差解释率需要单独拟合
                temp_scaler = StandardScaler()
                X_scaled_temp = temp_scaler.fit_transform(X_features)
                temp_pca = PCA(n_components=n_comp, whiten=True)
                temp_pca.fit(X_scaled_temp)
                variance_explained = temp_pca.explained_variance_ratio_.sum()

                # 计算LOOCV准确率
                from sklearn.metrics import accuracy_score
                
                # 使用LOOCV计算准确率
                loo = LeaveOneOut()
                loo_accuracies = []
                
                # 对每个CV折计算准确率
                for train_idx, test_idx in loo.split(X_features):
                    # 分割数据
                    X_train_fold, X_test_fold = X_features[train_idx], X_features[test_idx]
                    y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]
                    
                    # 创建Pipeline进行预处理和分类
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=n_comp, whiten=True)),
                        ('knn', KNeighborsClassifier(n_neighbors=1, metric='euclidean'))
                    ])
                    
                    # 训练和预测
                    pipeline.fit(X_train_fold, y_train_fold)
                    y_pred = pipeline.predict(X_test_fold)
                    
                    # 计算准确率
                    acc = accuracy_score(y_test_fold, y_pred)
                    loo_accuracies.append(acc)
                
                # 计算平均LOOCV准确率
                avg_loocv_accuracy = np.mean(loo_accuracies) if loo_accuracies else 0.0

                # 计算模型可靠性估计：基于模型性能和预测稳定性
                # 注意：这里不是样本级别的置信度，而是模型在特定PCA组件数下的整体可靠性指标
                # 使用模型间一致性和性能指标的加权组合
                # 计算模型间的一致性（三种模型性能的标准差越小，一致性越高）
                f1_scores = [
                    component_scores['euclidean']['f1_macro']['mean'],
                    component_scores['manhattan']['f1_macro']['mean'],
                    component_scores['cosine']['f1_macro']['mean']
                ]
                f1_std = np.std(f1_scores)
                
                # 计算模型可靠性：平均性能与一致性的组合
                # 一致性权重设为0.3，性能权重设为0.7
                consistency_term = 1.0 - min(f1_std * 3, 1.0)  # 标准差越大，一致性越低
                performance_term = np.mean(f1_scores)
                model_reliability = 0.3 * consistency_term + 0.7 * performance_term
                
                pca_results[n_comp] = {
                    'avg_scores': avg_scores,
                    'individual_scores': component_scores,
                    'variance_explained': variance_explained,
                    'model_reliability': model_reliability,
                    'loocv_accuracy': avg_loocv_accuracy  # 添加LOOCV准确率
                }

                # 更新每个指标的最佳组件数
                for metric in scoring_metrics:
                    if avg_scores[metric] > best_scores_dict[metric]:
                        best_scores_dict[metric] = avg_scores[metric]
                        best_n_components_dict[metric] = n_comp

            except Exception as e:
                # 根据规范，简化错误信息显示
                print(f"PCA组件数测试失败")
                continue

        # 更新PCA组件数 - 默认使用f1_macro的最佳组件数
        if 'f1_macro' in best_n_components_dict and best_n_components_dict['f1_macro'] > 0:
            best_n_components = best_n_components_dict['f1_macro']
            best_score = best_scores_dict['f1_macro']
        else:
            # 如果没有f1_macro，则使用第一个指标的结果
            first_metric = list(best_n_components_dict.keys())[0]
            best_n_components = best_n_components_dict[first_metric]
            best_score = best_scores_dict[first_metric]
            
        # 设置最优组件数，用于preprocess_data方法
        self.n_components = best_n_components
        # 只初始化PCA，不直接拟合，避免数据泄露
        self.pca = PCA(n_components=best_n_components, whiten=True)
        # 根据规范，只显示"PCA优化完成"，不显示最优组件数
        print("PCA优化完成")

        return {
            'best_n_components': best_n_components,  # 向后兼容
            'best_score': best_score,  # 向后兼容
            'best_n_components_dict': best_n_components_dict,  # 每个指标的最佳组件数
            'best_scores_dict': best_scores_dict,  # 每个指标的最佳分数
            'all_results': pca_results,
            # 返回一个按主成分数量排序的列表，方便UI展示
            'sorted_results': sorted(pca_results.items(), key=lambda x: x[0])
        }

    def _calculate_adaptive_weights(self, distances_e, distances_m, distances_c):
        # 基于距离的自适应权重
        inv_dist_e = 1.0 / (distances_e + 1e-8)
        inv_dist_m = 1.0 / (distances_m + 1e-8)
        inv_dist_c = 1.0 / (distances_c + 1e-8)

        total_inv = inv_dist_e + inv_dist_m + inv_dist_c
        weight_e = inv_dist_e / total_inv
        weight_m = inv_dist_m / total_inv
        weight_c = inv_dist_c / total_inv

        return weight_e, weight_m, weight_c

    def evaluate_predictions(self, y_true, y_pred):
        """
        评估预测结果的正确性
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        
        Returns:
        --------
        dict : 包含预测正确/错误信息的字典
        """
        correct = np.array(y_true) == np.array(y_pred)
        num_correct = np.sum(correct)
        num_total = len(y_true)
        num_incorrect = num_total - num_correct
        accuracy = num_correct / num_total if num_total > 0 else 0.0
        
        return {
            'correct': correct,
            'num_correct': int(num_correct),
            'num_incorrect': int(num_incorrect),
            'num_total': int(num_total),
            'accuracy': float(accuracy)
        }

    def predict(self, X):
        # 如果有多个训练的模型，使用多模型选择方式
        if hasattr(self, 'trained_models') and len(self.trained_models) > 1:
            print(f"\n正在使用 {len(self.trained_models)} 个模型进行预测...")
            return self._predict_with_model_selection(X)
        else:
            return self._predict_single_model(X)
    
    def _predict_single_model(self, X):
        # 记录预测开始
        if self.logger:
            self.logger.log_prediction_start(len(X))

        X_processed = self.preprocess_data(X, fit=False)

        pred_euclidean = self.knn_euclidean.predict(X_processed)
        pred_manhattan = self.knn_manhattan.predict(X_processed)
        pred_cosine = self.knn_cosine.predict(X_processed)

        distances_euclidean, _ = self.knn_euclidean.kneighbors(X_processed)
        distances_manhattan, _ = self.knn_manhattan.kneighbors(X_processed)
        distances_cosine, _ = self.knn_cosine.kneighbors(X_processed)

        final_predictions = []
        for i in range(len(X_processed)):
            dist_e = distances_euclidean[i][0]
            dist_m = distances_manhattan[i][0]
            dist_c = distances_cosine[i][0]

            weight_e, weight_m, weight_c = self._calculate_adaptive_weights(dist_e, dist_m, dist_c)

            # 如果某个距离明显最小，使用该预测
            if weight_e > 0.6:
                final_predictions.append(pred_euclidean[i])
            elif weight_m > 0.6:
                final_predictions.append(pred_manhattan[i])
            elif weight_c > 0.6:
                final_predictions.append(pred_cosine[i])
            else:
                # 否则使用加权投票
                votes = [pred_euclidean[i], pred_manhattan[i], pred_cosine[i]]
                if len(set(votes)) == 1:
                    final_predictions.append(votes[0])
                else:
                    final_predictions.append(pred_euclidean[i])  # 默认欧氏距离

        predictions = self.label_encoder.inverse_transform(final_predictions)

        # 记录预测结果
        if self.logger:
            probabilities = self.predict_proba(X)
            self.logger.log_prediction_results(predictions, probabilities)

        return predictions
    
    def _predict_with_model_selection(self, X):
        """
        使用多模型选择：各模型分别预测，选择平均置信度最高的模型
        """
        model_predictions = []  # 保存每个模型的预测结果
        model_confidences = []  # 保存每个模型的平均置信度
        
        print(f"\n正在使用第1个模型进行预测...")
        
        # 为每个训练的模型进行预测
        for idx, trained_model in enumerate(self.trained_models):
            print(f"  模型 {idx + 1}/{len(self.trained_models)} (n_components={trained_model['n_components']})", end="")
            
            # 使用这个模型的PCA和scaler预处理数据
            X_clean = np.nan_to_num(X, nan=0.0)
            X_features = self._extract_advanced_features(X_clean)
            X_scaled = trained_model['scaler'].transform(X_features)
            X_processed = trained_model['pca'].transform(X_scaled)
            
            # 使用这个模型的三个KNN模型预测
            pred_e = trained_model['models']['euclidean'].predict(X_processed)
            pred_m = trained_model['models']['manhattan'].predict(X_processed)
            pred_c = trained_model['models']['cosine'].predict(X_processed)
            
            # 计算置信度
            dist_e, _ = trained_model['models']['euclidean'].kneighbors(X_processed)
            dist_m, _ = trained_model['models']['manhattan'].kneighbors(X_processed)
            dist_c, _ = trained_model['models']['cosine'].kneighbors(X_processed)
            
            # 计算平均置信度
            confidences = []
            final_preds = []
            
            for i in range(len(X_processed)):
                # 计算这个样本的置信度
                conf_e = max(0.5, 1.0 / (1.0 + dist_e[i][0] * 2))
                conf_m = max(0.5, 1.0 / (1.0 + dist_m[i][0] * 1.5))
                conf_c = max(0.5, 1.0 / (1.0 + dist_c[i][0] * 3))
                avg_conf = (conf_e + conf_m + conf_c) / 3
                confidences.append(avg_conf)
                
                # 选择最优预测
                weight_e, weight_m, weight_c = self._calculate_adaptive_weights(dist_e[i][0], dist_m[i][0], dist_c[i][0])
                if weight_e > 0.6:
                    final_preds.append(pred_e[i])
                elif weight_m > 0.6:
                    final_preds.append(pred_m[i])
                elif weight_c > 0.6:
                    final_preds.append(pred_c[i])
                else:
                    votes = [pred_e[i], pred_m[i], pred_c[i]]
                    if len(set(votes)) == 1:
                        final_preds.append(votes[0])
                    else:
                        final_preds.append(pred_e[i])
            
            model_predictions.append(final_preds)
            avg_confidence = np.mean(confidences)
            model_confidences.append(avg_confidence)
            print(f" - 平均置信度: {avg_confidence:.4f}")
        
        # 选择平均置信度最高的模型
        best_model_idx = np.argmax(model_confidences)
        best_confidence = model_confidences[best_model_idx]
        best_n_components = self.trained_models[best_model_idx]['n_components']
        
        print(f"\n✅ 选择最优模型: 模型 {best_model_idx + 1} (n_components={best_n_components}, 平均置信度={best_confidence:.4f})")
        
        # 保存最优模型的信息，供后续analyze_pca_feature_contributions使用
        self.best_model_idx = best_model_idx
        self.best_n_components = best_n_components
        self.best_model_confidence = best_confidence
        
        # 使用最优模型的预测结果
        final_predictions = model_predictions[best_model_idx]
        predictions = self.label_encoder.inverse_transform(final_predictions)
        
        return predictions
    
    def predict_proba(self, X):
        # 如果有多个训练的模型，使用最优模型的预处理
        if hasattr(self, 'best_model_idx') and hasattr(self, 'trained_models') and len(self.trained_models) > 1:
            best_model = self.trained_models[self.best_model_idx]
            X_clean = np.nan_to_num(X, nan=0.0)
            X_features = self._extract_advanced_features(X_clean)
            X_scaled = best_model['scaler'].transform(X_features)
            X_processed = best_model['pca'].transform(X_scaled)
            
            knn_euclidean = best_model['models']['euclidean']
            knn_manhattan = best_model['models']['manhattan']
            knn_cosine = best_model['models']['cosine']
        else:
            # 单模型情形，使用当前的KNN模型
            X_processed = self.preprocess_data(X, fit=False)
            knn_euclidean = self.knn_euclidean
            knn_manhattan = self.knn_manhattan
            knn_cosine = self.knn_cosine

        distances_euclidean, indices_euclidean = knn_euclidean.kneighbors(X_processed)
        distances_manhattan, indices_manhattan = knn_manhattan.kneighbors(X_processed)
        distances_cosine, indices_cosine = knn_cosine.kneighbors(X_processed)

        proba = np.zeros((len(X_processed), len(self.label_encoder.classes_)))

        for i in range(len(X_processed)):
            dist_e = distances_euclidean[i][0]
            dist_m = distances_manhattan[i][0]
            dist_c = distances_cosine[i][0]

            conf_e = max(0.5, 1.0 / (1.0 + dist_e * 2))
            conf_m = max(0.5, 1.0 / (1.0 + dist_m * 1.5))
            conf_c = max(0.5, 1.0 / (1.0 + dist_c * 3))

            avg_confidence = (conf_e + conf_m + conf_c) / 3
            final_confidence = min(0.95, max(0.6, avg_confidence))

            # 获取最近邻的样本索引
            sample_idx_e = indices_euclidean[i][0]

            # 边界检查，确保样本索引不越界
            if sample_idx_e >= len(self.y_train_original):
                print(f"警告：样本索引{sample_idx_e}超出训练数据范围，训练样本数为{len(self.y_train_original)}")
                sample_idx_e = 0  # 使用第一个样本作为默认值

            # 获取该样本对应的类别索引
            # 使用原始标签而不是编码后的标签
            predicted_class = self.y_train_original[sample_idx_e]
            class_matches = np.where(self.label_encoder.classes_ == predicted_class)[0]

            if len(class_matches) == 0:
                print(f"警告：未找到类别'{predicted_class}'，可用类别: {self.label_encoder.classes_}")
                class_idx = 0  # 使用第一个类别作为默认值
            else:
                class_idx = class_matches[0]

            proba[i, class_idx] = final_confidence
            remaining = (1.0 - final_confidence) / (len(self.label_encoder.classes_) - 1)
            proba[i, :] += remaining
            proba[i, class_idx] = final_confidence  # 确保最终置信度正确设置

        return proba

    def get_feature_importance(self):
        if self.pca is None:
            return None

        # PCA成分的重要性
        pca_importance = np.abs(self.pca.components_).mean(axis=0)

        # 如果有成分名称，返回带名称的重要性
        if self.component_names is not None:
            importance_dict = {}
            for i, name in enumerate(self.component_names[:len(pca_importance)]):
                importance_dict[name] = pca_importance[i]
            return importance_dict

        return pca_importance

    def analyze_pca_feature_contributions(self, top_n=10, include_score_results=True):
        """
        分析PCA中贡献最大的特征，并返回不同主成分数的评分结果

        Parameters:
        -----------
        top_n : int, default=10
            返回贡献最大的前N个特征
        include_score_results : bool, default=True
            是否包含不同主成分数的评分结果（如果存在）

        Returns:
        --------
        dict : 包含特征贡献分析结果和主成分数评分结果的字典
        """
        if self.pca is None:
            return None

        # 获取PCA组件
        components = self.pca.components_

        # 获取原始特征数量（假设为40个代谢物特征）
        original_feature_count = 40

        # 只分析前original_feature_count个特征（原始代谢物特征）
        if len(components[0]) > original_feature_count:
            # 只取原始特征部分的PCA权重
            components_original = components[:, :original_feature_count]
            feature_contributions = np.abs(components_original).sum(axis=0)
        else:
            # 计算每个原始特征对所有主成分的总贡献
            feature_contributions = np.abs(components).sum(axis=0)

        # 归一化贡献值
        total_contribution = np.sum(feature_contributions)
        if total_contribution > 0:
            feature_contributions_normalized = feature_contributions / total_contribution
        else:
            feature_contributions_normalized = feature_contributions

        # 计算每个主成分的方差解释率
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # 找出贡献最大的前N个特征的索引
        top_feature_indices = np.argsort(feature_contributions)[::-1][:top_n]

        # 判断是否是多模型情形，并相应调整数据
        is_best_model_only = hasattr(self, 'best_model_idx') and hasattr(self, 'best_n_components')
        best_n_comp = self.best_n_components if is_best_model_only else None
        
        # 如果是多模型情形，仅保留最优主成分数的方差数据
        if is_best_model_only and best_n_comp is not None:
            explained_variance_ratio = explained_variance_ratio[:best_n_comp]
            cumulative_variance = np.cumsum(explained_variance_ratio)
            components_display = components[:best_n_comp]
        else:
            components_display = components

        # 准备结果
        results = {
            'top_features': [],
            'explained_variance': {
                'individual': explained_variance_ratio.tolist(),
                'cumulative': cumulative_variance.tolist()
            },
            'total_components': len(explained_variance_ratio),
            'total_features': original_feature_count,  # 只显示原始特征数量
            'extended_feature_count': len(components[0])  # 可选：显示扩展后的特征数量
        }

        # 如果存在PCA测试结果，添加评分信息用于可视化
        if include_score_results and hasattr(self, 'pca_test_results') and self.pca_test_results:
            # 判断是否是多模型或单模型
            if hasattr(self, 'best_model_idx') and hasattr(self, 'best_n_components'):
                # 多模型情形：仅提取最优主成分数的结果
                best_n_comp = self.best_n_components
                if best_n_comp in self.pca_test_results['all_results']:
                    result = self.pca_test_results['all_results'][best_n_comp]
                    results['pca_visualization_data'] = {
                        'n_components': [best_n_comp],
                        'f1_macro_scores': [result['avg_scores']['f1_macro']],
                        'model_reliability': [result.get('model_reliability', 0)],
                        'variance_explained': [result['variance_explained']],
                        'is_best_model': True
                    }
            else:
                # 单模型情形：提取所有主成分数的结果
                results['pca_visualization_data'] = {
                    'n_components': list(self.pca_test_results['all_results'].keys()),
                    'f1_macro_scores': [r['avg_scores']['f1_macro'] for r in self.pca_test_results['all_results'].values()],
                    'model_reliability': [r.get('model_reliability', 0) for r in self.pca_test_results['all_results'].values()],
                    'variance_explained': [r['variance_explained'] for r in self.pca_test_results['all_results'].values()],
                    'is_best_model': False
                }

        # 为每个重要特征准备详细信息
        for idx in top_feature_indices:
            # 计算该特征对每个主成分的贡献
            if len(components_display[0]) > original_feature_count:
                component_contributions = np.abs(components_display[:, :original_feature_count][:, idx])
            else:
                component_contributions = np.abs(components_display[:, idx])

            # 找出该特征影响最大的主成分
            max_component_idx = np.argmax(component_contributions)

            feature_info = {
                'index': int(idx),
                'contribution': float(feature_contributions[idx]),
                'contribution_percentage': float(feature_contributions_normalized[idx] * 100),
                'max_influence_component': int(max_component_idx),
                'max_influence_value': float(component_contributions[max_component_idx]),
                'component_influences': component_contributions.tolist()
            }

            # 如果有特征名称，添加名称
            if self.component_names and idx < len(self.component_names):
                feature_info['name'] = self.component_names[idx]
            else:
                feature_info['name'] = f'Compound{idx + 1}'  # 更直观的名称

            results['top_features'].append(feature_info)

        # 计算所有主成分的特征重要性分布
        results['component_feature_influence'] = []
        
        # 判断是否是多模型情形
        if hasattr(self, 'best_model_idx') and hasattr(self, 'best_n_components'):
            # 多模型情形：仅分析最优主成分数的主成分
            best_n_comp = self.best_n_components
            for i in range(best_n_comp):
                # 只分析主成分中原始特征的部分
                if len(components_display[i]) > original_feature_count:
                    # 只取原始特征部分的PCA权重
                    component_original = components_display[i, :original_feature_count]
                    abs_component = np.abs(component_original)
                else:
                    component_original = components_display[i]
                    abs_component = np.abs(component_original)

                # 只考虑原始特征中的前5个重要特征
                top_feat_indices = np.argsort(abs_component)[::-1][:5]

                component_info = {
                    'component_index': i,
                    'explained_variance_ratio': float(explained_variance_ratio[i]),
                    'top_features': []
                }

                for feat_idx in top_feat_indices:
                    feat_info = {
                        'feature_index': int(feat_idx),
                        'influence': float(abs_component[feat_idx])
                    }
                    if self.component_names and feat_idx < len(self.component_names):
                        feat_info['name'] = self.component_names[feat_idx]
                    else:
                        feat_info['name'] = f'Compound{feat_idx + 1}'  # 更直观的名称
                    component_info['top_features'].append(feat_info)

                results['component_feature_influence'].append(component_info)
        else:
            # 单模型情形：分析所有主成分
            for i in range(len(components)):  # 分析所有主成分
                # 仅分析主成分中原始特征的部分
                if len(components[i]) > original_feature_count:
                    # 只取原始特征部分的PCA权重
                    component_original = components[i, :original_feature_count]
                    abs_component = np.abs(component_original)
                else:
                    component_original = components[i]
                    abs_component = np.abs(component_original)

                # 只考虑原始特征中的前5个重要特征
                top_feat_indices = np.argsort(abs_component)[::-1][:5]

                component_info = {
                    'component_index': i,
                    'explained_variance_ratio': float(explained_variance_ratio[i]),
                    'top_features': []
                }

                for feat_idx in top_feat_indices:
                    feat_info = {
                        'feature_index': int(feat_idx),
                        'influence': float(abs_component[feat_idx])
                    }
                    if self.component_names and feat_idx < len(self.component_names):
                        feat_info['name'] = self.component_names[feat_idx]
                    else:
                        feat_info['name'] = f'Compound{feat_idx + 1}'  # 更直觃的名称
                    component_info['top_features'].append(feat_info)

                results['component_feature_influence'].append(component_info)

        return results

    def explain_prediction(self, X, sample_idx=0):
        X_processed = self.preprocess_data(X, fit=False)

        distances_e, indices_e = self.knn_euclidean.kneighbors([X_processed[sample_idx]])
        distances_m, indices_m = self.knn_manhattan.kneighbors([X_processed[sample_idx]])
        distances_c, indices_c = self.knn_cosine.kneighbors([X_processed[sample_idx]])

        explanation = {
            'euclidean_distance': distances_e[0][0],
            'manhattan_distance': distances_m[0][0],
            'cosine_distance': distances_c[0][0],
            'nearest_region_euclidean': self.label_encoder.inverse_transform([self.y_train[indices_e[0][0]]])[0],
            'nearest_region_manhattan': self.label_encoder.inverse_transform([self.y_train[indices_m[0][0]]])[0],
            'nearest_region_cosine': self.label_encoder.inverse_transform([self.y_train[indices_c[0][0]]])[0],
        }

        weight_e, weight_m, weight_c = self._calculate_adaptive_weights(
            distances_e[0][0], distances_m[0][0], distances_c[0][0]
        )
        explanation['distance_weights'] = {
            'euclidean': weight_e,
            'manhattan': weight_m,
            'cosine': weight_c
        }

        return explanation

    def generate_optimization_report(self, X, y, save_to_file=False, filename='optimization_report.txt'):
        """
        生成超参数优化报告

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        save_to_file : bool, default=False
            是否保存到文件
        filename : str, default='optimization_report.txt'
            报告文件名

        Returns:
        --------
        str : 优化报告内容
        """
        n_samples = len(X)
        n_unique_labels = len(np.unique(y))

        report = []
        report.append("=" * 60)
        report.append("药材地区分类器 - 超参数优化报告")
        report.append("=" * 60)

        # 数据集信息
        report.append(f"\n📊 数据集信息:")
        report.append(f"   样本数: {n_samples}")
        report.append(f"   类别数: {n_unique_labels}")
        report.append(f"   特征数: {X.shape[1]}")

        # 样本量分析
        if n_samples == n_unique_labels:
            report.append(f"\n⚠️  极小样本警告:")
            report.append(f"   每个类别只有1个样本，这是极小样本学习问题")
            report.append(f"   建议：考虑收集更多数据或使用其他机器学习方法")
        elif n_samples < 50:
            report.append(f"\n⚠️  小样本提示:")
            report.append(f"   样本数较少，使用留一法交叉验证(LOOCV)进行评估")

        # 交叉验证策略
        report.append(f"\n🔍 交叉验证策略:")
        if n_samples <= 20:
            report.append(f"   使用留一法交叉验证(LOOCV) - 适合极小样本")
        else:
            report.append(f"   使用5折分层交叉验证")

        # PCA组件数优化
        report.append("\n1. PCA组件数优化结果:")
        report.append("-" * 40)

        try:
            pca_results = self.optimize_pca_components(X, y)
            report.append(f"   最优PCA组件数: {pca_results['best_n_components']}")
            report.append(f"   最优分数: {pca_results['best_score']:.4f}")

            if pca_results['all_results']:
                    report.append("\n   详细结果:")
                    for n_comp, result in list(pca_results['all_results'].items())[:5]:  # 只显示前5个
                        report.append(
                            f"     组件数{n_comp}: F1-Macro={result['avg_scores']['f1_macro']:.4f}, Accuracy={result['avg_scores'].get('accuracy', 0):.4f}, 方差解释率={result['variance_explained']:.4f}")
        except Exception as e:
            report.append(f"   PCA优化失败: {e}")

        # K值优化
        report.append("\n2. K值优化结果:")
        report.append("-" * 40)

        try:
            k_results = self.optimize_k_values(X, y)
            for metric, result in k_results.items():
                report.append(f"\n   {metric}距离:")
                report.append(f"     最优K值: {result['best_k']}")
                report.append(f"     最优分数: {result['best_score']:.4f}")

                if result['all_results']:
                    report.append("     详细结果:")
                    for k, k_result in list(result['all_results'].items())[:3]:  # 只显示前3个
                        report.append(f"       K={k}: {k_result['mean_score']:.4f}±{k_result['std_score']:.4f}")
        except Exception as e:
            report.append(f"   K值优化失败: {e}")

        # 交叉验证结果
        report.append("\n3. 交叉验证评估结果:")
        report.append("-" * 40)

        try:
            cv_results = self.evaluate_with_cv(X, y)
            for model_name, metrics in cv_results.items():
                report.append(f"\n   {model_name}:")
                for metric_name, metric_result in metrics.items():
                    if metric_result['mean'] > 0:  # 只显示成功的指标
                        report.append(f"     {metric_name}: {metric_result['mean']:.4f}±{metric_result['std']:.4f}")
        except Exception as e:
            report.append(f"   交叉验证评估失败: {e}")

        # 建议和总结
        report.append("\n4. 优化建议和总结:")
        report.append("-" * 40)

        try:
            # 分析结果并给出建议
            if 'k_results' in locals():
                best_model = max(k_results.items(), key=lambda x: x[1]['best_score'])
                report.append(f"   🎯 推荐配置:")
                report.append(f"     距离度量: {best_model[0]}")
                report.append(f"     K值: {best_model[1]['best_k']}")

                if 'pca_results' in locals():
                    report.append(f"     PCA组件数: {pca_results['best_n_components']}")

            # 极小样本特殊建议
            if n_samples <= 20:
                report.append(f"\n   💡 极小样本建议:")
                report.append(f"     - 当前样本数({n_samples})较少，模型泛化能力有限")
                report.append(f"     - 建议收集更多训练数据提高模型稳定性")
                report.append(f"     - 可考虑使用数据增强技术扩充样本")
                report.append(f"     - 或尝试迁移学习、元学习等方法")

        except Exception as e:
            report.append(f"   建议生成失败: {e}")

        report.append(f"\n" + "=" * 60)
        report.append(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report_text = "\n".join(report)

        if save_to_file:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 优化报告已保存到: {filename}")

        return report_text


def load_data(train_file_path, test_file_path=None):
    """
    加载训练数据和测试数据

    Parameters:
    -----------
    train_file_path : str
        训练数据文件路径
    test_file_path : str, optional
        测试数据文件路径，如果为None则只加载训练数据

    Returns:
    --------
    tuple : (X_train, y_train, X_test, test_regions, test_samples) 或 (X_train, y_train)
    """
    # 加载训练数据
    train_data = pd.read_excel(train_file_path, sheet_name=0, header=None)

    # 提取特征数据（从第3行开始，第2列到第41列，共40个特征）
    X_train = train_data.iloc[2:, 1:41].values.astype(float)

    # 提取地区标签（第1列），并提取地区代码
    raw_labels = train_data.iloc[2:, 0].values
    y_train = []
    for label in raw_labels:
        if isinstance(label, str) and '-' in label:
            # 提取地区代码 (如 Y-FJ-1 -> Y-FJ)
            parts = label.split('-')
            if len(parts) >= 3:
                region_code = '-'.join(parts[:2])  # 取前两部分作为地区代码
            else:
                region_code = label
        else:
            region_code = 'Unknown'
        y_train.append(region_code)

    y_train = np.array(y_train)

    print(f"训练数据加载完成:")
    print(f"  样本数: {X_train.shape[0]}")
    print(f"  特征数: {X_train.shape[1]}")
    print(f"  地区数: {len(np.unique(y_train))}")
    print(f"  地区列表: {list(np.unique(y_train))}")

    # 如果没有提供测试数据文件，只返回训练数据
    if test_file_path is None:
        return X_train, y_train

    # 加载测试数据
    try:
        test_data = pd.read_excel(test_file_path, sheet_name=0, header=None)

        # 测试集样本ID（第1列，从第3行开始）
        test_sample_ids = test_data.iloc[2:, 0].dropna().tolist()

        # 测试集地区标签（第2列，从第3行开始）
        test_region_labels = test_data.iloc[2:, 1].dropna().tolist()

        # 从地区标签中提取地区代码
        test_regions = []
        for region_label in test_region_labels:
            if isinstance(region_label, str) and '-' in region_label:
                # 提取地区代码 (如 "Y-FJ aver" -> "Y-FJ")
                parts = region_label.strip().replace(' aver', '').split('-')
                if len(parts) >= 2:
                    region_code = '-'.join(parts[:2])
                else:
                    region_code = parts[0]
            else:
                region_code = 'Unknown'
            test_regions.append(region_code)

        # 测试集特征数据（第3列开始，从第3行开始，取40个特征以匹配训练集）
        X_test = test_data.iloc[2:, 2:42].values.astype(float)

        print(f"测试数据加载完成:")
        print(f"  测试样本数: {X_test.shape[0]}")
        print(f"  特征数: {X_test.shape[1]}")
        print(f"  地区数: {len(set(test_regions))}")
        print(f"  地区列表: {list(set(test_regions))}")
        print(f"  样本ID: {test_sample_ids}")

        return X_train, y_train, X_test, test_regions, test_sample_ids

    except Exception as e:
        print(f"测试数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return X_train, y_train, np.empty((0, 40)), [], []


def load_data_legacy(file_path):
    """
    兼容旧版本的数据加载函数（用于旧格式数据）
    """
    train_data = pd.read_excel(file_path, sheet_name=0)
    test_data = pd.read_excel(file_path, sheet_name=1)

    X_train = train_data.iloc[1:, 2:40].values.astype(float)
    y_train = train_data.iloc[1:, 1].values

    test_samples = []
    test_regions = []

    for col_idx in range(2, test_data.shape[1]):
        col_name = test_data.columns[col_idx]
        if 'aver' not in col_name:
            try:
                sample = pd.to_numeric(test_data.iloc[:38, col_idx], errors='coerce').values
                if not np.isnan(sample).all() and len(sample) == 38:
                    test_samples.append(sample)
                    region_parts = col_name.split('-')
                    if len(region_parts) >= 2:
                        region_name = f"{region_parts[0]}-{region_parts[1]}"
                    else:
                        region_name = col_name
                    test_regions.append(region_name)
            except:
                continue

    X_test = np.array(test_samples)

    return X_train, y_train, X_test, test_regions


def bootstrap_evaluation(classifier, X, y, n_bootstrap=100):
    n_samples = len(X)
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        classifier_boot = SmallSampleRegionClassifier()
        classifier_boot.train(X_boot, y_boot)

        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        if len(oob_indices) > 0:
            X_oob = X[oob_indices]
            y_oob = y[oob_indices]
            y_pred = classifier_boot.predict(X_oob)
            f1 = f1_score(y_oob, y_pred, average='macro')
            f1_scores.append(f1)

    return np.array(f1_scores)


def main():
    # 使用新的数据加载函数
    X_train, y_train = load_data('训练集.xlsx')

    print(f"训练数据: {X_train.shape}, 地区数: {len(np.unique(y_train))}")

    classifier = SmallSampleRegionClassifier()

    # 训练模型（包含超参数优化）
    print("\n=== 开始训练和优化 ===")
    classifier.train(X_train, y_train, optimize_hyperparams=True)

    # 生成优化报告
    print("\n=== 生成优化报告 ===")
    report = classifier.generate_optimization_report(X_train, y_train, save_to_file=True)
    print("\n" + report)

    # 使用新的交叉验证方法评估
    print("\n=== 标准交叉验证评估结果 ===")
    cv_results = classifier.evaluate_with_cv(X_train, y_train)
    for model_name, metrics in cv_results.items():
        print(f"\n{model_name}:")
        for metric_name, metric_result in metrics.items():
            print(f"  {metric_name}: {metric_result['mean']:.3f}±{metric_result['std']:.3f}")

    # LOOCV评估（保留原有功能）
    print("\n=== LOOCV评估结果 ===")
    loocv_results = classifier.evaluate_with_loocv(X_train, y_train)
    for model_name, metrics in loocv_results.items():
        print(f"{model_name}: F1={metrics['mean_f1']:.3f}±{metrics['std_f1']:.3f}")

    # Bootstrap评估（保留原有功能）
    print("\n=== Bootstrap评估结果 ===")
    bootstrap_scores = bootstrap_evaluation(classifier, X_train, y_train)
    print(f"Bootstrap F1: {bootstrap_scores.mean():.3f}±{bootstrap_scores.std():.3f}")
    print(f"95%置信区间: [{np.percentile(bootstrap_scores, 2.5):.3f}, {np.percentile(bootstrap_scores, 97.5):.3f}]")

    return classifier


if __name__ == "__main__":
    classifier = main()
