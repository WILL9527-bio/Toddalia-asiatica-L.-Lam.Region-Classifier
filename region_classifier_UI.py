import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import threading
import pickle
from region_classifier import SmallSampleRegionClassifier


class RegionClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Toddalia asiatica （L.） Lam. Region Classifier")
        self.root.geometry("800x600")

        self.classifier = None
        self.train_data = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Toddalia asiatica （L.） Lam. Region Classifier", font=("Arial", 16, "bold")).grid(
            row=0, column=0, columnspan=3, pady=10)

        ttk.Button(main_frame, text="Load Training Data", command=self.load_training_data).grid(row=1, column=0, padx=5,
                                                                                                pady=5, sticky=tk.W)
        self.train_status = ttk.Label(main_frame, text="No training data loaded")
        self.train_status.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # 训练模型相关控件
        train_frame = ttk.Frame(main_frame)
        train_frame.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(train_frame, text="Train Model", command=self.train_model).grid(row=0, column=0, padx=5)
        
        # 移除了自动测试PCA主成分数的复选框
        
        # 模型状态标签
        self.model_status = ttk.Label(main_frame, text="Model not trained")
        self.model_status.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(main_frame, text="Predict New Samples:", font=("Arial", 12, "bold")).grid(row=4, column=0,
                                                                                            columnspan=3, pady=5,
                                                                                            sticky=tk.W)

        ttk.Button(main_frame, text="Select Excel File", command=self.load_prediction_data).grid(row=5, column=0,
                                                                                                 padx=5, pady=5,
                                                                                                 sticky=tk.W)
        self.file_label = ttk.Label(main_frame, text="No file selected")
        self.file_label.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Or manually input 40 component values (comma separated):").grid(row=6, column=0,
                                                                                                    columnspan=3,
                                                                                                    pady=5, sticky=tk.W)

        self.manual_input = tk.Text(main_frame, height=3, width=80)
        self.manual_input.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, padx=5, pady=10, sticky=tk.W)

        ttk.Button(button_frame, text="Start Prediction", command=self.predict).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Explain Prediction", command=self.explain_prediction).grid(row=0, column=1,
                                                                                                  padx=5)
        ttk.Button(button_frame, text="Analyze Feature Contributions", command=self.analyze_feature_contributions).grid(
            row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Save Model", command=self.save_model).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Load Model", command=self.load_model).grid(row=0, column=4, padx=5)

        self.result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        self.result_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.result_tree = ttk.Treeview(self.result_frame, columns=('sample', 'prediction', 'confidence', 'true_label', 'correctness'),
                                        show='headings', height=10)
        self.result_tree.heading('sample', text='Sample')
        self.result_tree.heading('prediction', text='Predicted Region')
        self.result_tree.heading('confidence', text='Confidence')
        self.result_tree.heading('true_label', text='True Label')
        self.result_tree.heading('correctness', text='Correct')
        self.result_tree.column('sample', width=100)
        self.result_tree.column('prediction', width=150)
        self.result_tree.column('confidence', width=100)
        self.result_tree.column('true_label', width=150)
        self.result_tree.column('correctness', width=80)

        scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        self.result_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

    def load_training_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Training Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            try:
                train_data = pd.read_excel(file_path, sheet_name=0, header=None)

                # 正确提取特征数据（从第3行开始，第2列到第41列，共40个特征）
                X_train = train_data.iloc[2:, 1:41].values.astype(float)

                # 正确提取地区标签（第1列），并去掉样本编号
                raw_labels = train_data.iloc[2:, 0].values
                y_train = []
                for label in raw_labels:
                    # 去掉最后的样本编号 (如 Y-FJ-1 -> Y-FJ)
                    parts = label.split('-')
                    if len(parts) >= 3:
                        region = '-'.join(parts[:-1])
                    else:
                        region = label
                    y_train.append(region)

                self.train_data = {
                    'X': X_train,
                    'y': np.array(y_train),
                    'file_path': file_path
                }

                # 显示正确的统计信息
                unique_regions = len(np.unique(y_train))
                self.train_status.config(
                    text=f"Loaded: {len(X_train)} samples, {X_train.shape[1]} features, {unique_regions} regions")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load training data: {str(e)}")

    def train_model(self):
        if self.train_data is None:
            messagebox.showwarning("Warning", "Please load training data first")
            return

        def train_thread():
            try:
                self.progress.start()
                self.classifier = SmallSampleRegionClassifier(enable_logging=True)
                
                # 训练模型，不再传入auto_test_pca参数
                self.classifier.train(self.train_data['X'], self.train_data['y'])
                
                self.progress.stop()
                self.model_status.config(text="Model training completed")
                
                # 显示训练成功信息
                if hasattr(self.classifier, 'qualified_models') and self.classifier.qualified_models:
                    qualified_models = self.classifier.qualified_models
                    
                    # 构建符合条件的模型信息
                    model_info = "Qualified Models:\n" + "="*40 + "\n\n"
                    for idx, model in enumerate(qualified_models, 1):
                        model_info += f"Model {idx}:\n"
                        model_info += f"  PCA Components: {model['n_components']}\n"
                        model_info += f"  F1-Macro Score: {model['f1_macro']:.3f}\n"
                        model_info += f"  LOOCV Accuracy: {model['loocv_accuracy']:.3f}\n"
                        model_info += f"  Variance Explained: {model['variance_explained']:.3f}\n\n"
                    
                    messagebox.showinfo("Training Results", model_info)
                else:
                    messagebox.showinfo("Success", "Model training completed")
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Model training failed: {str(e)}")

        threading.Thread(target=train_thread, daemon=True).start()

    def load_prediction_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Prediction Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            self.prediction_file = file_path
            self.file_label.config(text=f"Selected: {file_path.split('/')[-1]}")

    def predict(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        def predict_thread():
            try:
                self.progress.start()

                for item in self.result_tree.get_children():
                    self.result_tree.delete(item)

                manual_text = self.manual_input.get("1.0", tk.END).strip()

                if manual_text:
                    values = [float(x.strip()) for x in manual_text.split(',')]
                    if len(values) != 40:
                        raise ValueError(f"Need 40 values, but got {len(values)}")
                    X_pred = np.array([values])
                    sample_names = ["Manual Input"]
                    true_labels = [None]  # 手动输入没有真实标签

                elif hasattr(self, 'prediction_file'):
                    pred_data = pd.read_excel(self.prediction_file, header=None)
                    X_pred = []
                    sample_names = []
                    true_labels = []

                    # 检查数据格式：如果第3行开始有数据，说明是新格式（行为样本）
                    if pred_data.shape[0] > 2:
                        # 新格式：每行是一个样本，从第3行开始
                        for row_idx in range(2, pred_data.shape[0]):
                            try:
                                # 提取特征数据（第2列到第41列，共40个特征）
                                sample = pd.to_numeric(pred_data.iloc[row_idx, 2:42], errors='coerce').values
                                if not np.isnan(sample).all() and len(sample) == 40:
                                    X_pred.append(sample)
                                    # 使用样本ID作为名称
                                    sample_id = pred_data.iloc[row_idx, 0]
                                    region_label = pred_data.iloc[row_idx, 1]
                                    # 从region_label中移除"Pred"后缀
                                    true_region = str(region_label).replace(' Pred', '').strip()
                                    sample_names.append(f"Sample_{sample_id}_{region_label}")
                                    true_labels.append(true_region)  # 提取真实标签（去掉Pred）
                            except:
                                continue
                    else:
                        # 旧格式：每列是一个样本
                        for col_idx in range(1, pred_data.shape[1]):
                            col_name = pred_data.columns[col_idx] if pred_data.columns[col_idx] else f"Sample_{col_idx}"
                            try:
                                sample = pd.to_numeric(pred_data.iloc[:40, col_idx], errors='coerce').values
                                if not np.isnan(sample).all() and len(sample) == 40:
                                    X_pred.append(sample)
                                    sample_names.append(col_name)
                                    true_labels.append(None)  # 旧格式没有真实标签
                            except:
                                continue

                    if not X_pred:
                        raise ValueError("No valid prediction data found")
                    X_pred = np.array(X_pred)
                else:
                    raise ValueError("Please select a file or input data manually")

                predictions = self.classifier.predict(X_pred)
                probabilities = self.classifier.predict_proba(X_pred)

                # 保存预测结果用于解释
                self.last_predictions = predictions
                self.last_X_pred = X_pred
                self.true_labels = true_labels  # 保存真实标签
                self.sample_names = sample_names  # 保存样本名称

                # 计算预测正确性
                correct_count = 0
                incorrect_count = 0
                
                for i, (sample_name, pred, prob, true_label) in enumerate(zip(sample_names, predictions, probabilities, true_labels)):
                    confidence = prob.max()
                    
                    # 判断是否正确
                    if true_label is not None:
                        is_correct = (pred == true_label)
                        correctness_str = "✓ Correct" if is_correct else "✗ Wrong"
                        if is_correct:
                            correct_count += 1
                        else:
                            incorrect_count += 1
                    else:
                        correctness_str = "N/A"  # 没有真实标签
                    
                    self.result_tree.insert('', 'end', values=(sample_name, pred, f"{confidence:.3f}", 
                                                                str(true_label) if true_label else "N/A", 
                                                                correctness_str))
                
                # 显示统计信息
                self.progress.stop()
                if true_labels.count(None) < len(true_labels):  # 有真实标签的样本
                    messagebox.showinfo("Prediction Summary", 
                                       f"Total Samples: {len(predictions)}\n"
                                       f"Correct: {correct_count}\n"
                                       f"Incorrect: {incorrect_count}")

            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")

        threading.Thread(target=predict_thread, daemon=True).start()

    def explain_prediction(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        if not hasattr(self, 'last_predictions') or self.last_predictions is None:
            messagebox.showwarning("Warning", "Please make a prediction first")
            return

        try:
            # 创建一个新的窗口来显示所有样本的预测解释
            explain_window = tk.Toplevel(self.root)
            explain_window.title("Prediction Explanations for All Samples")
            explain_window.geometry("900x700")

            # 创建主框架
            main_frame = ttk.Frame(explain_window, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # 添加标题
            title_label = ttk.Label(main_frame, text="Prediction Explanations for All Samples", 
                                   font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 10))

            # 创建Treeview来整齐显示所有样本的预测解释
            columns = ("sample", "euclidean_dist", "manhattan_dist", "cosine_dist", 
                      "cosine_region",
                      "euclidean_weight", "manhattan_weight", "cosine_weight")
            
            tree_frame = ttk.Frame(main_frame)
            tree_frame.pack(fill=tk.BOTH, expand=True)

            # 创建Treeview控件
            tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
            
            # 设置列标题和宽度
            tree.heading("sample", text="Sample")
            tree.heading("euclidean_dist", text="Euclidean Distance")
            tree.heading("manhattan_dist", text="Manhattan Distance")
            tree.heading("cosine_dist", text="Cosine Distance")
            tree.heading("cosine_region", text="Cosine Region")
            tree.heading("euclidean_weight", text="Euclidean Weight")
            tree.heading("manhattan_weight", text="Manhattan Weight")
            tree.heading("cosine_weight", text="Cosine Weight")
            
            # 设置列宽
            tree.column("sample", width=120)
            tree.column("euclidean_dist", width=120)
            tree.column("manhattan_dist", width=120)
            tree.column("cosine_dist", width=120)
            tree.column("cosine_region", width=120)
            tree.column("euclidean_weight", width=120)
            tree.column("manhattan_weight", width=120)
            tree.column("cosine_weight", width=120)
            
            # 添加滚动条
            scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
            scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            
            # 布局Treeview和滚动条
            tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
            scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
            
            tree_frame.columnconfigure(0, weight=1)
            tree_frame.rowconfigure(0, weight=1)
            
            # 为所有样本生成预测解释并插入到Treeview中
            for i in range(len(self.last_X_pred)):
                explanation = self.classifier.explain_prediction(self.last_X_pred, i)
                
                # 获取样本名称
                sample_name = f"Sample {i+1}"
                if hasattr(self, 'sample_names') and i < len(self.sample_names):
                    sample_name = self.sample_names[i]

                # 插入数据行
                tree.insert("", tk.END, values=(
                    sample_name,
                    f"{explanation['euclidean_distance']:.4f}",
                    f"{explanation['manhattan_distance']:.4f}",
                    f"{explanation['cosine_distance']:.4f}",
                    explanation['nearest_region_cosine'],
                    f"{explanation['distance_weights']['euclidean']:.3f}",
                    f"{explanation['distance_weights']['manhattan']:.3f}",
                    f"{explanation['distance_weights']['cosine']:.3f}"
                ))
            
            # 添加说明文本
            note_frame = ttk.Frame(main_frame)
            note_frame.pack(fill=tk.X, pady=(10, 0))
            
            note_label = ttk.Label(note_frame, 
                                  text="The prediction is based on the weighted combination of these distance metrics.",
                                  font=("Arial", 10))
            note_label.pack()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to explain predictions: {str(e)}")

    def save_model(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "No model to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(self.classifier, f)
                messagebox.showinfo("Success", "Model saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.model_status.config(text="Model loaded successfully")
                messagebox.showinfo("Success", "Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def analyze_feature_contributions(self):
        """
        分析并显示PCA特征贡献
        """
        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        def analyze_thread():
            try:
                self.progress.start()

                # 调用后端的特征贡献分析方法，设置top_n=40以显示全部原始特征
                results = self.classifier.analyze_pca_feature_contributions(top_n=40, include_score_results=True)

                if results is None:
                    messagebox.showerror("Error", "PCA analysis not available. Make sure model is properly trained.")
                    return

                # 创建结果窗口
                result_window = tk.Toplevel(self.root)
                result_window.title("PCA Feature Contribution Analysis")
                result_window.geometry("800x600")

                # 创建主框架
                main_frame = ttk.Frame(result_window, padding="10")
                main_frame.pack(fill=tk.BOTH, expand=True)

                # 创建笔记本（选项卡）控件
                notebook = ttk.Notebook(main_frame)
                notebook.pack(fill=tk.BOTH, expand=True, pady=5)

                # ===== PCA主成分数评分选项卡 =====
                # 检查是否有pca_test_results
                if hasattr(self.classifier, 'pca_test_results'):
                    pca_scores_frame = ttk.Frame(notebook)
                    notebook.add(pca_scores_frame, text="PCA Components Scores")
                    
                    # 创建表格显示不同主成分数的评分
                    columns = ("n_components", "f1_macro", "loocv_accuracy", "variance_explained", "euclidean_score", "manhattan_score", "cosine_score")
                    scores_tree = ttk.Treeview(pca_scores_frame, columns=columns, show="headings", height=15)
                    
                    # 设置列标题
                    scores_tree.heading("n_components", text="PCA Components")
                    scores_tree.heading("f1_macro", text="F1-Macro Score")
                    scores_tree.heading("loocv_accuracy", text="LOOCV Accuracy")
                    scores_tree.heading("variance_explained", text="Variance Explained")
                    scores_tree.heading("euclidean_score", text="Euclidean Score")
                    scores_tree.heading("manhattan_score", text="Manhattan Score")
                    scores_tree.heading("cosine_score", text="Cosine Score")
                    
                    # 设置列宽
                    scores_tree.column("n_components", width=120)
                    scores_tree.column("f1_macro", width=120)
                    scores_tree.column("loocv_accuracy", width=120)
                    scores_tree.column("variance_explained", width=150)
                    scores_tree.column("euclidean_score", width=120)
                    scores_tree.column("manhattan_score", width=120)
                    scores_tree.column("cosine_score", width=120)
                    
                    # 添加滚动条
                    scores_scrollbar = ttk.Scrollbar(pca_scores_frame, orient=tk.VERTICAL, command=scores_tree.yview)
                    scores_tree.configure(yscroll=scores_scrollbar.set)
                    
                    # 填充数据
                    pca_results = self.classifier.pca_test_results
                    best_n = pca_results['best_n_components']
                    
                    # 使用sorted_results来保证按主成分数排序
                    if 'sorted_results' in pca_results:
                        sorted_results = pca_results['sorted_results']
                    else:
                        # 备用方案：从all_results中排序
                        sorted_results = sorted(pca_results['all_results'].items(), key=lambda x: x[0])
                    
                    for n_comp, result in sorted_results:
                        # 标记最佳主成分数
                        tag = "best" if n_comp == best_n else ""
                        # 获取各距离度量的f1_macro评分
                        euclidean_f1 = result['individual_scores'].get('euclidean', {}).get('f1_macro', {}).get('mean', 0)
                        manhattan_f1 = result['individual_scores'].get('manhattan', {}).get('f1_macro', {}).get('mean', 0)
                        cosine_f1 = result['individual_scores'].get('cosine', {}).get('f1_macro', {}).get('mean', 0)
                        
                        # 获取平均f1_macro和accuracy分数
                        f1_macro = result['avg_scores'].get('f1_macro', 0)
                        accuracy = result['avg_scores'].get('accuracy', 0)
                        
                        # 获取LOOCV准确率（如果存在）
                        loocv_accuracy = result.get('loocv_accuracy', 0)
                        
                        scores_tree.insert('', tk.END, values=(
                            n_comp,
                            f"{f1_macro:.3f}",
                            f"{loocv_accuracy:.3f}",
                            f"{result['variance_explained']:.3f}",
                            f"{euclidean_f1:.3f}",
                            f"{manhattan_f1:.3f}",
                            f"{cosine_f1:.3f}"
                        ), tags=(tag,))
                    
                    # 设置最佳结果的样式
                    scores_tree.tag_configure("best", background="#a8d1ff")
                    
                    # 布局
                    scores_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                    scores_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
                    pca_scores_frame.columnconfigure(0, weight=1)
                    pca_scores_frame.rowconfigure(0, weight=1)
                    
                    # 不再显示最佳PCA组件数和评分信息
                
                # ===== 主成分信息选项卡 =====
                components_frame = ttk.Frame(notebook)
                notebook.add(components_frame, text="Principal Components Analysis")

                # 创建文本框显示主成分信息
                text_widget = tk.Text(components_frame, wrap=tk.WORD, padx=10, pady=10)
                text_widget.pack(fill=tk.BOTH, expand=True)

                # 添加主成分信息
                text_widget.insert(tk.END, "Principal Component Analysis Results:\n\n")
                text_widget.insert(tk.END, f"Total Components: {results['total_components']}\n")
                text_widget.insert(tk.END, f"Original Features (Metabolites): {results['total_features']}\n")

                # 如果有扩展特征数量信息，显示它
                if 'extended_feature_count' in results:
                    text_widget.insert(tk.END,
                                       f"Extended Features (After Feature Engineering): {results['extended_feature_count']}\n")
                    text_widget.insert(tk.END, "(Note: Analysis focuses only on original Compound features)\n\n")
                else:
                    text_widget.insert(tk.END, "\n")

                text_widget.insert(tk.END, "Explained Variance:\n")
                text_widget.insert(tk.END, "-----------------\n")
                for i, (ind_var, cum_var) in enumerate(
                        zip(results['explained_variance']['individual'], results['explained_variance']['cumulative'])):
                    text_widget.insert(tk.END, f"PC {i}: {ind_var * 100:.2f}% (Cumulative: {cum_var * 100:.2f}%)\n")

                # 禁用文本编辑
                text_widget.config(state=tk.DISABLED)

                # ===== 主成分特征影响选项卡 =====
                influence_frame = ttk.Frame(notebook)
                notebook.add(influence_frame, text="Component Feature Influence")

                # 创建文本框显示主成分特征影响
                influence_text = tk.Text(influence_frame, wrap=tk.WORD, padx=10, pady=10)
                influence_text.pack(fill=tk.BOTH, expand=True)

                # 添加影响信息 - 显示全部主成分
                if results['component_feature_influence']:
                    for comp_info in results['component_feature_influence']:
                        influence_text.insert(tk.END,
                                              f"Principal Component {comp_info['component_index']} (Explained Variance: {comp_info['explained_variance_ratio'] * 100:.2f}%)\n")
                        influence_text.insert(tk.END, "Top Influencing Features:\n")
                        for feat_info in comp_info['top_features']:
                            influence_text.insert(tk.END,
                                                  f"  - {feat_info['name'].replace('代谢物_', 'Compound')} (Index {feat_info['feature_index']}): Influence = {feat_info['influence']:.4f}\n")
                        influence_text.insert(tk.END, "\n")
                else:
                    influence_text.insert(tk.END, "No component feature influence data available.\n")

                # 禁用文本编辑
                influence_text.config(state=tk.DISABLED)

                # ===== 全部特征贡献选项卡 =====
                all_features_frame = ttk.Frame(notebook)
                notebook.add(all_features_frame, text="Compounds Feature Contribution")

                # 创建表格显示全部特征贡献
                columns = ("rank", "name", "index", "percentage", "max_component")
                tree = ttk.Treeview(all_features_frame, columns=columns, show="headings", height=20)

                # 设置列标题
                tree.heading("rank", text="Rank")
                tree.heading("name", text="Features Name")
                tree.heading("index", text="Index")
                tree.heading("percentage", text="Contribution (%)")
                tree.heading("max_component", text="Max Influence Component")

                # 设置列宽
                tree.column("rank", width=50)
                tree.column("name", width=200)
                tree.column("index", width=80)
                tree.column("percentage", width=120)
                tree.column("max_component", width=150)

                # 添加滚动条
                scrollbar = ttk.Scrollbar(all_features_frame, orient=tk.VERTICAL, command=tree.yview)
                tree.configure(yscroll=scrollbar.set)

                # 填充数据 - 显示全部特征
                for i, feature in enumerate(results['top_features']):
                    tree.insert('', tk.END, values=(
                        i + 1,
                        feature['name'].replace('代谢物_', 'Compound'),
                        feature['index'],
                        f"{feature['contribution_percentage']:.2f}%",
                        f"PC {feature['max_influence_component']} ({feature['max_influence_value']:.4f})"
                    ))

                # 布局
                tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
                all_features_frame.columnconfigure(0, weight=1)
                all_features_frame.rowconfigure(0, weight=1)

                # ===== 总结选项卡 =====
                summary_frame = ttk.Frame(notebook)
                notebook.add(summary_frame, text="Summary")

                # 创建总结文本
                summary_text = tk.Text(summary_frame, wrap=tk.WORD, padx=10, pady=10)
                summary_text.pack(fill=tk.BOTH, expand=True)

                # 生成总结
                summary_text.insert(tk.END, "PCA Feature Contribution Analysis Summary\n")
                summary_text.insert(tk.END, "============================================================\n\n")

                # 最重要的特征
                if results['top_features']:
                    top_feature = results['top_features'][0]
                    summary_text.insert(tk.END, f"Most Important Feature:\n")
                    summary_text.insert(tk.END, f"  - Name: {top_feature['name']}\n")
                    summary_text.insert(tk.END, f"  - Index: {top_feature['index']}\n")
                    summary_text.insert(tk.END,
                                        f"  - Contribution: {top_feature['contribution_percentage']:.2f}% of total variance\n")
                    summary_text.insert(tk.END,
                                        f"  - Max Influence on: Principal Component {top_feature['max_influence_component']}\n\n")

                # 主成分方差解释
                first_pc_var = results['explained_variance']['individual'][0] * 100 if results['explained_variance'][
                    'individual'] else 0
                cumulative_var_3pcs = results['explained_variance']['cumulative'][2] * 100 if len(
                    results['explained_variance']['cumulative']) >= 3 else 0

                summary_text.insert(tk.END, "Variance Explanation:\n")
                summary_text.insert(tk.END, f"  - First Principal Component: {first_pc_var:.2f}% of variance\n")
                summary_text.insert(tk.END, f"  - First 3 Components: {cumulative_var_3pcs:.2f}% of variance\n\n")

                # 添加关于特征扩展的说明
                if 'extended_feature_count' in results:
                    summary_text.insert(tk.END, "Feature Engineering Note:\n")
                    summary_text.insert(tk.END, "------------------------\n")
                    summary_text.insert(tk.END, "During model training, the original 40 Compound features are extended with:\n")
                    summary_text.insert(tk.END, "1. Statistical features (max, min, median, MAD)\n")
                    summary_text.insert(tk.END, "2. Ratio features between Compounds\n")
                    summary_text.insert(tk.END,
                                        f"This results in {results['extended_feature_count']} features used for PCA.\n")
                
                # 添加PCA自动测试说明
                if hasattr(self.classifier, 'best_n_components') and hasattr(self.classifier, 'best_model_confidence'):
                    summary_text.insert(tk.END,
                                        "\nPCA Component Optimization:\n")
                    summary_text.insert(tk.END,
                                        "---------------------------\n")
                    summary_text.insert(tk.END,
                                        f"Best Model PCA Components: {self.classifier.best_n_components}\n")
                    summary_text.insert(tk.END,
                                        f"Average Confidence: {self.classifier.best_model_confidence:.3f}\n")
                elif hasattr(self.classifier, 'pca_test_results'):
                    summary_text.insert(tk.END,
                                        "\nPCA Component Optimization:\n")
                    summary_text.insert(tk.END,
                                        "---------------------------\n")
                    summary_text.insert(tk.END,
                                        f"The optimal number of components ({self.classifier.pca_test_results['best_n_components']}) was selected based on maximum F1-Macro score.\n")
                    summary_text.insert(tk.END,
                                        f"The corresponding score was {self.classifier.pca_test_results['best_score']:.3f}.\n")
                else:
                    summary_text.insert(tk.END,
                                        "\nAfter feature expansion, PCA dimensionality reduction is automatically adjusted to maintain a reasonable sample-to-feature ratio.\n")
                    
                summary_text.insert(tk.END,
                                    "\nThis analysis focuses primarily on the original 40 compound features, aiming to clearly identify which compounds are of greatest importance.\n\n")

                # 解释
                summary_text.insert(tk.END, "Interpretation:\n")
                summary_text.insert(tk.END, "----------------\n")
                summary_text.insert(tk.END,
                                    "This analysis shows which original features contribute most to the PCA transformation.\n")
                summary_text.insert(tk.END,
                                    "Features with higher contribution percentages have more influence on the\n")
                summary_text.insert(tk.END,
                                    "principal components and thus play a more important role in distinguishing\n")
                summary_text.insert(tk.END, "between different plant origins.\n\n")

                summary_text.insert(tk.END, "These important features could be:\n")
                summary_text.insert(tk.END, "1. Key chemical components that vary significantly between regions\n")
                summary_text.insert(tk.END, "2. Environmental factors that affect plant growth in different areas\n")
                summary_text.insert(tk.END, "3. Genetic markers that are region-specific\n")

                # 禁用文本编辑
                summary_text.config(state=tk.DISABLED)

                self.progress.stop()

            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Failed to analyze feature contributions: {str(e)}")

        # 在单独的线程中运行分析
        threading.Thread(target=analyze_thread, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = RegionClassifierUI(root)
    root.mainloop()
