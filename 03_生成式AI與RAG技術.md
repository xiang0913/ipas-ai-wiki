# 03\_機器學習與模型評估指標

## 1. 單元定位與核心概念

* **白話理解**：如果說資料是食材，機器學習演算法就是「烹飪方式」，而評估指標就是「試吃員的評分表」。規劃師不需要自己推導微積分公式，但必須知道「什麼問題該挑哪種學習方式」，以及「模型考幾分才算真正能上線賺錢」。

* **iPAS 中級考點**：監督式與非監督式學習比較、過擬合（Overfitting）與欠擬合（Underfitting）、混淆矩陣（Confusion Matrix）四大指標取捨、ROC 曲線與 AUC 值。

## 2. 機器學習三大類型白話對照

根據訓練資料「有沒有標準答案（標籤 Label）」，主要分為三類：

1. **監督式學習（Supervised Learning）**：有標準答案的練習題。

   * **分類（Classification）**：預測離散的類別（如：這封信是垃圾郵件嗎？客戶會不會流失？）。常見演算法：邏輯斯迴歸（Logistic Regression）、決策樹（Decision Tree）、隨機森林（Random Forest）、XGBoost、支援向量機（SVM）。

   * **迴歸（Regression）**：預測連續的數值（如：明天的台積電股價是多少？下個月的電費是多少？）。常見演算法：線性迴歸（Linear Regression）。

2. **非監督式學習（Unsupervised Learning）**：沒有標準答案，讓 AI 自己找規律。

   * **分群（Clustering）**：物以類聚（如：將會員自動分成高消費、小資、潛力客群）。常見演算法：K-means。

   * **降維（Dimensionality Reduction）**：在保留核心資訊的前提下減少欄位數量，加速運算與視覺化。常見演算法：主成分分析（PCA）。

3. **強化學習（Reinforcement Learning）**：透過「獎勵（Reward）」與「懲罰（Penalty）」在環境中試錯學習（如：AlphaGo 下圍棋、自動駕駛決策）。

## 3. 模型訓練常見病症：過擬合 vs. 欠擬合

* **欠擬合（Underfitting）**：書沒念夠，模擬考與正式大考都考得很爛（高偏差 High Bias）。

  * **解法**：增加特徵數量、換更複雜的模型、延長訓練時間。

* **過擬合（Overfitting）**：死背考古題，模擬考（訓練集）考 100 分，正式大考（測試集）卻只考 50 分（高變異 High Variance）。

  * **解法（考前必背）**：增加訓練資料量、特徵篩選（減少雜訊特徵）、**正規化懲罰（L1 / L2 Regularization）**、早停法（Early Stopping），深度學習中則常用 **Dropout**（隨機丟棄部分神經元）。

## 4. 混淆矩陣（Confusion Matrix）與四大指標（最愛考！）

在二元分類問題中，預測結果與真實情況會組成四個象限：

* **TP（True Positive，真陽性）**：實際是壞人，模型也抓到是壞人（抓對了）。

* **TN（True Negative，真陰性）**：實際是好人，模型也判定是好人（放對了）。

* **FP（False Positive，偽陽性 / 型一錯誤）**：實際是好人，模型卻誤判成壞人（**誤殺**）。

* **FN（False Negative，偽陰性 / 型二錯誤）**：實際是壞人，模型卻放走當成好人（**漏抓**）。

### 四大核心指標公式與商業取捨：

1. **準確率（Accuracy）**：整體猜對的比例。

   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

   * *注意*：在「資料不平衡」時（如 99% 正常、1% 詐欺），Accuracy 會失真，不能單看它！

2. **精確率（Precision / 查準率）**：模型抓出來的人裡面，真正是壞人的比例（**寧可放過，不可誤殺**）。

   $$\text{Precision} = \frac{TP}{TP + FP}$$

   * *適用場景*：**垃圾郵件過濾**（怕把老闆的重要信件誤判成垃圾信丟掉）、**精準行銷推播**（怕亂槍打鳥惹怒客戶）。

3. **召回率（Recall / Sensitivity / 查全率）**：所有真正的壞人裡面，模型成功抓出多少比例（**寧可錯殺一百，不可放過一人**）。

   $$\text{Recall} = \frac{TP}{TP + FN}$$

   * *適用場景*：**癌症醫療篩檢**（漏抓一個病人會出人命，誤判頂多再做一次詳細檢查）、**金融詐欺/工廠重大瑕疵檢測**。

4. **F1-Score**：Precision 與 Recall 的調和平均數，用來在兩者之間取得平衡。

   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

## 5. ROC 曲線與 AUC 值（跨門檻評估神器）

* **ROC 曲線**：以「偽陽性率（FPR）」為 X 軸、「真陽性率（TPR，即 Recall）」為 Y 軸，畫出不同分類門檻（Threshold）下的表現曲線。

* **AUC（Area Under Curve）**：ROC 曲線底下的面積，數值介於 $0$ 到 $1$ 之間。

  * $\text{AUC} = 1.0$：完美分類器。

  * $\text{AUC} = 0.5$：跟「丟硬幣隨機猜」一樣爛（對角線）。

  * $\text{AUC} > 0.8$：通常代表模型具備良好的區分能力，且對資料不平衡較不敏感。

## 6. 快速複習檢核表（給簡報生成器的摘要提示）

* **初學者一句話總結**：評估模型不能只看總平均（Accuracy），要看商業代價——怕誤殺好人看 Precision，怕漏抓壞人看 Recall。

* **最常考的陷阱題**：癌症篩檢與詐欺偵測首重 **Recall**；垃圾信件過濾首重 **Precision**；死背考古題叫 **Overfitting**，可用 L1/L2 正規化或 Dropout 解決。