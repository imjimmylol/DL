
# 報告簡報模板

## 1. Implementation Details
### 1.1 Model Specification and Structure
- **UNet**  
  - Follows the original design by the authors, without adding any BatchNormalization layers.  
  - Unlike other common Unet variants, after calculating dimensions as described in the original paper, no padding is used in the downscaling layers.  
  - During upsampling, to preserve the original data at the encoder's edges, no cropping is performed; instead, zero-padding is applied to the upsampled data to ensure the dimensions match.

- **RES(34)UNet**  
  - Adopts the same encoder-decoder architecture as the original design.  
  - Utilizes the Residual block design from ResNet34, without using the bottleneck design.  
  - In the decoder, in addition to the standard Unet upsampling, it adds a sequence of convolution, ReLU, batch normalization, and CBAM.




### 1.2 Training Logic & Eval

- Training Flow Chart: 

    ```mermaid
    graph TD
        A["Start Training"] --> B["Load Datasets (Train & Validation)"]
        B --> C["Create DataLoaders"]
        C --> D{"Checkpoint Provided?"}
        D -- Yes --> E["Load Checkpoint"]
        D -- No --> F["Initialize Model"]
        E --> F
        F --> G["Define Loss & Optimizer"]
        G --> H["Train Loop"]
        H --> I["Validation Loop"]
        I --> J["Log & Save Checkpoint"]
        J --> K{"Is Validation Loss Best?"}
        K -- Yes --> L["Save Best Model"]
        K -- No --> M["Continue Training"]
        L --> M
        M --> N["End Training"]
    ```

- Training Settings:
    - `--resume` : resume the assigned checkpoint
    - `--runame` : To specify run name for tensorboard visulization comparison with different model


### 1.3 Importance Setting & reason 

- Loss function : `BCE Loss`
    - use this loss because the learning goal of binary segmentation is to assign each pixel in the image a label: foreground (1) or background (0).

- Model uutput : if size different then use `interpooling` to adjust size
    - for simplification and reducing the parameters 
- optimizer : `Adam`
- Learning rate : at default
- Learning rate schedular : only used in `RES_Unet` \
    since when inspeculating the training process using tensorboard, I found out a simulary loss of gradinet as the original RESnet paper, to solve this issue, I try to apply this learning rate schedular

---

## 2. Data Preprocessing (25%)

### Data Cleaning and Augmentation
    - With Transform : apply the following  
        - Normalization : `min max scalr`
        - Augmentation: `RandomHorizontalFlip`, `RandomRotation90`
    - No Transform : reamined the data unchanged

### Insights
    -  The RESUnet performs better when augmentation is applied, which is not the case for Unet
    
    -  The Unet 

---

## 3. Analyze the Experiment Results (25%)
### 超參數與訓練策略探索
- 比較不同超參數對結果的影響
- 模型配置與策略實驗比較（可用表格或圖）

### 實驗結果分析
- Training vs Validation 曲線（Loss / Accuracy）
- 可視化結果（如果有圖像任務）

### 分析與洞察
- 數據特性觀察與推論
- 有哪些重要結果影響了最終選擇

---

## 4. Execution Steps (0%)
### 執行指令與流程
- 訓練模型用的 command line 指令
- Inference 指令與流程

### 參數設定
- 實際使用的參數設定列表（表格或 code snippet）

---

## Q&A / References
- 有問題歡迎發問 🙋
- 參考文獻或外部資源列表（如有）
