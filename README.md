# Lab2 – EEG Signal Classification (EEGNet & DeepConvNet)

本專案為 EEG 訊號分類作業，使用 **BCI Competition IV Dataset 2b**，以 PyTorch 實作並比較不同深度學習模型在 EEG 分類任務上的表現。主要模型為 **EEGNet**，並額外實作 **DeepConvNet** 作為比較。

---

## Environment

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- tqdm

建議使用 conda 建立環境：

```bash
conda create -n eeg python=3.8
conda activate eeg
pip install torch numpy matplotlib tqdm
```

---

## Project Structure

```text
.
├── main.py                 # Training / testing entry
├── dataloader.py           # Load BCI 2b dataset
├── models/
│   └── EEGNet.py           # EEGNet & DeepConvNet implementation
├── data/                   # BCI 2b dataset (.npz files)
├── weights/
│   ├── best.pt             # Best model checkpoint
│   └── ckpt.pt             # Training checkpoint (for resume)
├── figures/
│   ├── train_acc.png
│   ├── train_loss.png
│   └── test_acc.png
└── README.md
```

---

## Models

### EEGNet
- 使用 temporal convolution 擷取 EEG 時域特徵  
- 透過 depthwise convolution 進行通道間空間濾波  
- 採用 separable convolution 降低模型參數量  
- 嘗試不同 activation functions（ELU、ReLU、LeakyReLU）與 dropout 設定  

最佳設定下，EEGNet 在測試集上可達 **87.5% accuracy**。

### DeepConvNet (Bonus)
- 較深層的卷積神經網路架構  
- 使用多層 temporal convolution 與 pooling  
- 參數量較大，在資料量有限時較容易過擬合  
- 作為與 EEGNet 的模型架構比較  

---

## Training

### Train on GPU 0

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -num_epochs 150 -batch_size 64 -lr 0.01
```

### Resume Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -num_epochs 150 -batch_size 64 -lr 0.01 --resume
```

Training 過程中會：
- 每個 epoch 儲存 checkpoint（weights/ckpt.pt）
- 自動記錄最佳測試準確率模型（weights/best.pt）
- 繪製訓練與測試曲線並輸出至 figures/

---

## Visualization

訓練完成後會產生以下圖表：
- Training Accuracy Curve  
- Training Loss Curve  
- Testing Accuracy Curve  

---

## Experiments

本專案嘗試以下設定以提升模型效能：
- 不同 activation functions（ELU / ReLU / LeakyReLU）
- 調整 ELU 的 α 值
- 不同 dropout rates
- EEGNet 與 DeepConvNet 架構比較

實驗結果顯示，**LeakyReLU 在本資料集上能保留更多負值 EEG 訊號資訊，並取得最佳測試準確率**。

---

## Requirements Checklist

- [x] Implement EEGNet architecture  
- [x] Visualize training / testing accuracy and loss  
- [x] Experiment with different settings  
- [x] Upload code with README  
- [x] Bonus: Implement DeepConvNet  
