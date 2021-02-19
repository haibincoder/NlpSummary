自己实现的TextCNN、LSTM、Transformers文本分类，在THUCNews数据集1个epoch准确率约90%
## 运行说明：
1. 修改train.py模型名
```python
model_name = 'Transformer'  # 可选TextCNN、TextLSTM
```
2. 直接运行python train.py

其他：修改train.py debug=True 可以直接debug模型