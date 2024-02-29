# CBSR
Collaborative Aware Bidirectional Semantic Reasoning for Video Question Answering

### Recommendation
We recommend the following operating environment:
- Python == 3.6.x
- Pytorch == 1.8.1 + cu102
- Torchvision == 0.9.1 +cu102
- And other packages

- ### Data Preparation
We use NExT-QA as an example to help get farmiliar with the code. You can download the pre-computed features [[Baidu Pan](https://pan.baidu.com/s/1Hm-BFv0epUpJhHJMPkyKsA), password: g8wv] and trained models.

- Please download this three datasets, and put them in the `./datasets/` folder.

- ### Demo 
Our model can be trained and verified by the following command:
```bash
python train.py
```
