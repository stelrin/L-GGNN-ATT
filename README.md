*The current implementation is still under development and is only intended for testing and experimentation purposes.*
# L-GGNN-ATT
This repository contains the implementation of the L-GGNN-ATT model, which is a variant of the SR-GNN model [1] that uses the Relevant Order Graph Formulation as its connection scheme.
![L-GGNN-ATT Architecture](https://i.ibb.co/rF6TBx1/L-GGNN-ATT-Architecture.png)
## Articles
[1]	Wu, S., Tang, Y., Zhu, Y., Wang, L., Xie, X., & Tan, T. (2019, July). [Session-based recommendation with graph neural networks](https://arxiv.org/abs/1811.00855). In _Proceedings of the AAAI conference on artificial intelligence_ (Vol. 33, No. 01, pp. 346-353).
# Usage
To train the model, create a directory with the name of your dataset in the  `datasets` folder and include two files `train.csv` and `test.csv` in it. These files should contain the training sessions and testing sessions respectively. A session is a sequence of item IDs separated by commas (e.g `5, 12, 52, 1004, 6, 52, 5`). Then, run `main.py` with `DATASET_NAME` set to the name of your dataset directory and the hyperparameters adjusted to your needs.
