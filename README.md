# RWMesh

<p align="center">
  <img src="https://github.com/k-oellers/RWMesh/blob/main/CatHKS.png" width="200" height="200">
</p>

PyTorch implementation of learning on meshes using random walks, which is inspired by [MeshWalker](https://arxiv.org/abs/2006.05353). The model performs random walks on the edges of the mesh to generate a sequence of vertices. The relative positions of the vertices are used by the model to solve the classification and segmentation tasks. In contrast to the MeshWalker, in addition to the positions of the vertices, the values of a spetral descriptor at the corresponding vertices are used, which improves the accuracy of the model for deformable meshes. Furthermore, it has the advantage that the model is better protected against [adversarial attacks](https://arxiv.org/pdf/2202.07453.pdf) than other approaches by using spectral descriptors like the heat kernel signature. In addition, self-supervised training with [Barlow Twins](https://arxiv.org/abs/2103.03230) is possible, where two walks are performed on the same mesh and mapped close in feature space.

Original tensorflow implementation of the MeshWalker can be found [here](https://github.com/AlonLahav/MeshWalker). The processing of segmentation data is based on the [MeshCNN](https://github.com/ranahanocka/MeshCNN).

The package includes

- The attention-based encoder along with the reimplemented RNN-based encoder from MeshWalker
- Optional heat kernel signature or wave kernel signature as input for the encoder
- Self-supervised learning using a modified Barlow Twins implementation
- Adversarial attack on mesh-based networks.
- Unittests

## Installation

```
pip install -r requirements.txt
```

## Datasets

The model was evaluated with the following datasets:

- Non-deformable mesh classification dataset [Engraved Cubes](https://arxiv.org/pdf/1809.05910.pdf). ([Download link](https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz))
- Deformable mesh segmentation dataset [Coseg](https://modelnet.cs.princeton.edu/). ([Download link](https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz))
- Deformable mesh classification dataset [Shrec11](http://reuter.mit.edu/blue/papers/shrec11/shrec11.pdf). ([Download link](https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz))

## Usage

### Supervised learning

Model can be trained by running **train.py**.

```
# train transformer model supervised using shrec11 dataset with default configs
python train.py --dataset datasets/shrec16 --default_config
# uses cubes dataset to train the rnn encoder for 1000 epochs with 2 walks, sequence length 100 and without spectral signature
python train.py --dataset datasets/Cubes --model rnn --epochs 1000 --walks 2 --sequence_length 100 --model_size 512 --eig_basis 0
```

To get help with the input parameters, train.py can also be called with -h:

```
# get description for all input parameters
python train.py -h
```

### Self-supervised learning

It can also be trained self-supervised (with online finetuning)

```
# train model self-supervised using shrec11 dataset with default configs
python train.py --dataset datasets/shrec16 --default_config --self_supervised barlow --name test
```

Using the name, a pretrained network can be finetuned with a specific amount of labels

```
# finetune pretrained model using 50% of the labels with default configs
python train.py --dataset datasets/shrec16 --default_config --self_supervised barlow --name test --pretrained --finetune 0.5 --walks 2
```

### Testing

In order to test a trained model, the script **run.py** can be executed.
```
# tests the supervised trained model called final
python run.py --name final --dataset datasets/shrec16 --default_config
# tests the self-supervised trained model called final_ssl
python run.py --name final_ssl --dataset datasets/shrec16 --default_config --self_supervised barlow
```

### Adversarial attack

To perform an adversarial attack, a network must first be trained. Then, the trained network can be used to train the imitating network. The learning rate for the attack can be very crucial. Immediately afterwards, the attack is automatically performed and the modified meshes are saved under output/visualization/attack.

```
# train model
python train.py --name final --dataset datasets/shrec16 --default_config
# attack model
python attack.py --name final --dataset datasets/shrec16 --default_config  --learning_rate 0.1
```

If the imitating network has already been trained, it can also be loaded directly and the attack performed by using the pretrained argument.

```
# tests the supervised trained model called final
python attack.py --name final --dataset datasets/shrec16 --pretrained
```

### Visualization

The attention of a trained model can visualized using visualize.py, which is based on the attention rollout proposed in [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928) by Samira Abnar. The script takes a number of randomly selected meshes of the test dataset and writes a .ply with the vertex colors and a .obj file with the normalized mesh. The colors of the vertices are related to the corresponding attention weight. The meshes are saved under in the folder output/visualization/*dataset_name*/*model*/*learning_objective*/*sequence_length*/*name*
```
# loads the trained model called final in the directory store and visualizes the attention for 15 randomly selected meshes
python visualize.py --name final --dataset datasets/Cubes --default_configs --amount 15
```
You can also specify which mesh should be visualized using a regular expression with
```
# only visualizes one mesh with the matching regular expression T74 (cat) in the filename
python visualize.py --name final --dataset datasets/shrec16 --default_configs --amount 1 --regex T74
```
