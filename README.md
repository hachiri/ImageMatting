# ImageMatting 使用手册

Author: shu pengyu

Property: June 25, 2021 5:52 AM

> 以下所有运行步骤都需在项目根目录下运行，即 `cd ImageMatting`

# 1. 模型训练

文件 `train_fcn8s_atonce.py`，模型为预训练VGG16+从零训练FCN8s

快速使用说明查看 `./train_fcn8s_atonce.py -h`

参数说明

```bash
# [*]为必须值
[*]-g # 训练使用的 GPU ID
[*]--img # 训练图片数据的路径，需以ImageMatting/dataset/为基准路径的相对路径
[*}--mask # 训练掩码数据的路径，需以ImageMatting/dataset/为基准路径的相对路径
--resume # 模型训练中继文件路径
--max-iteration # 模型最大迭代次数，默认100000
--lr # 学习率，默认1e-10
--weight-decay # 权重衰减，默认0.0005
--momentum # 动量，默认0.99

# Example
./train_fcn8s_atonce.py -g 2 --img=images_data_crop --mask=images_mask
```

- 模型训练的参数文件保存至 `ImageMatting/logs/训练开始时间/`

    `config.yaml` 中保存输入的参数值

    `log.csv` 中保存模型训练中的验证和测试值

    ```bash
    ['epoch',
    'iteration',
    'train/loss',
    'train/acc',
    'train/acc_cls',
    'train/mean_iu',
    'train/fwavacc',
    'valid/loss',
    'valid/acc',
    'valid/acc_cls',
    'valid/mean_iu',
    'valid/fwavacc',
    'elapsed_time',]
    ```

    `visualization_viz/iter_number` 中保存模型在验证集上的运行结果

    `ImageMatting/logs/训练开始时间/model_best.pth.tar` 为模型文件

- 训练模型时，每4k次后使用模型运行验证集，如需改动，修改 `train_fcn8s_atonce.py` 文件

    ```python
    # train_fcn8s_atonce.py
    line 118: interval_validate=4000,
    ```

> 训练总epoch数目为：$\frac{最大迭代数}{训练数据数}$

- 模型默认为**2分类**，如需改动，修改 `train_fcn8s_atonce.py` 文件中的 `n_class=2` 为相应数值。

# 2. 数据集格式

图片格式默认格式为 `.jpg` ，如不为默认格式，需要修改 `ImageMatting/fcnnet/datasets/voc.py` 文件

```bash
# voc.py
line 86: img_file = osp.join(dataset_dir, img_dir_path, '%s.jpg' % did)
# 将上面 jpg 改为新数据级对应格式
```

数据集必须存放在 `ImageMatting/dataset` 文件夹下，所有图像和掩码数据应分别保存在相应的文件夹中， `train.txt` 和 `val.txt` 文件分别对应训练数据和测试数据，其中存有图片无后缀名的名称。

# 3. 模型验证

文件 `evaluate_model.py`

推荐将 `logs` 中训练得到的模型，改名后放入 `ImageMatting/models/` 下

参数说明

```bash
# [*]为必须值
[*]-g # 验证使用的 GPU ID，默认0
[*]--img # 训练图片数据的路径，需以ImageMatting/dataset/为基准路径的相对路径
[*}--mask # 训练掩码数据的路径，需以ImageMatting/dataset/为基准路径的相对路径
-m # 模型文件路径，默认 models/FCN8s.pth.tar

# Example
./evaluate_model.py -g 2 --img=images_data_crop --mask=images_mask -m models/FCN8s.pth.tar
```

运行后输出 Accuracy, Accuracy Class, Mean IU, FWAV Accuracy

验证集中的9张图片实际效果图保存为 `ImageMatting/models/viz_evaluate.png`

# 4. 模型预测

文件 `predict.py` 

参数说明

```bash
# [*]为必须值
[*]-g # 预测使用的 GPU ID
[*]--f # 预测使用的图像文件路径
-m # 模型文件路径，默认 models/FCN8s.pth.tar
-c # 背景颜色转换，B为蓝色，W为白色
-bg # 背景图片合成，背景图片路径

# Example
# example 转换成白色背景
./predict.py -g 2 -f 文件路径 -c W
# example 转换成指定背景
./predict.py -g 2 -f 文件路径 -bg 背景路径
```

如 -c 和 -bg 参数都为空，则以 `ImageMatting/bg/top9.jpg` 为背景图片进行背景合成

转换后的图片保存为 `./tmp/图片名_changed.jpg`，图片对应的掩码文件为`./tmp/图片名_mask.jpg`

图片传入时需做一次 600×800 的大小转换，需使用手机的 3:4 拍照模式，才可进行等比压缩