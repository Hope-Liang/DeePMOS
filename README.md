# ProMOSNet

Author: Xinyu Liang, Fredrik Cumlin
Email: hopeliang990504@gmail.com, fcumlin@gmail.com

## Training

First, download the VCC2018 data, which can for example be done from [here](https://github.com/unilight/LDNet/tree/main/data). Then run ```train.py```, which depend on the following arguments:
* ```--num_epochs```: The number of epochs during training.
* ```--log_valid```: The number of epochs between logging results on validation data.
* ```--log_epoch```: The number of epochs between logging training losses.
* ```--data_path```: Path to the data folder.
* ```--id_table```: Path to the id_table folder.
* ```--save_path```: Path for the best model to be saved.

## Acknowledgement

This repository inherits from the [unofficial MBNet implementation](https://github.com/sky1456723/Pytorch-MBNet), the [LDNet implementation](https://github.com/unilight/LDNet) and [LaMOSNet implementation](https://github.com/fcumlin/LaMOSNet).

