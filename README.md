# DeePMOS

Authors: Xinyu Liang, Fredrik Cumlin <br/>
Emails: hopeliang990504@gmail.com, fcumlin@gmail.com

## Data Preparation
For VCC2018 data, it can be downloaded from [here](https://github.com/unilight/LDNet/tree/main/data). <br/>
For BVCC data, here's the [link](https://zenodo.org/record/6572573#.Yphw5y8RprQ).

## Training

Run ```train.py```, which depend on the following arguments:
* ```--num_epochs```: The number of epochs during training.
* ```--lamb_c```: Weight of consistency loss lambda_c.
* ```--lamb_t```: Weight of teacher model loss lambda_t.
* ```--log_valid```: The number of epochs between logging results on validation data.
* ```--log_epoch```: The number of epochs between logging training losses.
* ```--dataset```: Dataset. 'vcc2018' or 'bvcc'.
* ```--data_path```: Path to the dataset folder.
* ```--id_table```: Path to the id_table folder.
* ```--save_path```: Path for the best model to be saved.

## Acknowledgement

This repository inherits from the [unofficial MBNet implementation](https://github.com/sky1456723/Pytorch-MBNet), the [LDNet implementation](https://github.com/unilight/LDNet) and [LaMOSNet implementation](https://github.com/fcumlin/LaMOSNet).

