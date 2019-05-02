# ETNA : Embedding Transformation Network with Attention

This is our Pytorch implementation for the paper:

Raehyun Kim and Hyunjae Kim (2019). *[Predicting Multiple Demographic Attributes with Task Specific Embedding Transformation and Attention Network.](https://arxiv.org/abs/1903.10144)* In Proceedings of Siam  Data Mining (SDM'19)

Please cite our paper if you use the code or datasets.

The code is tested under a Linux desktop (w/ TiTan X - Pascal) with Pytorch 1.0.0. and Python 3.


## MVP (Multi-Vendor loyalty Program) Dataset
We provide dataset for demographic prediction. You can find our raw dataset in (`data/raw`).

MVP dataset consists of three files. `[Company_info, User_info, Purchase_history]`

* `company_info.csv` : Company's industrial categories are included in company info.
* `user_info.csv` : User's demographic information (processed as class).
* `purchase_history.json` : Each user's purchasing history.

## Model Training
We have two type of task settings. (New user and partial prediction)

And user should specify observation ratio for  partial prediction task.

To train our model on `partial task with 50% of observation ratio` (with default hyper-parameters): 

```
python main.py --model_type ETNA --task_type partial50 
```

