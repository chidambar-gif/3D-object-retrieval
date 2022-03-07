# Setup
## Install Related Packages
This code is developed in Python 3.7 and pytorch1.10.2+cu102. You can install the required packages as follows.
``` bash 
conda create -n shrec2022 python=3.7
conda activate shrec2022
pip install -r requirements.txt
```

## Configure Path
The datasets are placed under the "data" folder in the root directory. The code will create a new folder (name depends on the current time) to restore the checkpoint files under "cache/ckpts" folder for each run.
``` bash
├── cache
│   └── ckpts
│        └── OS-MN40_(current_time)
│           ├── cdist.txt
│           ├── ckpt.meta
│           └── ckpt.pth                   
│           
└── data
    ├── OS-MN40/
    └── OS-MN40-Miss/
```

# Train and Validation
Run "train.py". By default, 80% data in the train folder is used for training and the rest is used for validation.
``` bash
python train.py
```

## Testing/Generate Distance Matrix
Modify the data_root and ckpt_path in "line 17-18 in get_mat.py". Then run:
``` bash
python get_mat.py
```
The generated cdist.txt can be found in the same folder of the specified checkpoints. 

