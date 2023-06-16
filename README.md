"# GeRec" 



For dataset preprocessing (e.g., Amazon-Electronics), we can run process_electronics.py in the dataset_preprocessing file as follows:

`python process_electronics.py --user_k 10 --item_k 10 --gpu_id 0`

Then, we can obtain "Electronics-Sample.train.inter", "Electronics-Sample.valid.inter", "Electronics-Sample.test.inter", "Electronics-Sample.user2index", and "Electronics-Sample.item2index" as shown in the dataset file.

For example, 

"Electronics-Sample.train.inter":





"Electronics-Sample.valid.inter":



"Electronics-Sample.test.inter":