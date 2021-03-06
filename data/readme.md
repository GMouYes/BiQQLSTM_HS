In this folder, we provide a sampled dataset from different sources.
Things to note:
 - In actual experiments, we conducted a 10-fold-cv. You may wish to also do 5cv or 10cv so as to avoid cherry picking. Depending on your actual need, you can choose to split in stratified style with respect to either platforms or classes.
 - If you just wish to make it work, try split the data into 80/10/10 in train/valid/test.
 - None of these data is collected by us. All credits to prior works. please refer to the paper where we provided our full citations.

 This is just a sampled subset of all datasets for various reasons: 
 1. tweets on Twitter will eventually disappear due to deletion/suspension. 
 2. Other postings may be too short/long. 
 3. Some postings are not English while we focused on English postings, etc. 
 4. We have to balance the number of positive and negative instances.
 5. We tried our best in recovering these posts and applied our own filtering criteria. One may check our paper for detailed processes.

Do not use or share these data for any commercial usage.

An overlook to the recovered data: <br>
![Dataset](https://github.com/GMouYes/BiQQLSTM_HS/blob/main/data/dataset.jpg)