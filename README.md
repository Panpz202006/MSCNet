1.Abstract:

    This paper proposes the Context Enhanced Network (CENet). In the encoder stage, CENet retains low-level detail information, providing essential detail information for the final prediction results. In the decoder stage, MSC extracts multi-scale information through multiple dilation rates and different sizes to focus on both local and global information. Dynamic edge enhancement (EE) is achieved to reduce the impact of blurred edges on the final results. CF is used to fuse high-level semantic information and low-level detail information. To evaluate the effectiveness of CENet, we conducted experiments on the ISIC2018, ISIC2017, and PH2 dataset.

2.Experiment:

    1.The introduction of models:
    (1)CNN_* are CNN-based models. Mamba_* are Mamba-based models.
    2.The metrics:																			
![image](https://github.com/user-attachments/assets/b5e7274b-f82c-4214-a686-3bb599a95677)
![image](https://github.com/user-attachments/assets/ad398870-3873-44c5-bc23-8f01fd60aae5)						
![image](https://github.com/user-attachments/assets/2954ddc9-e08f-48c1-9286-acb8718c9fad)


3.Prepare datasets：

     (1)Push datastes to the data folder:
         Push ISIC2017 datasets to data/ISIC2017 folder, and so on.
     (2)The datasets link are as follows:
         ISIC2017: https://challenge.isic-archive.com/data/#2017 
         ISIC2018: https://challenge.isic-archive.com/data/#2018
         PH2: https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1

4.Run codes:

     (1)Enter the train fucntion, and then run the train.py.
     (2)python train.py --param v. For example,
        python train.py --model Mamba --imagesize 256
        This command was that set Mamba as backbone and set image resolution as 256*256. Then, concrete command need to be viewed at train.py.

5.Test:

     (1) test.py was used for testing the performance of model.
