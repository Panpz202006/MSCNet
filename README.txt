1.数据集准备：
	（1）将ISIC2017、ISIC2018、PH2数据集放入文件夹data目录下
	（2）进入main函数，修改以下两条代码中的路径中的数据集名称，即可运行指定数据集：
		    train_datasets_path=os.path.join(cwd_path,'data','ISIC2018','train')
    		    val_datasets_path=os.path.join(cwd_path,'data','ISIC2018','test')
        （3）数据集的路径结构：
		data:
			train:
				images:
				gt:
			test:
				images:
				gt:
	 (4) 在测试时，将test作为验证集和测试集。

2.模型运行：
	（1）进入main函数，直接运行即可

3. 运行结果查看：
	（1）进入log文件夹，即可查看运行情况与结果

