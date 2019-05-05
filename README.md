# 在原项目基础上添加自己的一些更改.

目前添加层了lstm、二维全连接层、mxnet2ncnn的调整以产生适应于lstm的bin文件。

注：pytorch 的lstm权重存储顺序IFGO, mxnet框架是IGFO。可通过下面参数选定。

operation param weight table:
```
bilstm: 
	param id			        param phase							      	default	

	  0					num_lstm_layers_to_stack(1 or 2)						0
	  1					isbilstm(0 for lstm,1 for bilstm)						0
	  2					num_output												0
	  3					weight_ih_data_size										0
	  4					weight_hh_data_size										0
	  5					isfrom_mxnet_weight(0 for pytorch,1 for mxnet)			0
```
# 相关项目文件

crnn : 包含crnn_chinese, crnn_eng;

face_recognition: 
```
	需用到sqlite3，提前安装sqlite到系统。
```
book_classify(目前无用)

所有项目源文件在example文件夹下， 可选择编译。
用到示例项目用到dlib，需下载dlib源文件放到ncnn根目录，并将文件名改为dlib（即去掉版本号）
