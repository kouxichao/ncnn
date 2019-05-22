# 在原项目基础上添加自己的一些更改.

add bilstm lstmcell fullyconnected dimop layers.
目前添加层了bilstm、lstmcell、三维全连接层、mxnet2ncnn的调整以产生适应于lstm的bin文件。

for bilstm, the default weight order is IFGO, same as pytorch, but can switch to IGFO(mxnet). 
注：pytorch 的lstm权重存储顺序IFGO, mxnet框架是IGFO。可通过下面参数选定。

operation param weight table:
```
bilstm: 
	param id			param phase                            default	

	  0			num_lstm_layers_to_stack(1 or 2)                  0
	  1			isbilstm(0 for lstm,1 for bilstm)                 0
	  2			num_output                                        0
	  3			weight_ih_data_size                               0
	  4			weight_hh_data_size                               0
	  5			isfrom_mxnet_weight(0 for pytorch,1 for mxnet)    0
```
# works 相关项目文件

crnn : 包含crnn_chinese(由mxnet转换过来的，所以bilstm 的第五个参数是1), crnn_english(由pytorch转换过来的，所以bilstm的第五个参数是0);

face_recognition: 
```
	need sqlite3, compile and install it on your system.
	需用到sqlite3，提前安装sqlite到系统。
```

image_caption(with attention):
```
	trained with pytorch.
```
book_classify(目前无用)

所有项目源文件在example文件夹下， 可选择编译。
用到示例项目用到dlib，需下载dlib源文件放到ncnn根目录，并将文件名改为dlib（即去掉版本号）

# models 模型下载
链接: https://pan.baidu.com/s/1GAg5LPN6-2MdKjIgz0i-ag 提取码: d7nv 

模型放入 ${root_ncnn}/models/crnn/,${root_ncnn}/models/face_recognition/
