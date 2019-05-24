# 在原项目基础上添加自己的一些更改.
Layers:
```
	bilstm
 	lstmcell
 	fullyconnected: multi_dim(3) innerproduct to support lstm or other cases like multi input in imagecaption
 	dimop(incomplete): Only contains the code in my case.                       
	binaryOp: change this layer using concise code and thus avoid some bugs(oprations like broadcasting in numpy).
````
tools:
```
	mxnet2ncnn的调整以产生适应于lstm的bin文件。
	pytorch 2 ncnn or caffe for reference.
```
note:
```
        for bilstm, the default weight order is IFGO, same as pytorch, but can switch to IGFO(mxnet). 
	pytorch 的lstm权重存储顺序IFGO, mxnet框架是IGFO。可通过下面参数选定。
```
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

lstmcell(3 input blobs, two output blobs):
	
	  0                     input_size                                        0                    
	  1                     hidden_size                                       0
	
param examples:
lstmcell(输入dims=2560，shape=(num_samples, 2560);输出dims=512, shape=(num_samples, 512)): 
LSTMCell         lstm   3 2 lstm_input decoder_h decoder_c lstm_out_h lstm_out_c 0=2560 1=512
```

# works 相关项目文件
all the works are in examples directory, and optional compilation is support.
所有项目源文件在example文件夹下， 可选择编译。
用到示例项目用到dlib，需下载dlib源文件放到ncnn根目录，并将文件名改为dlib（即去掉版本号)。

crnn :
```
包含两个项目文件：
	crnn_chinese(由mxnet转换过来的，所以bilstm 的第五个参数是1)； 
	crnn_english(由pytorch转换过来的，所以bilstm的第五个参数是0)。
```
face_recognition: 
```
	need sqlite3, compile and install it on your system.
	需用到sqlite3，提前安装sqlite到系统。
```

image_caption(with attention):
```
	trained with pytorch.
```
book_classify:
```
	just keep
```



# models 模型下载
链接(link): https://pan.baidu.com/s/1GAg5LPN6-2MdKjIgz0i-ag 提取码: d7nv 

模型放入 ${root_ncnn}/models/crnn/,${root_ncnn}/models/face_recognition/
