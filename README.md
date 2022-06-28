# ORL-Dataset-Face-Recognition
ORL Faces Dataset是一个拥有40类、400张尺寸为 112X92 的单通道人脸图片的数据集。每一类代表一个人，一共10张照片。对ORL数据集的详细介绍可以参考 https://blog.csdn.net/fengbingchun/article/details/79008891 

文件夹中一共有3个文件：ORL Faces Database.zip，facenet.py，keras_LeNet.py

1.ORL Faces Database.zip：ORL人脸数据集的zip压缩包，一共实际上有35类，每一类有10张bmp文件。


2.facenet.py：使用pytorch写facenet对ORL中每张图片所属的人物进行检索。使用前需要在指令面板中pip install facenet_pytorch，并且从 https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt 下载LFW人脸识别模型以让facenet的resnet模型进行提取特征（如果让facenet_pytorch自己下载的话速度会很慢，提前下载的时候建议开启github加速），并且将下载好的文件放在 C:\Users\Administrator\ .cache\torch\checkpoints 路径下

3.keras_LeNet.py：使用keras模仿MNIST的识别模型训练的代码，正确率97%左右。
