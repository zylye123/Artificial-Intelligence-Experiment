# Artificial-Intelligence-Experiment
Artificial Intelligence Experiment Midterm; Spine Diagnosis

utils.py

​ 该文件不需要输入。其功能为读取txt文件，将其转成网络需要的大小格式。主要的函数是load_label，其中包含核心代码。其功能主要是将txt的坐标转成对应的heatmap，分类的v1,v2等字符用字典转成相应的数字当作标签。为train.py和Inference.py文件所调用。​

union.py

​ 该文件不需要输入，主要内容为神经网络的定义。主要函数为union_model，其功能是把之前定义的ASPP、SPP等网络结合。为train.py文件所调用。

dataloader.py

​ 该文件不需要输入，其功能为对Dataset库函数定义。方便train.py文件调用DataLoader库函数。DataLoader库函数的作用是方便对数据进行预处理。

loss.py

​ 该文件不需要输入，其功能是实现分类loss的计算。其主要函数Lossfunction4为分类的loss函数，包括前5个二分类的loss计算和后4+1分类的loss计算。为 train.py文件所调用。

train.py

​ 要调用该文件需要将trian文件夹和该文件放在同一个文件夹下。其功能为训练模型，先调用utils.py和dataloader.py文件完成对数据的预处理，包括旋转、随即裁剪等。然后调用union.py文件，定义一个网络莫模型，将数据传进网络模型中，前向传播。之后调用loss.py文件，计算总的loss值，再进行反向传播更新网络参数。最后该文件会每5个epoch保存一次模型。

Inference.py

​ 要调用该文件需要将上面的train.py训练出来的模型、test文件夹和该python文件放在同一个文件夹下。其功能是测试训练出来模型的好坏和将模型输出保存为txt文件（txt文件保存在test文件夹下）。其中的interence函数是将网络的输出转为关键点的坐标和相应的类别。Model_Evaluation函数将预测的关键点的坐标、类别和真实标签的项比较，计算出坐标回归的Mseloss和分类的准确率，并输出出来。
