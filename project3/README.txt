代码说明：

环境：
    python 3.7.16
    pandas 1.3.5
    numpy 1.21.6
    torch 1.13.1
    以上为测试环境，实际使用时，对版本的差异应该不会特别敏感

tools.py:
    存储了一些被另外两份代码用到的方法，不应被直接运行。

trainer.py:（重要）
    运行命令：python trainer.py
    用于训练模型，训练时同级目录下须存在 data 文件夹，里面包括 traindata.csv 和 trainlabel.txt ，
    结果会存在同级目录下的 string_argument_dict.npy 和 AdultCensusIncomeNet.pt 中。
    训练开始后，中途可以随时停止，脚本会自动保存中间结果。

AdultCensusIncomeNet.pt:
    训练完成后的神经网络的参数

string_argument_dict.npy:
    用于将非数字参数转化成数字参数的字典

predictor.py:（重要）
    运行命令：python predictor.py
    用于进行预测，预测时同级目录下须存在上述两个文件，以及 data 文件夹，里面包括 testdata.csv ，
    结果会存在同级目录下的 testlabel.txt 中。

testlabel.txt：
    包含和 data/testdata.csv 行数一样多的 0 或 1 ，表示模型对每组数据的预测。

backup.zip：
    由于一旦运行了 trainer.py 或 predictor.py ，几个结果就会被覆盖，因此提供了一份备份。

research.py:
    运行命令：python research.py
    仅用于生成报告中用的一张图
