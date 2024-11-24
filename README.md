##### LSTM_STOCK
## 基于LSTM对股票进行预测研究
这段代码是一个使用长短期记忆网络（LSTM）来预测股票价格的Python脚本。它定义了一个名为LSTMStockPredictor的类，该类负责加载股票数据、预处理数据、构建LSTM模型、训练模型、进行预测以及评估模型性能。以下是代码的主要组成部分及其功能：

## 导入库：
代码开始部分导入了所需的Python库，包括数据处理（numpy, pandas）、数据获取（yfinance）、绘图（matplotlib）、机器学习（sklearn）和深度学习（tensorflow.keras）。

## 类定义：
LSTMStockPredictor类包含了以下方法：
__init__：初始化方法，设置股票代码、日期范围、未来预测天数等参数，并调用其他方法加载和预处理数据、分割训练测试数据集、构建模型。
_init_logger：初始化日志记录器。
_load_and_preprocess_data：从Yahoo Finance下载股票数据，并使用MinMaxScaler进行归一化处理。
_split_train_test_data：将数据集分割为训练集和测试集。
_build_model：构建LSTM模型。
create_dataset：将数据集转换为模型输入的格式。
train：训练LSTM模型。
predict：使用模型进行预测。
predict_future_prices：预测未来的股票价格。
plot_results：绘制实际价格与预测价格的对比图。
calculate_mape：计算平均绝对百分比误差（MAPE）。

## 辅助函数：
check_bool：将输入转换为布尔值。

## 主函数：
use_stock_predictor：使用LSTMStockPredictor类进行股票价格预测。
main：设置命令行参数解析器，并解析命令行参数。
启动方法：脚本可以通过命令行参数启动，以下是参数的意义：

-n 或 --stock_name：股票代码（例如"AAPL"）。
-s 或 --start_date：开始日期（格式为"yyyy-mm-dd"）。
-e 或 --end_date：结束日期（格式为"yyyy-mm-dd"）。
-o 或 --out_dir：保存结果的目录，默认为当前工作目录。
-p 或 --plot：是否绘制结果图，默认为False。
-f 或 --future：预测未来的股票价格天数。
-m 或 --metrics：是否显示评估指标，默认为False。

## 示例：
```
python LSTM.py -n "AAPL" -s "2020-01-01" -e "2023-12-01" -m True -p True -f 5
```

