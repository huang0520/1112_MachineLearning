HW1.py主要分為以下幾個部分：

1. func main()
    呼叫各函式運行的部分。

2. func split_train_test()
    將輸入之 dataset 隨機分為 train, test 兩個部分，其中 test 佔 60 筆資料。

3. func split_target_feature()
    將輸入之 dataset 分為 target, feature 方便後續處理。

4. class MAP()

    用於預測之MAP模型，將用於 training 之 target, feature 輸出來建立。

    1. __cal_param()
        通過輸入之 dataset 計算如 mean, standard variation, etc.

    2. pred()
        輸入要預測之 feature set 並產生對應的預測。 

    3. __pred_one()
        pred 的 subfunction，用於預測單一組的特徵。

    4. __log_likelihood()
        __pred_one 的 subfunction，計算 log likelihood。