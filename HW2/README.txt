========== 簡述 ==========
HW2.ipynb 中使用numpy實現 2 layer NN 及 3 layer NN，並繪製 model 的 loss function 及 decision region。

========== 說明 ==========
1. Configuration
此段主要包含各種超參數及隨機數固定

2. Read image & reduce dimension
此段通過 skimage 讀取圖片，並通過 sklearn 的 PCA 功能將圖片降成二維，最後再將其 normalize。

3. Activation & Utility function
此段定義了 sigmoid, tanh, ReLU, ELU 四種 Activation function 並在後續於 NN 中依選擇呼叫。除此之外，還定義了 softmax 及繪製 loss curve, evaluate accuracy 的 function。

4. NN model
此段通過 numpy 實現使用 SGD 作為 back propagation 的方式，並以 cross-entropy 作為 loss function。

5. Decision region function
此段定義了繪製 decision region 的 function。其中後方顏色區塊是使用 model 去預測圖片 1000 x 1000 的點並繪製，而前方分布的點則是實際資料，以此可以去分析 model。