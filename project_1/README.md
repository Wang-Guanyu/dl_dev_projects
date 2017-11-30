### Deep Learning Course Project 1

The project is about to solve the problem:

> Given two numbers **x** and **y** in the range of **0** to **100**, use deep learning method to predict the result of adding x and y.

The main idea is to consider it as a classification problem. Since x and y are in the range of 0 to 100, and result of addition is in the range of **0** to **200**, therefore there are **201** classes.

In addition, to make this problem challenging, we take out some values **(25,50,70)** from training set and use them in the prediction, so that we can check whether the model can deal with unknown cases.