# machine-learning-predict-wine

Our Whitewine dataset is produced in a particular area of Portugal. Data are collected on 12 different properties of the wines of which response is Quality(discrete), and the rest are chemical properties(all are continuous). 

## Data Preprocessing

After reading dataset, we need to perform necessary diagnostic (repeated data, missing data, outlier, multicollinearity issue and distribution of response) on dataset. For outliers, we start from calculating ’semi studentized deleted residuals’:

![equation](https://latex.codecogs.com/gif.latex?t_i%20%3D%20%5Cfrac%7Bd_i%7D%7BSE%28d_i%29%7D%3D%5Cfrac%7Be_i%7D%7B%5Csqrt%20%7BMSE_%7B%28i%29%7D%7D%281-h_%7Bii%7D%29%29%20%7D%5Csim%20t_%7Bn-p-1%7D)  
We consider the Bonferroni procedure and the rejection region for the ith data point is:

![equation](https://latex.codecogs.com/gif.latex?R_i%20%3D%20%5Clbrace%20%7C%28t_i%29%7C%20%3E%20t_%7B1-%20%5Cfrac%7B%5Calpha%7D%7B2n%7D%2Cn-p-1%7D%20%5Crbrace%2C%20i%3D1%2C%20%5Ccdots%2Cn)

We detected and deleted observation 2782 in original dataset as an outlier from methods above. We use vif() to detect multicollinearity problem and table() to see distribution of response. We combine to 5 quality  3,4→0; 5→1; 6→2; 7→3; 8,9→4.

Next, we could deploy multiple machine learning algorithms regularized linear regression, random forest, boosting and neural network on this dataset.

## Regularized Linear Models

We used regularized linear regression model using the package glmnet() and throwed in all interactions and second order. We applied grid search on alpha and nfolds to get the optimal parameters on 18 settings. The first figure shows how out of sample error changes with different setting, the bottom shows corresponding setting for the best glm.

<p align="middle">
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/glm.png" width="600" />
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/settingglm.png" width="600" />
</p>

## Tree
We compared single tree with two popular ensemble methods(random forest\&boosting). We also did grid search to find optimal hyperparameter for these 2 methods. 
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/treeooberror.png" width="600" />
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/tree%20compare.png" width="600" />
</p>
The top plot demonstrated out of bag error while the bottom showed ROC curve(with AUC) with regard to the best setting for each model. The other advantage for tree models is that it could provide variable importance like the figure below. From the plot we could see variable ’alcohol’ is the most important obviously, variable ’volatile.acidity’,’free.sulfur.dioxide’ and ’chlorides’ seem to be relatively important, ’fixed.acidity’ and ’sulphates’ turn out to be most negligible.
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/variable%20importance.png" width="600" />
</p>

## Neural Network

We implemented Single Layer Neural Network and Deep Layer Neural Network to train. In this case grid search is not an ideal methods since it may take few hours to run, however, we found another search method called random search. Random search is a technique where random combinations of the hyperparameters are used to find best solution for the built model. It tries random combination of the values we set. In order to compare, we used both and see how our of sample error looked like:

<p align="middle">
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/nnerror.png" width="600" />
</p>

We could see that random search for deep neural network obtained lowest error.

**Stacking**
From the lift curve below, it’s interesting that the category 0 and 4 looks good through lift curve however the accuracy rate is very low when predicting them. After examining the probability predicted, we think the main reason is the total number in those categories is too low. The probability predicted is too small to beat the most two populor category 2 and 3. If it is a binary case, it can be fixed by setting the threshold. But for five category, it’s hard to do that. Therefore, maybe it is possible to let the machine find the pattern behind the lift curve by stacking! Specifically, we could let the output from single neural network as the input of another neural network. And the right figure shows it really did improvement when predicting for the categories which had a good lift curve. 

<p align="middle">
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/nn%20improvement.png" width="600" />
</p>

Finally, let’s compare all best models: GLMs, RF and NN+NN by putting them all on the same lift plot.

<p align="middle">
  <img src="https://github.com/Shuyi-bomi/machine-learning-predict-wine/blob/main/figure/final%20lift%20curve.png" width="600" />
</p>
From the lift curve, we could see that NN+NN shows a good performance for all categories especially 1, 2 and 3, a lot better compared to the other two methods.
