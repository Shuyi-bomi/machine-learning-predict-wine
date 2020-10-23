# machine-learning-predict-wine

Our Whitewine dataset is produced in a particular area of Portugal. Data are collected on 12 different properties of the wines of which response is Quality(discrete), and the rest are chemical properties(all are continuous). 

## Data Preprocessing

After reading dataset, we need to perform necessary diagnostic (repeated data, missing data, outlier, multicollinearity issue and distribution of response) on dataset. For outliers, we start from calculating ’semi studentized deleted residuals’:

![equation](https://latex.codecogs.com/gif.latex?t_i%20%3D%20%5Cfrac%7Bd_i%7D%7BSE%28d_i%29%7D%3D%5Cfrac%7Be_i%7D%7B%5Csqrt%20%7BMSE_%7B%28i%29%7D%7D%281-h_%7Bii%7D%29%29%20%7D%5Csim%20t_%7Bn-p-1%7D)  

