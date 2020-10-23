# machine-learning-predict-wine

Our Whitewine dataset is produced in a particular area of Portugal. Data are collected on 12 different properties of the wines of which response is Quality(discrete), and the rest are chemical properties(all are continuous). 

## Data Preprocessing

After reading dataset, we need to perform necessary diagnostic (repeated data, missing data, outlier, multicollinearity issue and distribution of response) on dataset. For outliers, we start from calculating ’semi studentized deleted residuals’:

![equation]("https://www.codecogs.com/eqnedit.php?latex=t_i&space;=&space;\frac{d_i}{SE(d_i)}=\frac{e_i}{\sqrt&space;{MSE_{(i)}}(1-h_{ii}))&space;}\sim&space;t_{n-p-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t_i&space;=&space;\frac{d_i}{SE(d_i)}=\frac{e_i}{\sqrt&space;{MSE_{(i)}}(1-h_{ii}))&space;}\sim&space;t_{n-p-1}" title="t_i = \frac{d_i}{SE(d_i)}=\frac{e_i}{\sqrt {MSE_{(i)}}(1-h_{ii})) }\sim t_{n-p-1}")  

