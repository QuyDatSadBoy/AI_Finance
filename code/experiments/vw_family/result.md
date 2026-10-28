# Họ kế toán value-weighted (mô hình chốt học lại, không chỉnh siêu tham số)

Sharpe top 1/5/10% mỗi seed:

 seed  late9504  dev1  dev5  dev10  holdout1  holdout5  holdout10  full1  full5  full10
    0     0.033 0.312 0.170  0.134    -0.023    -0.044     -0.055  0.195  0.103   0.078
    1     0.052 0.233 0.171  0.087     0.050     0.043      0.081  0.175  0.131   0.081
    2     0.098 0.327 0.175  0.139     0.008    -0.040     -0.053  0.231  0.114   0.086
    3    -0.089 0.114 0.084  0.025     0.191     0.034      0.044  0.136  0.068   0.029
    4    -0.076 0.188 0.097  0.029     0.142     0.126      0.055  0.170  0.101   0.035
    5     0.550 0.611 0.580  0.524    -0.023     0.047      0.043  0.406  0.402   0.361

Seed chọn bằng 1995-2004: **5**

Bảng so với EB và |t|:

 top      kỳ               pp  lãi_năm      t  sharpe  p_vs_EB  p_vs_|t|
   1     dev Mô hình (seed 5)    3.503  2.868   0.611    0.676     0.965
   1     dev               EB    3.596  2.560   0.546      NaN       NaN
   1     dev              |t|    4.333  2.840   0.606    0.348       NaN
   1 holdout Mô hình (seed 5)   -0.079 -0.092  -0.023    0.637     0.700
   1 holdout               EB    0.509  0.527   0.132      NaN       NaN
   1 holdout              |t|    0.378  0.414   0.103    0.755       NaN
   1    full Mô hình (seed 5)    1.995  2.500   0.406    0.981     0.745
   1    full               EB    2.296  2.520   0.409      NaN       NaN
   1    full              |t|    2.668  2.759   0.448    0.446       NaN
   5     dev Mô hình (seed 5)    2.313  2.720   0.580    0.920     0.553
   5     dev               EB    2.765  2.764   0.589      NaN       NaN
   5     dev              |t|    2.603  2.371   0.505    0.268       NaN
   5 holdout Mô hình (seed 5)    0.119  0.189   0.047    0.631     0.682
   5 holdout               EB   -0.295 -0.398  -0.100      NaN       NaN
   5 holdout              |t|   -0.197 -0.297  -0.074    0.807       NaN
   5    full Mô hình (seed 5)    1.389  2.476   0.402    0.725     0.559
   5    full               EB    1.476  2.233   0.362      NaN       NaN
   5    full              |t|    1.424  2.045   0.332    0.620       NaN
  10     dev Mô hình (seed 5)    1.832  2.459   0.524    0.864     0.792
  10     dev               EB    2.169  2.537   0.541      NaN       NaN
  10     dev              |t|    2.124  2.296   0.490    0.516       NaN
  10 holdout Mô hình (seed 5)    0.097  0.170   0.043    0.490     0.604
  10 holdout               EB   -0.419 -0.668  -0.167      NaN       NaN
  10 holdout              |t|   -0.242 -0.419  -0.105    0.551       NaN
  10    full Mô hình (seed 5)    1.102  2.225   0.361    0.661     0.675
  10    full               EB    1.080  1.915   0.311      NaN       NaN
  10    full              |t|    1.128  1.911   0.310    0.991       NaN

IC dev: mô hình 0.036, EB 0.039, p=0.793  
IC holdout: mô hình -0.003, EB -0.001, p=0.941  
IC full: mô hình 0.019, EB 0.022, p=0.831  

Trung bình các seed: dev 0.30/0.21/0.16; holdout 0.06/0.03/0.02; full 0.22/0.15/0.11