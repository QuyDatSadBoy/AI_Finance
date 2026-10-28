# Huấn luyện chung EW + VW (cùng kiến trúc, cùng siêu tham số)

Sharpe mỗi seed:

 seed  VW_late  VW_dev1  VW_dev5  VW_dev10  VW_holdout1  VW_holdout5  VW_holdout10  EW_late  EW_dev1  EW_dev5  EW_dev10  EW_holdout1  EW_holdout5  EW_holdout10
    0   -0.086    0.160    0.126     0.118        0.236        0.054         0.045    1.668    1.788    1.787     1.673        0.656        0.707         0.703
    1    0.163    0.334    0.322     0.282        0.547        0.298         0.122    1.802    1.870    1.848     1.771        0.714        0.687         0.733
    2    0.170    0.412    0.307     0.240        0.059        0.046         0.056    1.659    1.823    1.758     1.639        0.765        0.705         0.723

Seed chọn bằng dev muộn 1995-2004 của họ VW: 2

Họ VW:

 top      kỳ             pp  lãi_năm      t  sharpe  p_vs_EB
   1     dev Chung (seed 2)    2.697  1.933   0.412    0.548
   1     dev             EB    3.596  2.560   0.546      NaN
   1     dev            |t|    4.333  2.840   0.606    0.348
   1 holdout Chung (seed 2)    0.232  0.237   0.059    0.840
   1 holdout             EB    0.509  0.527   0.132      NaN
   1 holdout            |t|    0.378  0.414   0.103    0.755
   1    full Chung (seed 2)    1.659  1.828   0.297    0.532
   1    full             EB    2.296  2.520   0.409      NaN
   1    full            |t|    2.668  2.759   0.448    0.446
   5     dev Chung (seed 2)    1.740  1.439   0.307    0.154
   5     dev             EB    2.765  2.764   0.589      NaN
   5     dev            |t|    2.603  2.371   0.505    0.268
   5 holdout Chung (seed 2)    0.116  0.184   0.046    0.674
   5 holdout             EB   -0.295 -0.398  -0.100      NaN
   5 holdout            |t|   -0.197 -0.297  -0.074    0.807
   5    full Chung (seed 2)    1.056  1.410   0.229    0.434
   5    full             EB    1.476  2.233   0.362      NaN
   5    full            |t|    1.424  2.045   0.332    0.620
  10     dev Chung (seed 2)    1.259  1.124   0.240    0.114
  10     dev             EB    2.169  2.537   0.541      NaN
  10     dev            |t|    2.124  2.296   0.490    0.516
  10 holdout Chung (seed 2)    0.124  0.223   0.056    0.549
  10 holdout             EB   -0.419 -0.668  -0.167      NaN
  10 holdout            |t|   -0.242 -0.419  -0.105    0.551
  10    full Chung (seed 2)    0.781  1.133   0.184    0.465
  10    full             EB    1.080  1.915   0.311      NaN
  10    full            |t|    1.128  1.911   0.310    0.991

IC VW dev: chung 0.040, EB 0.039, p=0.982  
IC VW holdout: chung -0.002, EB -0.001, p=0.976  
IC VW full: chung 0.022, EB 0.022, p=0.996  

TB seed VW: dev 0.30/0.25/0.21; holdout 0.28/0.13/0.07

Họ EW:

 top      kỳ             pp  lãi_năm     t  sharpe  p_vs_EB
   1     dev Chung (seed 2)    9.348 8.551   1.823    0.068
   1     dev             EB    8.949 7.263   1.549      NaN
   1     dev            |t|    9.860 9.137   1.948    0.045
   1 holdout Chung (seed 2)    2.588 3.059   0.765    0.781
   1 holdout             EB    2.791 3.274   0.818      NaN
   1 holdout            |t|    2.915 3.358   0.839    0.668
   1    full Chung (seed 2)    6.502 8.759   1.421    0.195
   1    full             EB    6.356 7.843   1.272      NaN
   1    full            |t|    6.936 9.363   1.519    0.056
   5     dev Chung (seed 2)    8.264 8.245   1.758    0.053
   5     dev             EB    8.415 8.970   1.913      NaN
   5     dev            |t|    8.834 9.339   1.991    0.375
   5 holdout Chung (seed 2)    1.994 2.821   0.705    0.449
   5 holdout             EB    2.292 3.111   0.778      NaN
   5 holdout            |t|    2.220 2.998   0.749    0.147
   5    full Chung (seed 2)    5.624 8.425   1.367    0.063
   5    full             EB    5.837 9.110   1.478      NaN
   5    full            |t|    6.049 9.336   1.515    0.466
  10     dev Chung (seed 2)    7.336 7.689   1.639    0.050
  10     dev             EB    7.730 8.735   1.862      NaN
  10     dev            |t|    7.760 8.696   1.854    0.818
  10 holdout Chung (seed 2)    1.816 2.893   0.723    0.251
  10 holdout             EB    2.258 3.420   0.855      NaN
  10 holdout            |t|    2.212 3.335   0.834    0.105
  10    full Chung (seed 2)    5.012 8.019   1.301    0.027
  10    full             EB    5.426 9.105   1.477      NaN
  10    full            |t|    5.424 9.031   1.465    0.581

IC EW dev: chung 0.157, EB 0.188, p=0.123  
IC EW holdout: chung 0.041, EB 0.021, p=0.448  
IC EW full: chung 0.108, EB 0.118, p=0.582  

TB seed EW: dev 1.83/1.80/1.69; holdout 0.71/0.70/0.72
