## Accuracy, K = 5 (mean over clients, then mean ± sd over seeds)

### Dataset B, iid

| Method | R² own | R² other clients | RMSE own (dB) | Δf_r own (GHz) | worst-client R² own |
|---|---|---|---|---|---|
| Local only | 0.262 ± 0.019 | 0.261 ± 0.019 | 2.547 ± 0.041 | 1.397 ± 0.265 | 0.197 ± 0.026 |
| FedAvg | 0.371 ± 0.014 | 0.371 ± 0.015 | 2.351 ± 0.032 | 0.864 ± 0.189 | 0.350 ± 0.015 |
| FedProx | 0.371 ± 0.012 | 0.371 ± 0.012 | 2.351 ± 0.028 | 0.872 ± 0.185 | 0.350 ± 0.012 |
| FedAvg + fine-tune | 0.374 ± 0.014 | 0.375 ± 0.015 | 2.345 ± 0.030 | 0.868 ± 0.226 | 0.353 ± 0.014 |
| Centralised (pooled) | 0.508 ± 0.016 | 0.508 ± 0.015 | 2.079 ± 0.038 | 1.819 ± 0.194 | 0.469 ± 0.011 |

### Dataset B, shift

| Method | R² own | R² other clients | RMSE own (dB) | Δf_r own (GHz) | worst-client R² own |
|---|---|---|---|---|---|
| Local only | 0.274 ± 0.005 | 0.118 ± 0.015 | 2.482 ± 0.014 | 0.320 ± 0.020 | 0.213 ± 0.014 |
| FedAvg | 0.332 ± 0.029 | 0.333 ± 0.029 | 2.381 ± 0.066 | 0.956 ± 0.064 | 0.293 ± 0.024 |
| FedProx | 0.353 ± 0.041 | 0.354 ± 0.041 | 2.342 ± 0.081 | 0.933 ± 0.074 | 0.317 ± 0.040 |
| FedAvg + fine-tune | 0.365 ± 0.027 | 0.276 ± 0.037 | 2.321 ± 0.063 | 0.290 ± 0.075 | 0.321 ± 0.031 |
| Centralised (pooled) | 0.512 ± 0.011 | 0.512 ± 0.011 | 2.035 ± 0.026 | 2.419 ± 0.062 | 0.478 ± 0.018 |

### Dataset C, iid

| Method | R² own | R² other clients | RMSE own (dB) | Δf_r own (GHz) | worst-client R² own |
|---|---|---|---|---|---|
| Local only | 0.963 ± 0.008 | 0.963 ± 0.011 | 0.241 ± 0.027 | 0.011 ± 0.000 | 0.939 ± 0.013 |
| FedAvg | 0.986 ± 0.003 | 0.985 ± 0.003 | 0.153 ± 0.012 | 0.012 ± 0.001 | 0.981 ± 0.005 |
| FedProx | 0.985 ± 0.003 | 0.985 ± 0.003 | 0.154 ± 0.016 | 0.011 ± 0.001 | 0.981 ± 0.006 |
| FedAvg + fine-tune | 0.984 ± 0.002 | 0.984 ± 0.004 | 0.159 ± 0.011 | 0.012 ± 0.001 | 0.979 ± 0.004 |
| Centralised (pooled) | 0.990 ± 0.003 | 0.989 ± 0.003 | 0.130 ± 0.017 | 0.010 ± 0.000 | 0.986 ± 0.006 |

### Dataset C, shift

| Method | R² own | R² other clients | RMSE own (dB) | Δf_r own (GHz) | worst-client R² own |
|---|---|---|---|---|---|
| Local only | 0.975 ± 0.002 | 0.348 ± 0.038 | 0.190 ± 0.008 | 0.011 ± 0.001 | 0.967 ± 0.008 |
| FedAvg | 0.813 ± 0.011 | 0.847 ± 0.003 | 0.455 ± 0.006 | 0.026 ± 0.001 | 0.458 ± 0.068 |
| FedProx | 0.818 ± 0.017 | 0.850 ± 0.010 | 0.449 ± 0.006 | 0.026 ± 0.002 | 0.473 ± 0.082 |
| FedAvg + fine-tune | 0.984 ± 0.000 | 0.294 ± 0.033 | 0.156 ± 0.003 | 0.011 ± 0.001 | 0.977 ± 0.001 |
| Centralised (pooled) | 0.990 ± 0.001 | 0.990 ± 0.001 | 0.120 ± 0.009 | 0.010 ± 0.000 | 0.987 ± 0.001 |

## Dataset B, shift: effect of the number of clients

| K | Local R² own | FedAvg R² own | FedAvg+FT R² own | Central R² own |
|---|---|---|---|---|
| 3 | 0.280 ± 0.019 | 0.368 ± 0.041 | 0.384 ± 0.038 | 0.515 ± 0.024 |
| 5 | 0.274 ± 0.005 | 0.332 ± 0.029 | 0.365 ± 0.027 | 0.512 ± 0.011 |
| 10 | 0.300 ± 0.018 | 0.271 ± 0.023 | 0.356 ± 0.010 | 0.498 ± 0.020 |

## Chance level of worst-client coverage (exactly valid local CP)

| Dataset | K | partition | designs per client test set | expected worst-client coverage |
|---|---|---|---|---|
| B | 3 | shift | 137 | 0.876 |
| B | 5 | iid | 82 | 0.847 |
| B | 5 | shift | 82 | 0.847 |
| B | 10 | shift | 41 | 0.793 |
| C | 5 | iid | 60 | 0.835 |
| C | 5 | shift | 60 | 0.835 |

## Conformal certificates on `rmse_d` (target 90 %), K = 5

Coverage per client then: mean, min over clients (mean over seeds); bound q.

### Dataset B, iid

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.906 | 0.850 | 3.409 |
| Local only | pooled-CP | 0.899 | 0.862 | 3.328 |
| Local only | QQ-CP | 0.899 | 0.866 | 3.331 |
| FedAvg | local-CP | 0.908 | 0.862 | 3.218 |
| FedAvg | pooled-CP | 0.891 | 0.858 | 3.152 |
| FedAvg | QQ-CP | 0.876 | 0.846 | 3.113 |
| FedProx | local-CP | 0.904 | 0.854 | 3.206 |
| FedProx | pooled-CP | 0.894 | 0.866 | 3.156 |
| FedProx | QQ-CP | 0.879 | 0.850 | 3.120 |
| FedAvg + fine-tune | local-CP | 0.904 | 0.854 | 3.219 |
| FedAvg + fine-tune | pooled-CP | 0.894 | 0.858 | 3.155 |
| FedAvg + fine-tune | QQ-CP | 0.898 | 0.866 | 3.173 |
| Centralised (pooled) | local-CP | 0.901 | 0.858 | 3.026 |
| Centralised (pooled) | pooled-CP | 0.894 | 0.874 | 2.987 |
| Centralised (pooled) | QQ-CP | 0.894 | 0.866 | 2.982 |

### Dataset B, shift

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.907 | 0.862 | 3.277 |
| Local only | pooled-CP | 0.904 | 0.870 | 3.245 |
| Local only | QQ-CP | 0.891 | 0.846 | 3.206 |
| FedAvg | local-CP | 0.912 | 0.866 | 3.199 |
| FedAvg | pooled-CP | 0.904 | 0.862 | 3.167 |
| FedAvg | QQ-CP | 0.901 | 0.862 | 3.151 |
| FedProx | local-CP | 0.915 | 0.874 | 3.185 |
| FedProx | pooled-CP | 0.901 | 0.862 | 3.139 |
| FedProx | QQ-CP | 0.904 | 0.866 | 3.142 |
| FedAvg + fine-tune | local-CP | 0.916 | 0.882 | 3.137 |
| FedAvg + fine-tune | pooled-CP | 0.908 | 0.874 | 3.120 |
| FedAvg + fine-tune | QQ-CP | 0.911 | 0.874 | 3.128 |
| Centralised (pooled) | local-CP | 0.916 | 0.883 | 2.964 |
| Centralised (pooled) | pooled-CP | 0.901 | 0.850 | 2.920 |
| Centralised (pooled) | QQ-CP | 0.898 | 0.850 | 2.917 |

### Dataset C, iid

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.897 | 0.789 | 0.323 |
| Local only | pooled-CP | 0.890 | 0.828 | 0.268 |
| Local only | QQ-CP | 0.887 | 0.817 | 0.266 |
| FedAvg | local-CP | 0.886 | 0.811 | 0.153 |
| FedAvg | pooled-CP | 0.896 | 0.844 | 0.143 |
| FedAvg | QQ-CP | 0.900 | 0.850 | 0.149 |
| FedProx | local-CP | 0.888 | 0.811 | 0.156 |
| FedProx | pooled-CP | 0.898 | 0.844 | 0.148 |
| FedProx | QQ-CP | 0.901 | 0.844 | 0.149 |
| FedAvg + fine-tune | local-CP | 0.894 | 0.822 | 0.172 |
| FedAvg + fine-tune | pooled-CP | 0.901 | 0.850 | 0.167 |
| FedAvg + fine-tune | QQ-CP | 0.892 | 0.839 | 0.164 |
| Centralised (pooled) | local-CP | 0.891 | 0.811 | 0.137 |
| Centralised (pooled) | pooled-CP | 0.898 | 0.850 | 0.133 |
| Centralised (pooled) | QQ-CP | 0.879 | 0.828 | 0.130 |

### Dataset C, shift

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.914 | 0.861 | 0.290 |
| Local only | pooled-CP | 0.914 | 0.861 | 0.252 |
| Local only | QQ-CP | 0.923 | 0.867 | 0.320 |
| FedAvg | local-CP | 0.899 | 0.811 | 0.837 |
| FedAvg | pooled-CP | 0.910 | 0.800 | 0.805 |
| FedAvg | QQ-CP | 0.954 | 0.883 | 1.128 |
| FedProx | local-CP | 0.899 | 0.811 | 0.814 |
| FedProx | pooled-CP | 0.907 | 0.789 | 0.766 |
| FedProx | QQ-CP | 0.956 | 0.889 | 1.122 |
| FedAvg + fine-tune | local-CP | 0.903 | 0.856 | 0.209 |
| FedAvg + fine-tune | pooled-CP | 0.910 | 0.783 | 0.177 |
| FedAvg + fine-tune | QQ-CP | 0.953 | 0.878 | 0.267 |
| Centralised (pooled) | local-CP | 0.920 | 0.850 | 0.146 |
| Centralised (pooled) | pooled-CP | 0.911 | 0.828 | 0.134 |
| Centralised (pooled) | QQ-CP | 0.939 | 0.856 | 0.164 |

## Conformal certificates on `sup` (target 90 %), K = 5

Coverage per client then: mean, min over clients (mean over seeds); bound q.

### Dataset B, iid

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.905 | 0.834 | 21.270 |
| Local only | pooled-CP | 0.904 | 0.854 | 21.013 |
| Local only | QQ-CP | 0.900 | 0.866 | 20.802 |
| FedAvg | local-CP | 0.898 | 0.833 | 20.065 |
| FedAvg | pooled-CP | 0.899 | 0.874 | 19.738 |
| FedAvg | QQ-CP | 0.889 | 0.851 | 19.459 |
| FedProx | local-CP | 0.898 | 0.833 | 20.076 |
| FedProx | pooled-CP | 0.902 | 0.874 | 19.803 |
| FedProx | QQ-CP | 0.890 | 0.851 | 19.436 |
| FedAvg + fine-tune | local-CP | 0.897 | 0.829 | 20.118 |
| FedAvg + fine-tune | pooled-CP | 0.895 | 0.862 | 19.686 |
| FedAvg + fine-tune | QQ-CP | 0.892 | 0.858 | 19.554 |
| Centralised (pooled) | local-CP | 0.899 | 0.847 | 19.091 |
| Centralised (pooled) | pooled-CP | 0.888 | 0.847 | 18.534 |
| Centralised (pooled) | QQ-CP | 0.884 | 0.847 | 18.452 |

### Dataset B, shift

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.903 | 0.841 | 19.703 |
| Local only | pooled-CP | 0.896 | 0.835 | 19.226 |
| Local only | QQ-CP | 0.889 | 0.815 | 18.943 |
| FedAvg | local-CP | 0.899 | 0.829 | 20.046 |
| FedAvg | pooled-CP | 0.905 | 0.858 | 19.814 |
| FedAvg | QQ-CP | 0.899 | 0.858 | 19.632 |
| FedProx | local-CP | 0.903 | 0.841 | 19.888 |
| FedProx | pooled-CP | 0.909 | 0.866 | 19.606 |
| FedProx | QQ-CP | 0.912 | 0.874 | 19.738 |
| FedAvg + fine-tune | local-CP | 0.902 | 0.837 | 19.046 |
| FedAvg + fine-tune | pooled-CP | 0.904 | 0.867 | 18.557 |
| FedAvg + fine-tune | QQ-CP | 0.899 | 0.859 | 18.347 |
| Centralised (pooled) | local-CP | 0.911 | 0.850 | 18.773 |
| Centralised (pooled) | pooled-CP | 0.903 | 0.850 | 18.368 |
| Centralised (pooled) | QQ-CP | 0.910 | 0.858 | 18.777 |

### Dataset C, iid

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.903 | 0.817 | 1.002 |
| Local only | pooled-CP | 0.883 | 0.817 | 0.781 |
| Local only | QQ-CP | 0.884 | 0.822 | 0.762 |
| FedAvg | local-CP | 0.894 | 0.844 | 0.472 |
| FedAvg | pooled-CP | 0.893 | 0.844 | 0.412 |
| FedAvg | QQ-CP | 0.890 | 0.839 | 0.421 |
| FedProx | local-CP | 0.898 | 0.817 | 0.468 |
| FedProx | pooled-CP | 0.896 | 0.839 | 0.422 |
| FedProx | QQ-CP | 0.892 | 0.839 | 0.428 |
| FedAvg + fine-tune | local-CP | 0.898 | 0.850 | 0.501 |
| FedAvg + fine-tune | pooled-CP | 0.897 | 0.850 | 0.463 |
| FedAvg + fine-tune | QQ-CP | 0.890 | 0.833 | 0.472 |
| Centralised (pooled) | local-CP | 0.902 | 0.833 | 0.403 |
| Centralised (pooled) | pooled-CP | 0.896 | 0.856 | 0.383 |
| Centralised (pooled) | QQ-CP | 0.896 | 0.850 | 0.423 |

### Dataset C, shift

| Model | CP | mean coverage | worst-client coverage | mean bound (dB) |
|---|---|---|---|---|
| Local only | local-CP | 0.916 | 0.867 | 0.939 |
| Local only | pooled-CP | 0.916 | 0.850 | 0.750 |
| Local only | QQ-CP | 0.921 | 0.850 | 1.057 |
| FedAvg | local-CP | 0.904 | 0.833 | 2.097 |
| FedAvg | pooled-CP | 0.901 | 0.817 | 1.746 |
| FedAvg | QQ-CP | 0.958 | 0.922 | 2.982 |
| FedProx | local-CP | 0.904 | 0.844 | 2.026 |
| FedProx | pooled-CP | 0.903 | 0.828 | 1.749 |
| FedProx | QQ-CP | 0.958 | 0.922 | 2.859 |
| FedAvg + fine-tune | local-CP | 0.914 | 0.850 | 0.678 |
| FedAvg + fine-tune | pooled-CP | 0.922 | 0.817 | 0.527 |
| FedAvg + fine-tune | QQ-CP | 0.953 | 0.889 | 0.933 |
| Centralised (pooled) | local-CP | 0.916 | 0.844 | 0.466 |
| Centralised (pooled) | pooled-CP | 0.899 | 0.806 | 0.394 |
| Centralised (pooled) | QQ-CP | 0.940 | 0.867 | 0.560 |

## Conformal certificates on `dfr` (target 90 %), K = 5

Coverage per client then: mean, min over clients (mean over seeds); bound q.

### Dataset B, iid

| Model | CP | mean coverage | worst-client coverage | mean bound (GHz) |
|---|---|---|---|---|
| Local only | local-CP | 0.914 | 0.875 | 5.007 |
| Local only | pooled-CP | 0.905 | 0.866 | 4.910 |
| Local only | QQ-CP | 0.909 | 0.866 | 5.000 |
| FedAvg | local-CP | 0.933 | 0.886 | 4.813 |
| FedAvg | pooled-CP | 0.925 | 0.878 | 4.727 |
| FedAvg | QQ-CP | 0.935 | 0.886 | 4.910 |
| FedProx | local-CP | 0.929 | 0.878 | 4.812 |
| FedProx | pooled-CP | 0.922 | 0.866 | 4.730 |
| FedProx | QQ-CP | 0.929 | 0.878 | 4.850 |
| FedAvg + fine-tune | local-CP | 0.930 | 0.886 | 4.880 |
| FedAvg + fine-tune | pooled-CP | 0.922 | 0.870 | 4.817 |
| FedAvg + fine-tune | QQ-CP | 0.928 | 0.878 | 4.880 |
| Centralised (pooled) | local-CP | 0.910 | 0.874 | 5.916 |
| Centralised (pooled) | pooled-CP | 0.914 | 0.882 | 5.907 |
| Centralised (pooled) | QQ-CP | 0.908 | 0.874 | 5.820 |

### Dataset B, shift

| Model | CP | mean coverage | worst-client coverage | mean bound (GHz) |
|---|---|---|---|---|
| Local only | local-CP | 0.921 | 0.871 | 2.566 |
| Local only | pooled-CP | 0.920 | 0.687 | 3.486 |
| Local only | QQ-CP | 0.800 | 0.578 | 0.760 |
| FedAvg | local-CP | 0.906 | 0.866 | 4.352 |
| FedAvg | pooled-CP | 0.911 | 0.810 | 4.363 |
| FedAvg | QQ-CP | 0.856 | 0.733 | 3.787 |
| FedProx | local-CP | 0.912 | 0.871 | 4.566 |
| FedProx | pooled-CP | 0.911 | 0.778 | 4.610 |
| FedProx | QQ-CP | 0.870 | 0.722 | 4.000 |
| FedAvg + fine-tune | local-CP | 0.915 | 0.883 | 3.073 |
| FedAvg + fine-tune | pooled-CP | 0.909 | 0.811 | 3.270 |
| FedAvg + fine-tune | QQ-CP | 0.843 | 0.746 | 2.276 |
| Centralised (pooled) | local-CP | 0.920 | 0.854 | 5.395 |
| Centralised (pooled) | pooled-CP | 0.900 | 0.707 | 6.003 |
| Centralised (pooled) | QQ-CP | 0.734 | 0.488 | 4.213 |

### Dataset C, iid

| Model | CP | mean coverage | worst-client coverage | mean bound (GHz) |
|---|---|---|---|---|
| Local only | local-CP | 0.908 | 0.806 | 0.041 |
| Local only | pooled-CP | 0.909 | 0.872 | 0.038 |
| Local only | QQ-CP | 0.928 | 0.883 | 0.045 |
| FedAvg | local-CP | 0.896 | 0.817 | 0.038 |
| FedAvg | pooled-CP | 0.910 | 0.878 | 0.039 |
| FedAvg | QQ-CP | 0.940 | 0.917 | 0.045 |
| FedProx | local-CP | 0.912 | 0.856 | 0.038 |
| FedProx | pooled-CP | 0.911 | 0.861 | 0.037 |
| FedProx | QQ-CP | 0.934 | 0.900 | 0.041 |
| FedAvg + fine-tune | local-CP | 0.899 | 0.806 | 0.039 |
| FedAvg + fine-tune | pooled-CP | 0.902 | 0.861 | 0.038 |
| FedAvg + fine-tune | QQ-CP | 0.924 | 0.889 | 0.043 |
| Centralised (pooled) | local-CP | 0.911 | 0.828 | 0.037 |
| Centralised (pooled) | pooled-CP | 0.922 | 0.883 | 0.037 |
| Centralised (pooled) | QQ-CP | 0.932 | 0.894 | 0.041 |

### Dataset C, shift

| Model | CP | mean coverage | worst-client coverage | mean bound (GHz) |
|---|---|---|---|---|
| Local only | local-CP | 0.887 | 0.800 | 0.036 |
| Local only | pooled-CP | 0.883 | 0.822 | 0.034 |
| Local only | QQ-CP | 0.903 | 0.839 | 0.037 |
| FedAvg | local-CP | 0.913 | 0.861 | 0.060 |
| FedAvg | pooled-CP | 0.899 | 0.744 | 0.062 |
| FedAvg | QQ-CP | 0.964 | 0.883 | 0.076 |
| FedProx | local-CP | 0.894 | 0.844 | 0.054 |
| FedProx | pooled-CP | 0.897 | 0.717 | 0.057 |
| FedProx | QQ-CP | 0.960 | 0.867 | 0.073 |
| FedAvg + fine-tune | local-CP | 0.914 | 0.856 | 0.039 |
| FedAvg + fine-tune | pooled-CP | 0.914 | 0.856 | 0.038 |
| FedAvg + fine-tune | QQ-CP | 0.904 | 0.850 | 0.037 |
| Centralised (pooled) | local-CP | 0.901 | 0.817 | 0.034 |
| Centralised (pooled) | pooled-CP | 0.898 | 0.828 | 0.032 |
| Centralised (pooled) | QQ-CP | 0.902 | 0.817 | 0.034 |
