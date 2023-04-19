from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

j = 10**4

for i in range(0, 9):
  if (i <= 6) :
    '''
    For generating K-Means datasets
    '''
    X, y = make_blobs(n_samples=10**i + 200,center_box=(-20, 20) ,centers=i + 1, random_state=42)
    df_kmeans = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1]})
    plt.scatter(df_kmeans['x'], df_kmeans['y'])
    # plt.show()
    df_kmeans.to_csv(str(i) + ".csv", index=False, header=False)
  else:
    '''
    For generating DBSCAN datasets
    '''
    X, y = make_circles(n_samples=j, noise=0.05, factor=0.5, random_state=42)
    df_dbscan = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1]})
    plt.scatter(df_dbscan['x'], df_dbscan['y'])
    # plt.show()
    df_dbscan.to_csv(str(i) + ".csv", index=False, header=False)
    j = j * 10