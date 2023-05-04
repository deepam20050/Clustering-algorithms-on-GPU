import pandas as pd
import matplotlib.pyplot as plt

s = "meanshift-"

for i in range(0, 9):
    data = s + str(i) + ".csv"
    print(data)
    df = pd.read_csv(data)
    plt.scatter(x = df['x'], y = df['y'], c = df['label'])
    plt.plot()
    plt.show()
    # plt.savefig(data[0:-4] + ".jpg")
    # plt.close()