import numpy as np
import pandas as pd
import ember

X_train, y_train, X_test, y_test = ember.read_vectorized_features("/data/ember2018/")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
x = np.concatenate((X_train, X_test))
print(x.shape)
y = np.concatenate((y_train, y_test))
print(y.shape)

chunk_size = 10000
x_pos = []
x_neg = []

for i in range(0, len(y), chunk_size):
    chunk = x[i:i+chunk_size]
    chunk_y = y[i:i+chunk_size]
    
    x_pos_chunk = chunk[chunk_y == 1]
    print("{i}th pos chunk with shape={shape}".format(i=i, shape=x_pos_chunk.shape))
    x_neg_chunk = chunk[chunk_y == 0]
    print("{i}th neg chunk with shape={shape}".format(i=i, shape=x_neg_chunk.shape))

    # save x_pos_chunk and x_neg_chunk to csv file
    if i == 0:
        with open("/data/EMBER/ember_pos.csv", "w") as f:
            pd.DataFrame(x_pos_chunk).to_csv(f, header=False, index=False)
        with open("/data/EMBER/ember_neg.csv", "w") as f:
            pd.DataFrame(x_neg_chunk).to_csv(f, header=False, index=False)
    else:
        with open("/data/EMBER/ember_pos.csv", "a") as f:
            pd.DataFrame(x_pos_chunk).to_csv(f, header=False, index=False)
        with open("/data/EMBER/ember_neg.csv", "a") as f:
            pd.DataFrame(x_neg_chunk).to_csv(f, header=False, index=False)
