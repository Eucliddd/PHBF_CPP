from filters import HPBF
from data_loader import load_mnist

delta = 32
input_dim = 784 # d
sample_factor = 100 # s
bytes_per_element = 0.1

x_train, _, x_test, _ = load_mnist([0, 1, 4]) # The sets X and Y of paper. Any numpy array of shape (n, d)

bitarray_size = int(x_train.shape[0] * bytes_per_element * 8) # m
hash_count = bitarray_size // delta
print(f"bitarray_size = {bitarray_size}")
print(f"hash_count = {hash_count}")
hpbf = HPBF(
            bitarray_size,
            hash_count,
            input_dim,
            sample_factor=sample_factor,
        )

hpbf.initialize(x_train, x_test) # select the vectors
hpbf.bulk_add(x_train) # compute hashes and populate the filter
fpr = hpbf.compute_fpr(x_test)
print(fpr)