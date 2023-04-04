import lmdb
from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


lmdb_env = lmdb.open(r"C:\Users\bartek\GitHub\DualStyleGAN\data\anime2_tenth\lmdb")
lmdb_env = lmdb.open(r"C:\Users\bartek\GitHub\DualStyleGAN\data\met\lmdb")

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

for key, value in lmdb_cursor:
    print(f"{key=}")

img_bytes = lmdb_cursor.get(b'1024-00004')
buffer = BytesIO(img_bytes)
img = Image.open(buffer)
img.show()
# self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

# self.resolution = resolution
# self.transform = transform

# self.env = lmdb.open(
#     path,
#     max_readers=32,
#     readonly=True,
#     lock=False,
#     readahead=False,
#     meminit=False,
# )
# with self.env.begin(write=False) as txn:
#     key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
#     img_bytes = txn.get(key)

# buffer = BytesIO(img_bytes)
# img = Image.open(buffer)
# img = self.transform(img)
