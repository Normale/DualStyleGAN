from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        lmdb_txn = self.env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        formatted = f'{self.resolution}-{str(index).zfill(5)}'
        # print("FORMATTED =================================", formatted)
        img_bytes = lmdb_cursor.get(formatted.encode())
        # img_bytes = lmdb_cursor.get(b'512-30251')
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        # print("IMGSHAPE =================================", img.size())
        # with self.env.begin(write=False) as txn:
        #     key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
        #     img_bytes = txn.get(key)

        # buffer = BytesIO(img_bytes)
        # img = Image.open(buffer)
        # img = self.transform(img)

        return img
