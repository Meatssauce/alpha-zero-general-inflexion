import unittest
from multiprocessing import Pool


class MyTestCase(unittest.TestCase):
    def printff(self, x):
        print(self.data)
        x.append(1)

    def test_something(self):
        self.data = [1, 2, 3]
        with Pool(processes=2) as p:
            list(p.imap_unordered(self.printff, self.data))


if __name__ == '__main__':
    unittest.main()
