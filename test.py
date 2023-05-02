from multiprocessing import Pool


class Test:
    def __init__(self, data):
        self.data = data

    def printff(self, x):
        print(self.data)
        x.append(1)

    def test_something(self):
        with Pool(processes=2) as p:
            list(p.imap_unordered(self.printff, self.data))
        print(f"final {self.data}")


if __name__ == '__main__':
    t = Test([[1], [2], [3]])
    t.test_something()
