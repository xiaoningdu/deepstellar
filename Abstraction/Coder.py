

class Coder(object):

    def __init__(self, bits, dim):
        """
        :param bits: each dimension is encoded with how many bits
        :param dim: how many dimensions of the vectors
        """
        self.bits = bits
        self.dim = dim
        assert self.bits * self.dim <= 64

    def encode(self, vec):
        assert len(vec) == self.dim
        d = 0
        for i in range(self.dim):
            di = vec[i] << (self.bits * i)
            d = d | di
        return d

    def decode(self, d):
        mask = 2 ** self.bits - 1
        vec = []
        for i in range(self.dim):
            di = d & mask
            d = d >> self.bits
            vec.append(di)
        return vec


if __name__ == '__main__':
    coder = Coder(8, 3)
    en = coder.encode([1, 2, 3])
    print(format(en, '02x'))
    de = coder.decode(en)
    print(de)
