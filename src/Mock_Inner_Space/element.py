

class Element:
    def __init__(self, UNIcode: str, origin_point=(0,0,0), shape: list = None, location: tuple = None):
        self.code = UNIcode
        self.origin_point = origin_point
        self.shape = shape
        self.location = location
        self.colour = None
        self.real_shape = self.shape_point_transfer()

    def shape_point_transfer(self):
        real_shape = []
        for point in self.shape:
            real_shape.append(
                tuple( p-o+l for p,o,l in zip(point,self.origin_point,self.location))
            )

        return real_shape

    def output(self):
        return {
                'code': self.code,
                'shape': self.real_shape,
                'colour': self.colour
            }


if __name__ == '__main__':
    test = Element(UNIcode="0001",
            shape=[(1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)],
            location=(-1, 3, 1))

    print(test.output())
