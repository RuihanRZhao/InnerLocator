import json
import sys, time

from element import Element


class Room:
    def __init__(self,
                 shape: list = None,
                 stuff: list[Element] = None
                 ):
        self.shape = shape
        self.stuff = stuff

    def colour_apply(self):
        # generate colors for each demo stuff
        colors = (
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.0, 1.0),  # Magenta
            (1.0, 0.5, 0.0),  # Orange
            (0.5, 0.0, 0.5)  # Purple
        )
        print("Applying colours for stuff in the room ...    0.00%", end="")

        for number, element in enumerate(self.stuff):
            # time.sleep(1)
            element.colour = colors[number % 8]
            sys.stdout.write("\b\b\b\b\b\b")
            sys.stdout.flush()
            sys.stdout.write(f"{((number+1) / len(self.stuff) * 100):5.2f}%")
            sys.stdout.flush()

        print()
        print("Done.")

    def output(self, style):
        Output = {
            "shape": self.shape,
            "stuff": [i.output() for i in self.stuff]
        }
        if style == "JSON":
            return json.dumps(Output, indent=4)
        else:
            return None


if __name__ == "__main__":
    test = Room(
        shape=[(5, -5, -5), (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)],
        stuff=[
            Element(UNIcode="0001", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=(-1,  3,  1)),
            Element(UNIcode="0002", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=(-1, -3,  1)),
            Element(UNIcode="0003", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=(-1, -3,  2)),
            Element(UNIcode="0004", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=(-2, -3,  2)),
            Element(UNIcode="0005", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=(-2,  3,  2)),
            Element(UNIcode="0006", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=( 2,  3,  4)),
            Element(UNIcode="0007", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=( 1,  2,  4)),
            Element(UNIcode="0008", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=( 1,  2,  4)),
            Element(UNIcode="0009", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=( 1,  2,  4)),
            Element(UNIcode="0010", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=( 1,  6,  1)),
            Element(UNIcode="0011", shape=[(1, -1, -1),(1, 1, -1),(-1, 1, -1),(-1, -1, -1),(1, -1, 1),(1, 1, 1),(-1, -1, 1),(-1, 1, 1)], location=( 0,  2,  1))
        ]
    )
    test.colour_apply()
    print(test.output("JSON"))
