class Rectangle:
    x = None
    y = None
    width = None
    height = None
    style = "fill:white;stroke:black;stroke-width:5;fill-opacity:0;stroke-opacity:1"

    def toHtml(self):
        return r'<rect x="' + str(
            self.x) + '" y="' + str(
            self.y) + '" width="' + str(
            self.width) + '" height="' + str(
            self.height) + '" style="' + str(self.style) + '"/>'


class Diamond:
    points = None
    stype = "fill:white;stroke:#000000;stroke-width:5"

    def toHtml(self):
        return r'<polygon points="' + str(self.points2str()) + '"style="fill:white;stroke:#000000;stroke-width:5"/>'

    def points2str(self):
        result = ""
        for x, y in self.points:
            result += str(x) + "," + str(y) + " "
        return result
