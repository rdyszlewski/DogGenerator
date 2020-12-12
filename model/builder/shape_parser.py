from model.builder.shared import LayersPattern


class ShapeParser:


    @staticmethod
    def parse(shape_text):
        if type(shape_text) == int:
            return shape_text
        if "(" in shape_text:
            value = LayersPattern.getParameters(shape_text).split(",")
            return int(value[0]), int(value[1]), int(value[2])