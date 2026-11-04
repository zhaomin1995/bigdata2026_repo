"""
This filter uses CRAFT (Baek et al., 2019) to detect the text in the image.
To use it, please install PyTorch-based CRAFT before using this filter.
"pip install craft-text-detector"

Reference:
    https://github.com/fcakyon/craft-text-detector
    https://arxiv.org/pdf/1904.01941.pdf
"""

import torch

from craft_text_detector import Craft


class ImageFilter:

    def __init__(self, output_dir='temp_output/', crop_type='poly', min_boxes=5, min_boxes_percent=0.85):

        # the parameters for CRAFT
        self.output_dir = output_dir
        self.crop_type = crop_type
        self.cuda = True if torch.cuda.is_available() else False
        self.craft = Craft(output_dir=self.output_dir, crop_type=self.crop_type, cuda=self.cuda)

        # the parameters for self-defined rules
        self.min_boxes = min_boxes
        self.min_boxes_percent = min_boxes_percent

    def check_boxes(self, boxes):
        """Determine if the image is text-style based on the box information.

        We use the number of horizontally-placed rectangle boxes to identify the text-style image,
        because these boxes are more likely where formal text place.
        The horizontally-placed rectangle boxes need to satisfy these constraints:
        1. the top edge and bottom edge are parallel and placed nearly horizontally, and
        2. the widths of top edge and bottom edge should be close to each other.

        Args:
            The coordinates of all boxes.
        Returns:
            True: the image is a text-style image
            False: the image is not a text-style image
        """
        # if the number of boxes is too small, then it cannot be the text-style image
        if len(boxes) < self.min_boxes:
            return False

        # check how many horizontally-placed rectangle boxes in the image
        rectangles = 0
        for box in boxes:

            # get the coordinates of the vertices of the box
            top_left_x, top_left_y = box[0]
            top_right_x, top_right_y = box[1]
            bottom_left_x, bottom_left_y = box[2]
            bottom_right_x, bottom_right_y = box[3]

            # check if the box is horizontally put in the image
            top_edge_height_difference = abs(top_left_y - top_right_y)
            bottom_edge_height_difference = abs(bottom_left_y - bottom_right_y)
            top_edge_horizontal = True if top_edge_height_difference <= 0.02 else False
            bottom_edge_horizontal = True if bottom_edge_height_difference <= 0.02 else False

            if top_edge_horizontal and bottom_edge_horizontal:
                rectangles += 1

        # if most boxes are this kind of rectangle, then this image is a text-style image
        if rectangles / len(boxes) >= self.min_boxes_percent:
            return True
        else:
            return False

    def detect_text(self, image_filepath):
        """Detect the text in the image.

        Use CRAFT to detect text in the image.
        Apply the box filters to identify the text-style image.

        Args:
            The path to the image file.
        Returns:
            True: the image is a text-style image
            False: the image is not a text-style image
        """
        prediction = self.craft.detect_text(image_filepath)
        boxes = prediction['boxes_as_ratios']
        return self.check_boxes(boxes)

    def unload_model(self):
        """Remove the pre-trained model from the GPU and memory.

        Clear the GPU cache and free the memory.

        Args:
            None.
        Return:
            None.
        """
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()
