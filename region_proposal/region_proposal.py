from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

import selectivesearch
import numpy as np

BoundingBox = namedtuple('BoundingBox', ["x_tl", "y_tl", "x_br", "y_br"])

@dataclass
class RegionProposal:
    scale: int = 900
    sigma: float = 1.2
    min_size: int = 300

    def get_region_proposals(self, image: np.ndarray) -> List[BoundingBox]:
        _, result = selectivesearch.selective_search(image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)

        proposed_regions: List[Tuple[int, int, int, int]] = [tuple(*region["rect"]) for region in result.values()]

        boxes = [self._format_box(proposed_region) for proposed_region in proposed_regions]

        return boxes

    def _format_box(self, box) -> BoundingBox:
        min_x, min_y, width, height = box
        top_left = (min_x, min_y + height)
        bottom_right = (min_x + width, height)
        
        return BoundingBox(*top_left, *bottom_right)