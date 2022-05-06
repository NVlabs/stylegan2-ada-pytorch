from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

import selectivesearch
import numpy as np

@dataclass
class RegionProposal:
    scale: int = 900
    sigma: float = 1.2
    min_size: int = 300

    def get_region_proposals(self, image: np.ndarray) -> List[Tuple]:
        _, result = selectivesearch.selective_search(image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)

        # Only return the 5 largest boxes to limit run time
        result.sort(reverse=True, key=lambda x: x['size'])
        proposed_regions: List[Tuple[int, int, int, int]] = [tuple(region["rect"]) for region in result[:5]]

        boxes = [self._format_box(proposed_region) for proposed_region in proposed_regions]

        return boxes

    def _format_box(self, box) -> Tuple:
        min_x, min_y, width, height = box
        top_left = (min_x, min_y + height)
        bottom_right = (min_x + width, min_y)
        
        return tuple([*top_left, *bottom_right])