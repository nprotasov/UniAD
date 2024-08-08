
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.structures.base_data_element import BaseDataElement as DC #TODO check that it's correct replace https://github.com/open-mmlab/mmcv/pull/2216#issuecomment-1754875721

from mmdet.registry import TRANSFORMS
from mmcv.transforms import to_tensor
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs #TODO not sure in it: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/notes/compatibility.md

@TRANSFORMS.register_module()
class CustomDefaultFormatBundle3D(Pack3DDetInputs):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results