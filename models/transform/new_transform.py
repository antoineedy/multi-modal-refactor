import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS


@TRANSFORMS.register_module()
class MultipleScales(BaseTransform):
    def __init__(self, direction: str):
        pass

    def transform(self, results: dict) -> dict:
        pass
