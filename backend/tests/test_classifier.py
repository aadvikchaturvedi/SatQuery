import numpy as np
import pytest

from app.agent.classifier import classify
from app.exceptions import ValidationFailed
from app.schemas import TaskType
from app.types import InputBundle, Modality, SceneImage, SourceKind


def _scene(modality=Modality.OPTICAL, hw=(4, 4)):
    return SceneImage(
        modality=modality, source_kind=SourceKind.GEOTIFF_BANDS, hw=hw,
        array=np.zeros((13, *hw), dtype=np.float32),
        rgb_preview=np.zeros((*hw, 3), dtype=np.uint8),
    )


def test_bitemporal_optical_pair_is_change_vqa():
    bundle = InputBundle(query="What changed?", optical_t1=_scene(), optical_t2=_scene())
    assert classify(bundle) == TaskType.CHANGE_VQA


def test_bitemporal_sar_pair_is_change_vqa():
    bundle = InputBundle(
        query="Has the built-up area increased?",
        sar_t1=_scene(Modality.SAR), sar_t2=_scene(Modality.SAR),
    )
    assert classify(bundle) == TaskType.CHANGE_VQA


def test_full_bitemporal_cross_modal_is_still_change_vqa_not_fusion():
    """Fusion mode is handled *inside* ChangeVqaTool when all four images are
    given; the classifier must route there, not to CrossModalFusionTool
    (which is single-timestep only)."""
    bundle = InputBundle(
        query="Use optical and SAR together across both dates.",
        optical_t1=_scene(), optical_t2=_scene(),
        sar_t1=_scene(Modality.SAR), sar_t2=_scene(Modality.SAR),
    )
    assert classify(bundle) == TaskType.CHANGE_VQA


def test_single_time_cross_modal_pair_is_fusion():
    bundle = InputBundle(
        query="Use the optical and SAR images together to identify built-up regions.",
        optical_t1=_scene(), sar_t1=_scene(Modality.SAR),
    )
    assert classify(bundle) == TaskType.CROSS_MODAL_FUSION


def test_single_image_question_is_vqa():
    bundle = InputBundle(query="Is there a water body visible in this image?", optical_t1=_scene())
    assert classify(bundle) == TaskType.VQA


def test_single_image_describe_request_is_captioning():
    bundle = InputBundle(
        query="Describe the land-cover and major objects visible in this image.",
        optical_t1=_scene(),
    )
    assert classify(bundle) == TaskType.CAPTIONING


def test_no_images_raises_validation_error():
    with pytest.raises(ValidationFailed):
        classify(InputBundle(query="anything"))
