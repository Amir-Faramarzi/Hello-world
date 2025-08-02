import numpy as np

from redness_tracker.model.algorithms import (
    DeltaProcessor,
    FlatFieldCorrector,
    MorphCleaner,
    RedMask,
    RunningMedian,
)


def test_red_mask() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [0, 0, 200]  # red pixel
    mask = RedMask()(frame)
    assert mask[0, 0]
    assert not mask[1, 1]


def test_flat_field_corrector() -> None:
    frame = np.full((2, 2, 3), 100, dtype=np.uint8)
    corrected = FlatFieldCorrector(blur_k=(1, 1))(frame)
    assert corrected.dtype == np.uint8


def test_running_median() -> None:
    rm = RunningMedian(size=3)
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
    for f in frames:
        median = rm.update(f)
    assert np.array_equal(median, np.full((2, 2, 3), 1, dtype=np.uint8))


def test_delta_processor() -> None:
    frame = np.full((1, 1, 3), 10, dtype=np.uint8)
    ref = np.zeros((1, 1, 3), dtype=np.uint8)
    delta = DeltaProcessor(threshold=5)(frame, ref)
    assert delta[0, 0, 0] == 10


def test_morph_cleaner() -> None:
    mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    cleaned = MorphCleaner(ksize=3, iterations=1)(mask)
    assert cleaned.dtype == bool
