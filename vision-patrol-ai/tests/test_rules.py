import pytest

from vision_patrol.rules import center_in_zone, evaluate_detection


def test_center_in_zone_true():
    assert center_in_zone([40, 40, 60, 60], [0.25, 0.25, 0.75, 0.75], 100, 100)


def test_center_in_zone_false():
    assert not center_in_zone([0, 0, 10, 10], [0.25, 0.25, 0.75, 0.75], 100, 100)


def test_no_zone_returns_false():
    assert not center_in_zone([1, 1, 5, 5], None, 10, 10)


def test_invalid_frame_size():
    with pytest.raises(ValueError):
        center_in_zone([1, 1, 5, 5], [0, 0, 1, 1], 0, 10)


def test_alert_only_for_configured_class_in_zone():
    detection = {"class_name": "backpack", "confidence": 0.9, "bbox": [40, 40, 60, 60]}
    result = evaluate_detection(detection, [0.25, 0.25, 0.75, 0.75], {"backpack"}, 100, 100)
    assert result["in_zone"] is True
    assert result["alert"] is True


def test_unconfigured_class_does_not_alert():
    detection = {"class_name": "person", "confidence": 0.9, "bbox": [40, 40, 60, 60]}
    result = evaluate_detection(detection, [0.25, 0.25, 0.75, 0.75], {"backpack"}, 100, 100)
    assert result["in_zone"] is True
    assert result["alert"] is False
