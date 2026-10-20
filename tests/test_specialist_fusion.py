"""Behavioral tests for optional Sam specialist in the existing website."""
from backend.specialist_fusion import refine_with_sam

def det(label, confidence, box=(0.1, 0.1, 0.6, 0.6)):
    return {'label': label, 'confidence': confidence, 'bbox': list(box), 'class_id': 0}

def test_compatible_resistor_specific_visual_estimate():
    result = refine_with_sam([det('Resistor', .92)], [det('RESISTOR 10K', .87)])
    assert len(result) == 1
    assert result[0]['label'] == 'Resistor 10K'
    assert result[0]['source_model'] == 'sam_yolo11s'
    assert result[0]['review_status'] == 'specialist_visual_estimate'
    assert result[0]['core_confidence'] == .92

def test_opposed_component_cannot_replace_core():
    result = refine_with_sam([det('LED', .83)], [det('RESISTOR 2K', .99)])
    assert len(result) == 1 and result[0]['label'] == 'LED'

def test_conflicting_value_predictions_are_not_asserted():
    result = refine_with_sam([det('Resistor', .81)],
                             [det('RESISTOR 2K', .93),det('RESISTOR 10K', .91)])
    assert len(result) == 1
    assert result[0]['label'] == 'Resistor'
    assert result[0]['review_status'] == 'specialist_conflict'

def test_unmatched_details_are_not_claimed():
    assert refine_with_sam([], [det('RESISTOR 52K', .98)]) == []

def test_unseen_generic_object_needs_high_confidence_and_no_overlap():
    result = refine_with_sam([det('Resistor', .72)],
                             [det('BUZZER', .91, (.68, .67, .91, .92))])
    assert len(result) == 2 and result[0]['label'] == 'Buzzer'
    result = refine_with_sam([det('Resistor', .72)], [det('BUZZER', .91)])
    assert len(result) == 1

def test_low_confidence_specific_does_not_replace_broad():
    result = refine_with_sam([det('LED', .86)], [det('LED (GREEN)', .56)])
    assert result[0]['label'] == 'LED'
