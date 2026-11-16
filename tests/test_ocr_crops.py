from anpr.ocr.crops import yolo_to_xyxy


def test_box_is_clipped_to_image():
    assert yolo_to_xyxy(0.5, 0.5, 1.0, 1.0, 100, 50, pad=0.1) == (0, 0, 100, 50)


def test_centre_box_pixels():
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.2, 100, 100) == (40, 40, 60, 60)
