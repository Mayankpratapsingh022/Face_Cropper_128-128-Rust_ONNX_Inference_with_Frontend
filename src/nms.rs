// src/nms.rs

/// Compute Intersection‐over‐Union between two boxes (x1, y1, x2, y2).
pub fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax1, ay1, ax2, ay2) = a;
    let (bx1, by1, bx2, by2) = b;

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = ((ax2 - ax1).max(0.0)) * ((ay2 - ay1).max(0.0));
    let area_b = ((bx2 - bx1).max(0.0)) * ((by2 - by1).max(0.0));
    let union_area = area_a + area_b - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}
