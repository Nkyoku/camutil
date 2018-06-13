// �����̒������v�Z����
static inline double length(const cv::Point2d &a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

// �_����`�̒��ɂ��邩���肷��
// px, py : �_�̍��W
// ax, ay : ��`�̂���pA�̍��W
// bx, by : A�Ƃ͑Ίp�ɂ���pB�̍��W
static inline bool isPointInRectangle(double px, double py, double ax, double ay, double bx, double by) {
    if (bx < ax) {
        std::swap(ax, bx);
    }
    if (by < ay) {
        std::swap(ay, by);
    }
    return (ax <= px) && (px <= bx) && (ay <= py) && (py <= by);
}

// �_����`�̒��ɂ��邩���肷��
// p : �_�̍��W
// a : ��`�̂���pA�̍��W
// b : A�Ƃ͑Ίp�ɂ���pB�̍��W
static inline bool isPointInRectangle(const cv::Point2d &p, const cv::Point2d &a, const cv::Point2d &b) {
    return isPointInRectangle(p.x, p.y, a.x, a.y, b.x, b.y);
}
