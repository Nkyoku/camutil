// 線分の長さを計算する
static inline double length(const cv::Point2d &a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

// 点が矩形の中にあるか判定する
// px, py : 点の座標
// ax, ay : 矩形のある角Aの座標
// bx, by : Aとは対角にある角Bの座標
static inline bool isPointInRectangle(double px, double py, double ax, double ay, double bx, double by) {
    if (bx < ax) {
        std::swap(ax, bx);
    }
    if (by < ay) {
        std::swap(ay, by);
    }
    return (ax <= px) && (px <= bx) && (ay <= py) && (py <= by);
}

// 点が矩形の中にあるか判定する
// p : 点の座標
// a : 矩形のある角Aの座標
// b : Aとは対角にある角Bの座標
static inline bool isPointInRectangle(const cv::Point2d &p, const cv::Point2d &a, const cv::Point2d &b) {
    return isPointInRectangle(p.x, p.y, a.x, a.y, b.x, b.y);
}
