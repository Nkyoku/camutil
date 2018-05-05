#pragma once

#include <QObject>
#include <QSize>
#include <QMutex>
#include <opencv2/core.hpp>

namespace cv {
class VideoCapture;
}

class VideoInput : public QObject {
    Q_OBJECT

public:
    // ソースの種類
    enum SourceType {
        kUvc,
        kMovie,
        kSequence
    };

    // コンストラクタ
    explicit VideoInput(QObject *parent = nullptr);

    // デストラクタ
    ~VideoInput();

    // UVCカメラをソースとして開く
    bool openUvcCamera(int index, const QSize &resolution, double framerate);

    // 動画ファイルをソースとして開く
    bool openMovie(const QString &path_to_file);

    // 連番画像ファイルをソースとして開く
    bool openSequence(const QString &path_to_first_file, double framerate);

    // ソースを閉じる
    void close(void);

    // ソースが開かれているか取得する
    bool isOpened(void) const;

    // ソースの解像度を取得する
    QSize sourceResolution(void) const {
        return QSize(m_Width, m_Height);
    }

    // ソースのフレームレート
    double sourceFramerate(void) const {
        return m_Framerate;
    }

    // ソースのフレーム数を返す
    int sourceFrameCount(void) const {
        return m_FrameCount;
    }

    // 現在のフレーム番号を取得する
    int currentFrameNumber(void) const {
        return m_CurrentFrameNumber;
    }

    // 再生中か取得する
    bool isPlaying(void) const {
        return m_IsPlaying;
    }

    // 再生を始める・一時停止する
    Q_SLOT bool startOrPausePlaying(void);

    // 再生を始める
    Q_SLOT bool startPlaying(void);

    // 再生を一時停止する
    Q_SLOT void pausePlaying(void);

    // 再生を停止する
    Q_SLOT void stopPlaying(void);

    // 再生しているフレーム番号を変更する
    // 変更後のフレーム番号を返値として返す
    Q_SLOT int setFrameNumber(int frame_number);

    // シーク可能か取得する
    bool isSeekable(void) const;

    // 新たなフレームを読み込む
    bool readFrame(cv::Mat &frame);

    // フレーム番号が変化したときにときに送られるシグナル
    Q_SIGNAL void frameNumberChanged(int frame_number);

    // フレーム数が変化したときに送られるシグナル
    Q_SIGNAL void frameCountChanged(int frame_count);

private:
    // UIスレッドと計算スレッドからm_CvCaptureを操作するためのミューテックス
    QMutex m_Mutex;

    // 再生しているかどうかのフラグ
    bool m_IsPlaying = false;

    // ソースの種類
    SourceType m_Type = kUvc;

    // ソースの解像度
    int m_Width = 0, m_Height = 0;

    // ソースのフレームレート
    double m_Framerate = 0;

    // ソースのフレーム数
    int m_FrameCount = 0;

    // 現在のフレーム番号
    int m_CurrentFrameNumber = 0;

    // ソースを読み込むためのcv::VideoCapture
    cv::VideoCapture *m_CvCapture = nullptr;

    // ソースを開く共通処理
    bool openCommon(SourceType type);
};

//Q_DECLARE_METATYPE(cv::Mat)
