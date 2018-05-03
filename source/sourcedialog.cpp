#include "sourcedialog.h"
#include "ui_sourcedialog.h"
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QSettings>

SourceDialog::SourceDialog(QWidget *parent) : QDialog(parent) {
    m_ui = new Ui_SourceDialog;
    m_ui->setupUi(this);

    // UvcResolutionの選択項目が変更されたときにUvcFramerateを書き換える
    connect(m_ui->UvcResolution, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [&](int index) {
        QString fps = m_ui->UvcFramerate->currentText();
        m_ui->UvcFramerate->clear();
        switch (m_ui->UvcResolution->currentData().toInt()) {
        case kResolutionWvga:
            m_ui->UvcFramerate->addItem(tr("100 fps"), kFramerate100fps);
        case kResolution720p:
            m_ui->UvcFramerate->addItem(tr("60 fps"), kFramerate60fps);
        case kResolution1080p:
            m_ui->UvcFramerate->addItem(tr("30 fps"), kFramerate30fps);
        case kResolution2K2:
            m_ui->UvcFramerate->addItem(tr("15 fps"), kFramerate15fps);
        }
        m_ui->UvcFramerate->setCurrentIndex(m_ui->UvcFramerate->count() - 1);
        m_ui->UvcFramerate->setCurrentText(fps);
    });
    
    connect(m_ui->MoviePathOpen, QOverload<bool>::of(&QToolButton::clicked), this, &SourceDialog::selectMovieFile);
    connect(m_ui->SequencePathOpen, QOverload<bool>::of(&QToolButton::clicked), this, &SourceDialog::selectFirstSequenceFile);
    connect(m_ui->Buttons, &QDialogButtonBox::accepted, this, &SourceDialog::accept);
    connect(m_ui->Buttons, &QDialogButtonBox::rejected, this, &SourceDialog::reject);

    // UvcResolutionに解像度を追加する
    m_ui->UvcResolution->addItem(tr("2.2K (2x2208x1242)"), kResolution2K2);
    m_ui->UvcResolution->addItem(tr("1080p (2x1920x1080)"), kResolution1080p);
    m_ui->UvcResolution->addItem(tr("720p (2x1280x720)"), kResolution720p);
    m_ui->UvcResolution->addItem(tr("WVGA (2x672x376)"), kResolutionWvga);
}

void SourceDialog::restoreSettings(const QSettings &settings) {
    restoreGeometry(settings.value(tr("SourceWindow")).toByteArray());

    QString type = settings.value(tr("SourceType"), tr("UVC")).toString();
    if (type.compare(tr("UVC"), Qt::CaseInsensitive) == 0) {
        m_ui->Tab->setCurrentIndex(kUvcPageNumber);
    } else if (type.compare(tr("Movie"), Qt::CaseInsensitive) == 0) {
        m_ui->Tab->setCurrentIndex(kMoviePageNumber);
    } else if (type.compare(tr("Sequence"), Qt::CaseInsensitive) == 0) {
        m_ui->Tab->setCurrentIndex(kSequencePageNumber);
    } else {
        m_ui->Tab->setCurrentIndex(kUvcPageNumber);
    }

    m_ui->UvcId->setValue(settings.value(tr("UvcId"), 0).toInt());
    m_ui->UvcResolution->setCurrentIndex(0);
    m_ui->UvcResolution->setCurrentText(settings.value("UvcResolution").toString());
    m_ui->UvcFramerate->setCurrentText(settings.value("UvcFramerate").toString());

    m_ui->MoviePath->setText(settings.value(tr("MoviePath")).toString());

    m_ui->SequencePath->setText(settings.value(tr("SequencePath")).toString());
    m_ui->SequenceFramerate->setValue(settings.value(tr("SequenceFramerate")).toDouble());

    saveState();
}

void SourceDialog::saveSettings(QSettings &settings) const {
    settings.setValue(tr("SourceWindow"), saveGeometry());

    switch (sourceType()) {
    case VideoInput::kUvc:
        settings.setValue(tr("SourceType"), tr("UVC"));
        break;
    case VideoInput::kMovie:
        settings.setValue(tr("SourceType"), tr("Movie"));
        break;
    case VideoInput::kSequence:
        settings.setValue(tr("SourceType"), tr("Sequence"));
        break;
    default:
        Q_ASSERT(false);
    }

    settings.setValue(tr("UvcId"), m_ui->UvcId->value());
    settings.setValue(tr("UvcResolution"), m_ui->UvcResolution->currentText());
    settings.setValue(tr("UvcFramerate"), m_ui->UvcFramerate->currentText());

    settings.setValue(tr("MoviePath"), m_ui->MoviePath->text());

    settings.setValue(tr("SequencePath"), m_ui->SequencePath->text());
    settings.setValue(tr("SequenceFramerate"), m_ui->SequenceFramerate->value());
}

VideoInput::SourceType SourceDialog::sourceType(void) const {
    switch (m_ui->Tab->currentIndex()) {
    case kUvcPageNumber:
        return VideoInput::kUvc;
    case kMoviePageNumber:
        return VideoInput::kMovie;
    case kSequencePageNumber:
        return VideoInput::kSequence;
    default:
        Q_ASSERT(false);
        return VideoInput::kUvc;
    }
}

int SourceDialog::sourceId(void) const {
    if (sourceType() == VideoInput::kUvc) {
        return m_ui->UvcId->value();
    } else {
        return -1;
    }
}

QSize SourceDialog::sourceResolution(void) const {
    switch (sourceType()) {
    case VideoInput::kUvc:
        switch (m_ui->UvcResolution->currentData().toInt()) {
        case kResolution2K2:
            return QSize(4416, 1242);
        case kResolution1080p:
            return QSize(3840, 1080);
        case kResolution720p:
            return QSize(2560, 720);
        case kResolutionWvga:
            return QSize(1344, 376);
        default:
            Q_ASSERT(false);
            return QSize();
        }
    case VideoInput::kMovie:
        return QSize();
    case VideoInput::kSequence:
        return QSize();
    default:
        Q_ASSERT(false);
        return QSize();
    }
}

double SourceDialog::sourceFramerate(void) const {
    switch (sourceType()) {
    case VideoInput::kUvc:
        return m_ui->UvcFramerate->currentData().toInt();
    case VideoInput::kMovie:
        return 0;
    case VideoInput::kSequence:
        return m_ui->SequenceFramerate->value();
    default:
        Q_ASSERT(false);
        return 0;
    }
}

QString SourceDialog::sourcePath(void) const {
    switch (sourceType()) {
    case VideoInput::kUvc:
        return QString();
    case VideoInput::kMovie:
        return m_ui->MoviePath->text();
    case VideoInput::kSequence:
        return m_ui->SequencePath->text();
    default:
        Q_ASSERT(false);
        return QString();
    }
}

void SourceDialog::done(int r) {
    QDialog::done(r);
    if (r == QDialog::Accepted) {
        saveState();
    } else {
        discardState();
    }
}

void SourceDialog::selectMovieFile(void) {
    QFileInfo last_path(m_ui->MoviePath->text());
    QString selected_file = QFileDialog::getOpenFileName(this, tr("Select a movie file"), last_path.path(), tr(""));
    if (selected_file.isEmpty() == false) {
        m_ui->MoviePath->setText(selected_file);
    }
}

void SourceDialog::selectFirstSequenceFile(void) {
    QFileInfo last_path(m_ui->SequencePath->text());
    QString selected_file = QFileDialog::getOpenFileName(this, tr("Select a first picture file"), last_path.path(), tr(""));
    if (selected_file.isEmpty() == false) {
        m_ui->SequencePath->setText(selected_file);
    }
}

void SourceDialog::saveState(void) {
    m_State.Type = m_ui->Tab->currentIndex();
    m_State.UvcId = m_ui->UvcId->value();
    m_State.UvcResolution = m_ui->UvcResolution->currentIndex();
    m_State.UvcFramerate = m_ui->UvcFramerate->currentIndex();
    m_State.MoviePath = m_ui->MoviePath->text();
    m_State.SequencePath = m_ui->SequencePath->text();
    m_State.SequenceFramerate = m_ui->SequenceFramerate->value();
}

void SourceDialog::discardState(void) {
    m_ui->Tab->setCurrentIndex(m_State.Type);
    m_ui->UvcId->setValue(m_State.UvcId);
    m_ui->UvcResolution->setCurrentIndex(m_State.UvcResolution);
    m_ui->UvcFramerate->setCurrentIndex(m_State.UvcFramerate);
    m_ui->MoviePath->setText(m_State.MoviePath);
    m_ui->SequencePath->setText(m_State.SequencePath);
    m_ui->SequenceFramerate->setValue(m_State.SequenceFramerate);
}
