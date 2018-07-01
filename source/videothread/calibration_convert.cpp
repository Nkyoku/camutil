#include "calibration_convert.h"
#include "ui_calibration_convert.h"

CalibrationConvertDialog::CalibrationConvertDialog(QWidget *parent)
    : QDialog(parent)
{
    m_ui = new Ui_CalibrationConvert;
    m_ui->setupUi(this);

    m_ui->TargetWidth->setMinimum(1);
    m_ui->TargetHeight->setMinimum(1);

    connect(m_ui->TargetWidth, QOverload<int>::of(&QSpinBox::valueChanged), [&](int value) {
        m_ui->TargetAspectRatio->setText(calculateAspectRatio(value, m_ui->TargetHeight->value()));
    });
    connect(m_ui->TargetHeight, QOverload<int>::of(&QSpinBox::valueChanged), [&](int value) {
        m_ui->TargetAspectRatio->setText(calculateAspectRatio(m_ui->TargetWidth->value(), value));
    });
    connect(m_ui->Buttons, &QDialogButtonBox::accepted, [&](void) {
        emit savePushed(m_ui->TargetWidth->value(), m_ui->TargetHeight->value());
        accept();
    });
    connect(m_ui->Buttons, &QDialogButtonBox::rejected, this, &CalibrationConvertDialog::reject);
}

void CalibrationConvertDialog::setCurrentResolution(int width, int height) {
    m_ui->SourceWidth->setText(QString::number(width));
    m_ui->SourceHeight->setText(QString::number(height));
    m_ui->SourceAspectRatio->setText(calculateAspectRatio(width, height));
    m_ui->TargetWidth->setMaximum(width);
    m_ui->TargetHeight->setMaximum(height);
}

QSize CalibrationConvertDialog::targetResolution(void) const {
    return QSize(m_ui->TargetWidth->value(), m_ui->TargetHeight->value());
}

QString CalibrationConvertDialog::calculateAspectRatio(int width, int height) {
    int a = std::max(width, height);
    int b = std::min(width, height);
    while (int r = a % b) {
        a = b;
        b = r;
    }
    return QString("%1:%2").arg(width / b).arg(height / b);
}
