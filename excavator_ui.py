import math
from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# -----------------------------
# Excavator geometry parameters
# -----------------------------
@dataclass
class ExcavatorParameters:
    # Approximate link lengths for a 30-ton excavator (meters)
    boom_length_m: float = 6.15  # Boom
    arm_length_m: float = 3.20   # Arm (stick)
    bucket_length_m: float = 1.50  # Bucket effective tip length

    # Joint limits in degrees (boom, arm, bucket)
    boom_min_deg: float = 0.0
    boom_max_deg: float = 75.0
    arm_min_deg: float = -130.0
    arm_max_deg: float = 130.0
    bucket_min_deg: float = -120.0
    bucket_max_deg: float = 120.0


@dataclass
class JointAngles:
    boom_rad: float
    arm_rad: float
    bucket_rad: float


class ExcavatorKinematics:
    def __init__(self, params: ExcavatorParameters) -> None:
        self.params = params

    # Forward kinematics: return positions of joints and tip in base frame
    def forward(self, joints: JointAngles) -> Tuple[QPointF, QPointF, QPointF, QPointF]:
        l1 = self.params.boom_length_m
        l2 = self.params.arm_length_m
        l3 = self.params.bucket_length_m

        q1 = joints.boom_rad
        q2 = joints.arm_rad
        q3 = joints.bucket_rad

        p0 = QPointF(0.0, 0.0)
        p1 = QPointF(l1 * math.cos(q1), l1 * math.sin(q1))
        p2 = QPointF(
            p1.x() + l2 * math.cos(q1 + q2),
            p1.y() + l2 * math.sin(q1 + q2),
        )
        p3 = QPointF(
            p2.x() + l3 * math.cos(q1 + q2 + q3),
            p2.y() + l3 * math.sin(q1 + q2 + q3),
        )
        return p0, p1, p2, p3

    # Inverse kinematics: given end effector pose (x,y,theta), compute joint angles
    def inverse(
        self, x: float, y: float, theta_rad: float, current: JointAngles
    ) -> Optional[JointAngles]:
        l1 = self.params.boom_length_m
        l2 = self.params.arm_length_m
        l3 = self.params.bucket_length_m

        # Wrist position (end of arm/stick)
        wx = x - l3 * math.cos(theta_rad)
        wy = y - l3 * math.sin(theta_rad)

        r2 = wx * wx + wy * wy
        # Check reachability (numerical safety)
        min_reach = max(0.0, abs(l1 - l2))
        max_reach = l1 + l2
        r = math.sqrt(r2)
        if r < 1e-6:
            return None
        if r < min_reach - 1e-3 or r > max_reach + 1e-3:
            # Unreachable
            return None

        # Law of cosines for q2
        cos_q2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_q2 = max(-1.0, min(1.0, cos_q2))
        q2a = math.acos(cos_q2)  # elbow-up
        q2b = -q2a               # elbow-down

        # Corresponding q1 values
        def solve_q1(q2: float) -> float:
            k1 = l1 + l2 * math.cos(q2)
            k2 = l2 * math.sin(q2)
            return math.atan2(wy, wx) - math.atan2(k2, k1)

        candidates = []
        for q2 in (q2a, q2b):
            q1 = solve_q1(q2)
            q3 = theta_rad - (q1 + q2)
            cand = JointAngles(q1, q2, q3)
            if self._within_limits(cand):
                candidates.append(cand)

        if not candidates:
            # If none within limits, allow best-effort clamped bucket, prefer closer to current
            q1a = solve_q1(q2a)
            q3a = theta_rad - (q1a + q2a)
            q1b = solve_q1(q2b)
            q3b = theta_rad - (q1b + q2b)
            candidates = [JointAngles(q1a, q2a, q3a), JointAngles(q1b, q2b, q3b)]

        # Select solution closest to current joint configuration
        def dist(a: JointAngles, b: JointAngles) -> float:
            return (
                abs(a.boom_rad - b.boom_rad)
                + abs(a.arm_rad - b.arm_rad)
                + abs(a.bucket_rad - b.bucket_rad)
            )

        best = min(candidates, key=lambda c: dist(c, current))
        # Clamp to limits
        best = self._clamp_to_limits(best)
        return best

    def _within_limits(self, j: JointAngles) -> bool:
        p = self.params
        return (
            math.degrees(j.boom_rad) >= p.boom_min_deg
            and math.degrees(j.boom_rad) <= p.boom_max_deg
            and math.degrees(j.arm_rad) >= p.arm_min_deg
            and math.degrees(j.arm_rad) <= p.arm_max_deg
            and math.degrees(j.bucket_rad) >= p.bucket_min_deg
            and math.degrees(j.bucket_rad) <= p.bucket_max_deg
        )

    def _clamp_to_limits(self, j: JointAngles) -> JointAngles:
        p = self.params
        b = math.radians(
            max(p.boom_min_deg, min(p.boom_max_deg, math.degrees(j.boom_rad)))
        )
        a = math.radians(
            max(p.arm_min_deg, min(p.arm_max_deg, math.degrees(j.arm_rad)))
        )
        k = math.radians(
            max(p.bucket_min_deg, min(p.bucket_max_deg, math.degrees(j.bucket_rad)))
        )
        return JointAngles(b, a, k)


# -----------------------------
# UI Components
# -----------------------------
class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        minimum: float,
        maximum: float,
        step: float,
        unit: str,
        initial: float,
    ) -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(f"{title}")
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setValue(initial)
        self.unit_label = QLabel(unit)

        self.slider = QSlider(Qt.Horizontal)
        # map float to int slider
        self._scale = 1.0 / step if step > 0 else 100.0
        self.slider.setMinimum(int(minimum * self._scale))
        self.slider.setMaximum(int(maximum * self._scale))
        self.slider.setValue(int(initial * self._scale))

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.unit_label)

    def _on_slider(self, ival: int) -> None:
        fval = ival / self._scale
        if abs(self.spin.value() - fval) > 1e-9:
            self.spin.blockSignals(True)
            self.spin.setValue(fval)
            self.spin.blockSignals(False)
        self.valueChanged.emit(self.value())

    def _on_spin(self, fval: float) -> None:
        ival = int(round(fval * self._scale))
        if self.slider.value() != ival:
            self.slider.blockSignals(True)
            self.slider.setValue(ival)
            self.slider.blockSignals(False)
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, fval: float) -> None:
        self.spin.setValue(fval)


class ExcavatorView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.scale(60.0, -60.0)  # meters to pixels, invert Y
        self._init_items()

    def _init_items(self) -> None:
        self.scene.clear()
        # Ground line
        ground = self.scene.addLine(-10, 0, 20, 0, QPen(QColor("#888"), 0.02))
        ground.setZValue(-1)
        # Links
        self.link1 = QGraphicsLineItem()
        self.link2 = QGraphicsLineItem()
        self.link3 = QGraphicsLineItem()
        for link, color in [
            (self.link1, QColor("#2a9d8f")),
            (self.link2, QColor("#264653")),
            (self.link3, QColor("#e9c46a")),
        ]:
            link.setPen(QPen(color, 0.08, Qt.SolidLine, Qt.RoundCap))
            self.scene.addItem(link)
        # Joints as small circles (drawn using thick points via short lines)
        self.joint_pen = QPen(QColor("#e76f51"), 0.12, Qt.SolidLine, Qt.RoundCap)

    def update_links(self, p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF) -> None:
        self.link1.setLine(p0.x(), p0.y(), p1.x(), p1.y())
        self.link2.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        self.link3.setLine(p2.x(), p2.y(), p3.x(), p3.y())
        # Optional: add joint markers by short zero-length lines
        for pt in (p0, p1, p2, p3):
            self.scene.addLine(pt.x(), pt.y(), pt.x(), pt.y(), self.joint_pen)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Excavator IK/FK Demo (30t approx)")
        self.params = ExcavatorParameters()
        self.model = ExcavatorKinematics(self.params)
        # current joints
        self.joints = JointAngles(
            boom_rad=math.radians(35.0),
            arm_rad=math.radians(-20.0),
            bucket_rad=math.radians(10.0),
        )
        self._build_ui()
        self._update_view()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Top: Excavator view
        self.view = ExcavatorView()
        self.view.setMinimumHeight(420)
        layout.addWidget(self.view, 3)

        # Mode buttons: IK / FK
        mode_row = QHBoxLayout()
        self.ik_btn = QRadioButton("역기구학 (IK)")
        self.fk_btn = QRadioButton("기구학 (FK)")
        self.fk_btn.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.ik_btn, 0)
        group.addButton(self.fk_btn, 1)
        self.ik_btn.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.ik_btn)
        mode_row.addWidget(self.fk_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Controls area
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        layout.addWidget(self.controls_container, 2)

        self._build_fk_controls()  # default visible

    def _build_fk_controls(self) -> None:
        self._clear_controls()
        self.fk_boom = LabeledSlider("붐 (deg)", self.params.boom_min_deg, self.params.boom_max_deg, 0.5, "deg", math.degrees(self.joints.boom_rad))
        self.fk_arm = LabeledSlider("암 (deg)", self.params.arm_min_deg, self.params.arm_max_deg, 0.5, "deg", math.degrees(self.joints.arm_rad))
        self.fk_bucket = LabeledSlider("버킷 (deg)", self.params.bucket_min_deg, self.params.bucket_max_deg, 0.5, "deg", math.degrees(self.joints.bucket_rad))
        for w in (self.fk_boom, self.fk_arm, self.fk_bucket):
            w.valueChanged.connect(self.on_any_value_changed)
            self.controls_layout.addWidget(w)

    def _build_ik_controls(self) -> None:
        self._clear_controls()
        # Reasonable bounds for a 30t excavator working envelope (meters)
        max_reach = self.params.boom_length_m + self.params.arm_length_m + self.params.bucket_length_m
        self.ik_x = LabeledSlider("x (m)", -1.0, max_reach, 0.01, "m", self.end_pose()[0])
        self.ik_y = LabeledSlider("y (m)", 0.0, max_reach, 0.01, "m", self.end_pose()[1])
        self.ik_theta = LabeledSlider("theta (deg)", -180.0, 180.0, 0.5, "deg", math.degrees(self.end_pose()[2]))
        for w in (self.ik_x, self.ik_y, self.ik_theta):
            w.valueChanged.connect(self.on_any_value_changed)
            self.controls_layout.addWidget(w)

    def _clear_controls(self) -> None:
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _on_mode_changed(self) -> None:
        if self.ik_btn.isChecked():
            self._build_ik_controls()
        else:
            self._build_fk_controls()
        self._update_view()

    def on_any_value_changed(self) -> None:
        if self.ik_btn.isChecked():
            # IK mode: compute joints from pose
            x = self.ik_x.value()
            y = self.ik_y.value()
            theta = math.radians(self.ik_theta.value())
            ik = self.model.inverse(x, y, theta, self.joints)
            if ik is not None:
                self.joints = ik
            # If no solution, keep previous joints
        else:
            # FK mode: read joints
            b = math.radians(self.fk_boom.value())
            a = math.radians(self.fk_arm.value())
            k = math.radians(self.fk_bucket.value())
            self.joints = self.model._clamp_to_limits(JointAngles(b, a, k))
        self._update_view()

    def end_pose(self) -> Tuple[float, float, float]:
        p0, p1, p2, p3 = self.model.forward(self.joints)
        theta = self.joints.boom_rad + self.joints.arm_rad + self.joints.bucket_rad
        return (p3.x(), p3.y(), theta)

    def _update_view(self) -> None:
        p0, p1, p2, p3 = self.model.forward(self.joints)
        self.view.update_links(p0, p1, p2, p3)
        # Sync FK controls with current joints
        if not self.ik_btn.isChecked():
            self.fk_boom.set_value(math.degrees(self.joints.boom_rad))
            self.fk_arm.set_value(math.degrees(self.joints.arm_rad))
            self.fk_bucket.set_value(math.degrees(self.joints.bucket_rad))
        else:
            x, y, th = self.end_pose()
            self.ik_x.set_value(x)
            self.ik_y.set_value(y)
            self.ik_theta.set_value(math.degrees(th))


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.resize(1000, 800)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
