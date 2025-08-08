import math
from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter, QPainterPath, QTransform
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
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
    def __init__(self, params: ExcavatorParameters) -> None:
        super().__init__()
        self.params = params
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        # 스크롤바 비활성화
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 뷰 변환은 씬 크기에 맞춰 동적으로 적용
        # 총 길이 (작업 반경 추정)
        self.total_length = (
            self.params.boom_length_m
            + self.params.arm_length_m
            + self.params.bucket_length_m
        )
        # 뷰 내에서 굴착기를 위로 올리는 Y 오프셋 (m)
        self.view_offset_y_m = 1.0
        self._init_items()
        self._apply_fit_transform()

    def _init_items(self) -> None:
        self.scene.clear()
        # 작업 반경에 맞춘 씬 경계 (하단 여유를 더 확보)
        left_margin = -2.0
        bottom_margin = -2.0
        width = self.total_length + 4.0
        height = self.total_length + 3.0
        self.scene.setSceneRect(QRectF(left_margin, bottom_margin, width, height))

        # 지면선 (씬 전체 가로폭)
        ground_y = self.view_offset_y_m
        ground = self.scene.addLine(
            left_margin, ground_y, left_margin + width, ground_y, QPen(QColor("#888"), 0.02)
        )
        ground.setZValue(-1)

        # 스윙축(턴테이블) 원형 아이템 (원점)
        self.swing_radius_m = 0.45  # 약 0.9 m 직경
        swing_pen = QPen(QColor("#6a4c93"), 0.04, Qt.SolidLine, Qt.RoundCap)
        swing_brush = QBrush(QColor(106, 76, 147, 40))
        self.swing_axis: QGraphicsEllipseItem = self.scene.addEllipse(
            -self.swing_radius_m,
            self.view_offset_y_m - self.swing_radius_m,
            2 * self.swing_radius_m,
            2 * self.swing_radius_m,
            swing_pen,
            swing_brush,
        )
        self.swing_axis.setZValue(0)
        # 스윙축-붐축 연결용 짧은 축(라인)
        self.swing_shaft_len_m = 0.30
        self.swing_shaft = QGraphicsLineItem()
        self.swing_shaft.setPen(QPen(QColor("#6a4c93"), 0.06, Qt.SolidLine, Qt.RoundCap))
        self.scene.addItem(self.swing_shaft)
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
        # 관절 마커: 원형 아이템
        self.joint_pen = QPen(QColor("#e76f51"), 0.05, Qt.SolidLine, Qt.RoundCap)
        self.joint_brush = QBrush(QColor("#e76f51"))
        self.joint_radius_m = 0.10
        self.joint0 = QGraphicsEllipseItem()
        self.joint1 = QGraphicsEllipseItem()
        self.joint2 = QGraphicsEllipseItem()
        self.joint3 = QGraphicsEllipseItem()
        for j in (self.joint0, self.joint1, self.joint2, self.joint3):
            j.setPen(self.joint_pen)
            j.setBrush(self.joint_brush)
            self.scene.addItem(j)

        # 버킷 궤적 (QPainterPath 누적)
        self.trajectory_path = QPainterPath()
        traj_pen = QPen(QColor("#ff9500"), 0.03, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.trajectory_item = self.scene.addPath(self.trajectory_path, traj_pen)
        self.trajectory_item.setZValue(-0.2)
        self._last_tip_point: Optional[QPointF] = None

    def update_links(self, p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF) -> None:
        oy = self.view_offset_y_m
        self.link1.setLine(p0.x(), p0.y() + oy, p1.x(), p1.y() + oy)
        self.link2.setLine(p1.x(), p1.y() + oy, p2.x(), p2.y() + oy)
        self.link3.setLine(p2.x(), p2.y() + oy, p3.x(), p3.y() + oy)
        # 관절 원 업데이트
        r = self.joint_radius_m
        self.joint0.setRect(p0.x() - r, p0.y() + oy - r, 2 * r, 2 * r)
        self.joint1.setRect(p1.x() - r, p1.y() + oy - r, 2 * r, 2 * r)
        self.joint2.setRect(p2.x() - r, p2.y() + oy - r, 2 * r, 2 * r)
        self.joint3.setRect(p3.x() - r, p3.y() + oy - r, 2 * r, 2 * r)
        # 스윙축-붐축 연결용 짧은 축 업데이트 (p0->p1 방향으로 매우 짧게)
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length > 1e-6:
            ux = dx / length
            uy = dy / length
        else:
            ux, uy = 1.0, 0.0
        pbx = p0.x() + ux * self.swing_shaft_len_m
        pby = p0.y() + uy * self.swing_shaft_len_m
        self.swing_shaft.setLine(p0.x(), p0.y() + oy, pbx, pby + oy)

        # 버킷 궤적 업데이트
        tip_x = p3.x()
        tip_y = p3.y() + oy
        tip_point = QPointF(tip_x, tip_y)
        if self._last_tip_point is None:
            self.trajectory_path = QPainterPath(tip_point)
            self.trajectory_item.setPath(self.trajectory_path)
            self._last_tip_point = QPointF(tip_point)
        else:
            dxp = tip_point.x() - self._last_tip_point.x()
            dyp = tip_point.y() - self._last_tip_point.y()
            if (dxp * dxp + dyp * dyp) >= (0.005 * 0.005):  # 최소 5mm 이동 시만 추가 (연속성 강화)
                self.trajectory_path.lineTo(tip_point)
                self.trajectory_item.setPath(self.trajectory_path)
                self._last_tip_point = QPointF(tip_point)

    def reset_trajectory(self) -> None:
        self.trajectory_path = QPainterPath()
        self.trajectory_item.setPath(self.trajectory_path)
        self._last_tip_point = None

    def _apply_fit_transform(self) -> None:
        rect = self.scene.sceneRect()
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        if rect.width() <= 0 or rect.height() <= 0:
            return
        scale_x = vw / rect.width()
        scale_y = vh / rect.height()
        s = min(scale_x, scale_y) * 0.98  # 약간의 여백
        dx = (vw - s * rect.width()) / 2.0
        dy = (vh - s * rect.height()) / 2.0
        transform = QTransform(
            s,
            0.0,
            0.0,
            -s,
            dx - s * rect.left(),
            dy + s * rect.bottom(),
        )
        self.setTransform(transform)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_fit_transform()


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
        self.view = ExcavatorView(self.params)
        self.view.setMinimumHeight(420)
        layout.addWidget(self.view, 3)

        # Mode buttons: IK / FK
        mode_row = QHBoxLayout()
        self.ik_btn = QRadioButton("역기구학 (IK)")
        self.fk_btn = QRadioButton("기구학 (FK)")
        self.fk_btn.setChecked(True)
        # 궤적 리셋 버튼
        self.reset_btn = QPushButton("궤적 리셋")
        self.reset_btn.clicked.connect(self._on_reset_trajectory)
        # 샘플 궤적 자동 재생 버튼
        self.demo_btn = QPushButton("샘플 궤적")
        self.demo_btn.clicked.connect(self._on_toggle_demo)
        group = QButtonGroup(self)
        group.addButton(self.ik_btn, 0)
        group.addButton(self.fk_btn, 1)
        self.ik_btn.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.ik_btn)
        mode_row.addWidget(self.fk_btn)
        mode_row.addWidget(self.reset_btn)
        mode_row.addWidget(self.demo_btn)
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

    def _on_reset_trajectory(self) -> None:
        # 뷰의 궤적 초기화
        self.view.reset_trajectory()

    # ---------------- Demo (Sample Trajectory) ----------------
    def _on_toggle_demo(self) -> None:
        if getattr(self, "_demo_running", False):
            self._stop_demo()
        else:
            self._start_demo()

    def _start_demo(self) -> None:
        self._demo_running = True
        self.demo_btn.setText("샘플 정지")
        # 기존 궤적을 유지하여 연속적인 샘플 궤적을 그리게 함
        self._demo_t = 0
        if not hasattr(self, "_demo_timer"):
            self._demo_timer = QTimer(self)
            self._demo_timer.timeout.connect(self._demo_tick)
        self._demo_timer.start(16)  # ~60 FPS

    def _stop_demo(self) -> None:
        if getattr(self, "_demo_running", False):
            self._demo_running = False
            if hasattr(self, "_demo_timer"):
                self._demo_timer.stop()
        self.demo_btn.setText("샘플 궤적")

    def _demo_tick(self) -> None:
        # 시간 진행 및 경로 점 샘플링
        self._demo_t += 1
        t = self._demo_t / 90.0
        # 안전한 작업 반경 내 임의 궤적 (리사주/타원 혼합)
        r = (self.params.boom_length_m + self.params.arm_length_m) * 0.7
        cx = r * 0.55
        cy = r * 0.45 + 0.5
        x = cx + 0.45 * r * math.sin(1.1 * t) * math.cos(0.7 * t)
        y = max(0.15, cy + 0.35 * r * abs(math.sin(t)))
        theta = math.radians(12.0) * math.sin(0.6 * t)

        ik = self.model.inverse(x, y, theta, self.joints)
        if ik is not None:
            self.joints = ik
            self._update_view()
        # 종료 조건: 일정 시간 진행 후 자동 종료
        if self._demo_t > 1800:
            self._stop_demo()

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
