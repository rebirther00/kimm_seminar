import math
from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication,  # type: ignore
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


@dataclass
class TowerCraneParameters:
    jib_length_m: float = 60.0
    counter_jib_length_m: float = 15.0
    tower_height_m: float = 50.0

    trolley_min_m: float = 3.0
    trolley_max_m: float = 58.0

    slew_min_deg: float = -180.0
    slew_max_deg: float = 180.0

    hoist_min_m: float = 0.0
    hoist_max_m: float = 60.0


@dataclass
class CraneState:
    slew_rad: float    # psi
    trolley_m: float   # radius along jib
    hoist_m: float     # hook height above ground


class TowerCraneKinematics:
    def __init__(self, params: TowerCraneParameters) -> None:
        self.params = params

    def forward(self, state: CraneState) -> Tuple[QPointF, QPointF, QPointF, float]:
        # Plan view (top): origin at tower base; draw jib in XY plane
        psi = state.slew_rad
        r = state.trolley_m
        # End of jib
        jib_end = QPointF(
            self.params.jib_length_m * math.cos(psi),
            self.params.jib_length_m * math.sin(psi),
        )
        # Trolley position along jib
        trolley = QPointF(r * math.cos(psi), r * math.sin(psi))
        return QPointF(0.0, 0.0), trolley, jib_end, state.hoist_m

    def inverse(self, x: float, y: float, hoist_m: float, current: CraneState) -> Optional[CraneState]:
        r = math.hypot(x, y)
        if r < 1e-6:
            # At mast center; undefined slew. Keep current slew.
            psi = current.slew_rad
        else:
            psi = math.atan2(y, x)
        # Clamp trolley within bounds
        r_clamped = max(self.params.trolley_min_m, min(self.params.trolley_max_m, r))
        # Clamp slew within limits (wrap to [-pi, pi] first)
        psi = (psi + math.pi) % (2 * math.pi) - math.pi
        psi_deg = max(self.params.slew_min_deg, min(self.params.slew_max_deg, math.degrees(psi)))
        psi = math.radians(psi_deg)
        # Clamp hoist
        h = max(self.params.hoist_min_m, min(self.params.hoist_max_m, hoist_m))
        return CraneState(psi, r_clamped, h)


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

        self.label = QLabel(title)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setValue(initial)
        self.unit_label = QLabel(unit)

        self.slider = QSlider(Qt.Horizontal)
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


class CraneView(QGraphicsView):
    def __init__(self, params: TowerCraneParameters) -> None:
        super().__init__()
        self.params = params
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        # meters to pixels; Y up
        self.scale(8.0, -8.0)
        self._init_items()

    def _init_items(self) -> None:
        self.scene.clear()
        # Tower base marker (circle)
        self.tower_marker = QGraphicsEllipseItem(-0.6, -0.6, 1.2, 1.2)
        self.tower_marker.setPen(QPen(QColor("#5c677d"), 0.05))
        self.scene.addItem(self.tower_marker)
        # Jib and counter-jib lines
        self.jib = QGraphicsLineItem()
        self.counter_jib = QGraphicsLineItem()
        pen_jib = QPen(QColor("#006d77"), 0.15, Qt.SolidLine, Qt.RoundCap)
        pen_cj = QPen(QColor("#8338ec"), 0.10, Qt.SolidLine, Qt.RoundCap)
        self.jib.setPen(pen_jib)
        self.counter_jib.setPen(pen_cj)
        self.scene.addItem(self.jib)
        self.scene.addItem(self.counter_jib)
        # Trolley and hook
        self.trolley_marker = QGraphicsEllipseItem(-0.25, -0.25, 0.5, 0.5)
        self.trolley_marker.setPen(QPen(QColor("#e76f51"), 0.05))
        self.trolley_marker.setBrush(QColor("#e76f51"))
        self.scene.addItem(self.trolley_marker)
        # Text for hoist height
        self.hoist_text = QGraphicsTextItem()
        self.scene.addItem(self.hoist_text)
        self.hoist_text.setDefaultTextColor(QColor("#444"))

        # Ground grid (light)
        grid_pen = QPen(QColor(220, 220, 220), 0.02)
        for gx in range(-80, 81, 10):
            self.scene.addLine(gx, -80, gx, 80, grid_pen)
        for gy in range(-80, 81, 10):
            self.scene.addLine(-80, gy, 80, gy, grid_pen)

    def update_graphics(self, state: CraneState) -> None:
        psi = state.slew_rad
        L = self.params.jib_length_m
        Lc = self.params.counter_jib_length_m
        # Jib endpoints
        jib_end = QPointF(L * math.cos(psi), L * math.sin(psi))
        cj_end = QPointF(-Lc * math.cos(psi), -Lc * math.sin(psi))
        self.jib.setLine(0.0, 0.0, jib_end.x(), jib_end.y())
        self.counter_jib.setLine(0.0, 0.0, cj_end.x(), cj_end.y())
        # Trolley
        trolley = QPointF(state.trolley_m * math.cos(psi), state.trolley_m * math.sin(psi))
        self.trolley_marker.setPos(trolley)
        # Hoist label near trolley
        self.hoist_text.setPlainText(f"hoist: {state.hoist_m:.2f} m")
        self.hoist_text.setPos(trolley.x() + 1.0, trolley.y() + 1.0)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tower Crane IK/FK Demo")
        self.params = TowerCraneParameters()
        self.model = TowerCraneKinematics(self.params)
        self.state = CraneState(slew_rad=math.radians(15.0), trolley_m=20.0, hoist_m=20.0)
        self._build_ui()
        self._update_view()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Top view
        self.view = CraneView(self.params)
        self.view.setMinimumHeight(500)
        layout.addWidget(self.view, 3)

        # Mode
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

        # Controls container
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        layout.addWidget(self.controls_container, 2)

        self._build_fk_controls()

    def _build_fk_controls(self) -> None:
        self._clear_controls()
        self.fk_slew = LabeledSlider("slew (deg)", self.params.slew_min_deg, self.params.slew_max_deg, 1.0, "deg", math.degrees(self.state.slew_rad))
        self.fk_trolley = LabeledSlider("trolley (m)", self.params.trolley_min_m, self.params.trolley_max_m, 0.1, "m", self.state.trolley_m)
        self.fk_hoist = LabeledSlider("hoist (m)", self.params.hoist_min_m, self.params.hoist_max_m, 0.1, "m", self.state.hoist_m)
        for w in (self.fk_slew, self.fk_trolley, self.fk_hoist):
            w.valueChanged.connect(self.on_any_value_changed)
            self.controls_layout.addWidget(w)

    def _build_ik_controls(self) -> None:
        self._clear_controls()
        max_r = self.params.jib_length_m
        self.ik_x = LabeledSlider("x (m)", -max_r, max_r, 0.1, "m", self.hook_pose()[0])
        self.ik_y = LabeledSlider("y (m)", -max_r, max_r, 0.1, "m", self.hook_pose()[1])
        self.ik_h = LabeledSlider("hoist (m)", self.params.hoist_min_m, self.params.hoist_max_m, 0.1, "m", self.state.hoist_m)
        for w in (self.ik_x, self.ik_y, self.ik_h):
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
            x = self.ik_x.value()
            y = self.ik_y.value()
            h = self.ik_h.value()
            new_state = self.model.inverse(x, y, h, self.state)
            if new_state is not None:
                self.state = new_state
        else:
            psi = math.radians(self.fk_slew.value())
            r = self.fk_trolley.value()
            h = self.fk_hoist.value()
            # Clamp
            psi_deg = max(self.params.slew_min_deg, min(self.params.slew_max_deg, math.degrees(psi)))
            self.state = CraneState(math.radians(psi_deg), max(self.params.trolley_min_m, min(self.params.trolley_max_m, r)), max(self.params.hoist_min_m, min(self.params.hoist_max_m, h)))
        self._update_view()

    def hook_pose(self) -> Tuple[float, float]:
        _, trolley, _, _ = self.model.forward(self.state)
        return (trolley.x(), trolley.y())

    def _update_view(self) -> None:
        p0, trolley, jib_end, h = self.model.forward(self.state)
        self.view.update_graphics(self.state)
        # Sync controls
        if self.ik_btn.isChecked():
            x, y = self.hook_pose()
            self.ik_x.set_value(x)
            self.ik_y.set_value(y)
            self.ik_h.set_value(self.state.hoist_m)
        else:
            self.fk_slew.set_value(math.degrees(self.state.slew_rad))
            self.fk_trolley.set_value(self.state.trolley_m)
            self.fk_hoist.set_value(self.state.hoist_m)


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.resize(1000, 820)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
