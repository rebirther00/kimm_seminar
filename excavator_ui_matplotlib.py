import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use('QtAgg')  # Qt Agg 백엔드 사용 (PySide6 호환)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.patches as patches

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
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
    bucket_length_m: float = 1.90  # Bucket effective tip length

    # Joint limits in degrees (boom, arm, bucket)
    boom_min_deg: float = -90.0
    boom_max_deg: float = 90.0
    arm_min_deg: float = -156.0
    arm_max_deg: float = -34.0
    bucket_min_deg: float = -133.0
    bucket_max_deg: float = 43.0


@dataclass
class JointAngles:
    boom_rad: float
    arm_rad: float
    bucket_rad: float


class ExcavatorKinematics:
    def __init__(self, params: ExcavatorParameters) -> None:
        self.params = params

    # Forward kinematics: return positions of joints and tip in base frame
    def forward(self, joints: JointAngles) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        l1 = self.params.boom_length_m
        l2 = self.params.arm_length_m
        l3 = self.params.bucket_length_m

        q1 = joints.boom_rad
        q2 = joints.arm_rad
        q3 = joints.bucket_rad

        p0 = (0.0, 0.0)
        p1 = (l1 * math.cos(q1), l1 * math.sin(q1))
        p2 = (
            p1[0] + l2 * math.cos(q1 + q2),
            p1[1] + l2 * math.sin(q1 + q2),
        )
        p3 = (
            p2[0] + l3 * math.cos(q1 + q2 + q3),
            p2[1] + l3 * math.sin(q1 + q2 + q3),
        )
        return p0, p1, p2, p3

    # Inverse kinematics: given end effector pose (x,z,theta), compute joint angles
    def inverse(
        self, x: float, z: float, theta_rad: float, current: JointAngles
    ) -> Optional[JointAngles]:
        l1 = self.params.boom_length_m
        l2 = self.params.arm_length_m
        l3 = self.params.bucket_length_m

        # Wrist position (end of arm/stick)
        wx = x - l3 * math.cos(theta_rad)
        wz = z - l3 * math.sin(theta_rad)

        r2 = wx * wx + wz * wz
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
            return math.atan2(wz, wx) - math.atan2(k2, k1)

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
    valueChanged = Signal(float)

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
    
    def set_locked(self, locked: bool) -> None:
        """슬라이더와 스핀박스 잠금/해제"""
        self.slider.setEnabled(not locked)
        self.spin.setEnabled(not locked)
        if locked:
            self.label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.label.setStyleSheet("")


class ExcavatorMatplotlibView(FigureCanvas):
    def __init__(self, params: ExcavatorParameters) -> None:
        self.figure = Figure(figsize=(10, 8), dpi=100)
        super().__init__(self.figure)
        
        self.params = params
        self.ax = self.figure.add_subplot(111)
        
        # 총 길이 (작업 반경 추정)
        self.total_length = (
            self.params.boom_length_m
            + self.params.arm_length_m
            + self.params.bucket_length_m
        )
        
        # 뷰 설정
        self.view_offset_y_m = 1.0
        self.trajectory_points = []
        
        self._setup_plot()
        
    def _setup_plot(self) -> None:
        """matplotlib 플롯 기본 설정"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 축 범위 설정
        margin = 2.0
        self.ax.set_xlim(-margin, self.total_length + margin)
        self.ax.set_ylim(-margin, self.total_length + margin)
        
        # 지면선
        ground_y = self.view_offset_y_m
        self.ax.axhline(y=ground_y, color='gray', linewidth=2, alpha=0.7)
        
        # 좌표축 라벨
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('굴착기 시뮬레이션 (Matplotlib)')
        
        # 0.5m 간격 그리드
        self.ax.grid(True, which='major', alpha=0.5)
        self.ax.grid(True, which='minor', alpha=0.2)
        self.ax.minorticks_on()
        
        # 스윙축(원점) 표시
        swing_circle = Circle((0, self.view_offset_y_m), 0.45, 
                            fill=True, facecolor='purple', alpha=0.3, 
                            edgecolor='purple', linewidth=2)
        self.ax.add_patch(swing_circle)
        
        # 축 라벨 텍스트 제거됨

    def update_excavator(self, p0: Tuple[float, float], p1: Tuple[float, float], 
                        p2: Tuple[float, float], p3: Tuple[float, float]) -> None:
        """굴착기 링크 업데이트 (안전한 버전)"""
        try:
            # 업데이트 빈도 제한 (성능 최적화)
            if not hasattr(self, '_last_update_time'):
                self._last_update_time = 0
            
            import time
            current_time = time.time()
            if current_time - self._last_update_time < 0.02:  # 50 FPS 제한
                return
            self._last_update_time = current_time
            
            # Y 오프셋 적용
            oy = self.view_offset_y_m
            p0_plot = (p0[0], p0[1] + oy)
            p1_plot = (p1[0], p1[1] + oy)
            p2_plot = (p2[0], p2[1] + oy)
            p3_plot = (p3[0], p3[1] + oy)
            
            # 기존 굴착기 요소들 안전하게 제거
            artists_to_remove = []
            
            for artist in self.ax.lines + self.ax.collections:
                if hasattr(artist, '_excavator_element') and not hasattr(artist, '_trajectory'):
                    artists_to_remove.append(artist)
            
            for artist in artists_to_remove:
                try:
                    artist.remove()
                except:
                    pass  # 이미 제거된 경우 무시
            
            # 새로운 굴착기 링크 그리기 (단순화된 파라미터)
            # 붐 (Boom)
            boom_line = self.ax.plot([p0_plot[0], p1_plot[0]], [p0_plot[1], p1_plot[1]], 
                                   'g-', linewidth=6)[0]
            boom_line._excavator_element = True
            
            # 암 (Arm)
            arm_line = self.ax.plot([p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]], 
                                  'b-', linewidth=5)[0]
            arm_line._excavator_element = True
            
            # 버킷 (Bucket)
            bucket_line = self.ax.plot([p2_plot[0], p3_plot[0]], [p2_plot[1], p3_plot[1]], 
                                     'orange', linewidth=4)[0]
            bucket_line._excavator_element = True
            
            # 관절 포인트 (단순화)
            joints_x = [p0_plot[0], p1_plot[0], p2_plot[0], p3_plot[0]]
            joints_y = [p0_plot[1], p1_plot[1], p2_plot[1], p3_plot[1]]
            joints_scatter = self.ax.scatter(joints_x, joints_y, c='red', s=80, zorder=5)
            joints_scatter._excavator_element = True
            
            # 궤적 업데이트
            self.trajectory_points.append((p3_plot[0], p3_plot[1]))
            
            # 기존 궤적 제거
            traj_lines_to_remove = []
            for line in self.ax.lines:
                if hasattr(line, '_excavator_element') and hasattr(line, '_trajectory'):
                    traj_lines_to_remove.append(line)
            
            for line in traj_lines_to_remove:
                try:
                    line.remove()
                except:
                    pass
            
            # 새로운 궤적 그리기
            if len(self.trajectory_points) > 1:
                traj_x = [pt[0] for pt in self.trajectory_points]
                traj_y = [pt[1] for pt in self.trajectory_points]
                traj_line = self.ax.plot(traj_x, traj_y, 'orange', linewidth=2, alpha=0.7)[0]
                traj_line._excavator_element = True
                traj_line._trajectory = True
            
            # 안전한 화면 업데이트
            try:
                self.figure.canvas.draw_idle()
            except:
                # draw_idle 실패 시 기본 draw 사용
                self.figure.canvas.draw()
                
        except Exception as e:
            print(f"굴착기 업데이트 오류: {e}")
            # 오류 발생 시 무시하고 계속 진행

    def reset_trajectory(self) -> None:
        """궤적만 초기화 (굴착기는 유지)"""
        self.trajectory_points = []
        # 궤적만 제거하고 굴착기는 유지
        lines_to_remove = []
        for line in self.ax.lines:
            if hasattr(line, '_excavator_element') and hasattr(line, '_trajectory'):
                lines_to_remove.append(line)
        for line in lines_to_remove:
            line.remove()
        self.figure.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("굴착기 IK/FK 데모 (Matplotlib)")
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

        # Top: Excavator view (matplotlib)
        self.view = ExcavatorMatplotlibView(self.params)
        self.view.setMinimumHeight(500)
        layout.addWidget(self.view)
        layout.setStretchFactor(self.view, 3)  # stretch factor 별도 설정

        # Mode buttons: IK / FK
        mode_row = QHBoxLayout()
        self.ik_btn = QRadioButton("역기구학 (IK)")
        self.fk_btn = QRadioButton("기구학 (FK)")
        self.fk_btn.setChecked(True)
        # 궤적 리셋 버튼
        self.reset_btn = QPushButton("궤적 리셋")
        self.reset_btn.clicked.connect(self._on_reset_trajectory)
        # 평탄화 작업 자동 재생 버튼
        self.demo_btn = QPushButton("평탄화 작업")
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
        layout.addWidget(self.controls_container)
        layout.setStretchFactor(self.controls_container, 2)

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
        self.ik_z = LabeledSlider("z (m)", -5.0, max_reach, 0.01, "m", self.end_pose()[1])
        self.ik_theta = LabeledSlider("theta (deg)", -180.0, 180.0, 0.5, "deg", math.degrees(self.end_pose()[2]))
        for w in (self.ik_x, self.ik_z, self.ik_theta):
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
        
        # 모드 변경 시 모든 락 해제
        self._unlock_all_sliders()
        self._update_view()
    
    def _unlock_all_sliders(self) -> None:
        """모든 슬라이더 락 해제"""
        if self.ik_btn.isChecked():
            if hasattr(self, 'ik_x'):
                self.ik_x.set_locked(False)
            if hasattr(self, 'ik_z'):
                self.ik_z.set_locked(False)
            if hasattr(self, 'ik_theta'):
                self.ik_theta.set_locked(False)
        else:
            if hasattr(self, 'fk_boom'):
                self.fk_boom.set_locked(False)
            if hasattr(self, 'fk_arm'):
                self.fk_arm.set_locked(False)
            if hasattr(self, 'fk_bucket'):
                self.fk_bucket.set_locked(False)

    def _check_joint_limits_and_lock(self, test_joints: JointAngles) -> bool:
        """조인트 한계를 검사하고 필요시 슬라이더 락"""
        p = self.params
        
        # 각 조인트가 한계를 넘었는지 확인
        boom_over = (math.degrees(test_joints.boom_rad) < p.boom_min_deg or 
                    math.degrees(test_joints.boom_rad) > p.boom_max_deg)
        arm_over = (math.degrees(test_joints.arm_rad) < p.arm_min_deg or 
                   math.degrees(test_joints.arm_rad) > p.arm_max_deg)
        bucket_over = (math.degrees(test_joints.bucket_rad) < p.bucket_min_deg or 
                      math.degrees(test_joints.bucket_rad) > p.bucket_max_deg)
        
        # IK 모드일 때만 슬라이더 락 적용
        if self.ik_btn.isChecked():
            # 한계 초과 시 해당 목표값 슬라이더 락
            any_over_limit = boom_over or arm_over or bucket_over
            
            if any_over_limit:
                # 모든 슬라이더 락
                self.ik_x.set_locked(True)
                self.ik_z.set_locked(True) 
                self.ik_theta.set_locked(True)
                return False  # 업데이트 중지
            else:
                # 모든 슬라이더 언락
                self.ik_x.set_locked(False)
                self.ik_z.set_locked(False)
                self.ik_theta.set_locked(False)
                return True  # 업데이트 허용
        else:
            # FK 모드에서는 개별 슬라이더 락
            if hasattr(self, 'fk_boom'):
                self.fk_boom.set_locked(boom_over)
            if hasattr(self, 'fk_arm'):
                self.fk_arm.set_locked(arm_over)
            if hasattr(self, 'fk_bucket'):
                self.fk_bucket.set_locked(bucket_over)
            return True  # FK는 항상 업데이트 (클램핑됨)

    def on_any_value_changed(self) -> None:
        if self.ik_btn.isChecked():
            # IK mode: compute joints from pose
            x = self.ik_x.value()
            z = self.ik_z.value()
            theta = math.radians(self.ik_theta.value())
            ik = self.model.inverse(x, z, theta, self.joints)
            
            if ik is not None:
                # 조인트 한계 검사
                if self._check_joint_limits_and_lock(ik):
                    self.joints = ik
                else:
                    # 한계 초과 시 이전 위치 유지하고 업데이트 중지
                    return
            else:
                # IK 해가 없으면 슬라이더 락
                self.ik_x.set_locked(True)
                self.ik_z.set_locked(True)
                self.ik_theta.set_locked(True)
                return
        else:
            # FK mode: read joints
            b = math.radians(self.fk_boom.value())
            a = math.radians(self.fk_arm.value())
            k = math.radians(self.fk_bucket.value())
            test_joints = JointAngles(b, a, k)
            
            # 조인트 한계 검사 및 슬라이더 락
            self._check_joint_limits_and_lock(test_joints)
            
            # FK는 클램핑된 값 사용
            self.joints = self.model._clamp_to_limits(test_joints)
            
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
        self.demo_btn.setText("평탄화 정지")
        # 현재 굴착기 위치를 시작점으로 저장
        self._start_pose = self.end_pose()  # (x, z, theta)
        self._demo_t = 0
        self._demo_phase = 0  # 0: 이동, 1: 평탄화, 2: 복귀
        if not hasattr(self, "_demo_timer"):
            self._demo_timer = QTimer(self)
            self._demo_timer.timeout.connect(self._demo_tick)
        self._demo_timer.start(33)  # ~30 FPS (더 부드러운 렌더링을 위해 조정)

    def _stop_demo(self) -> None:
        if getattr(self, "_demo_running", False):
            self._demo_running = False
            if hasattr(self, "_demo_timer"):
                self._demo_timer.stop()
        self.demo_btn.setText("평탄화 작업")

    def _calculate_bucket_angle(self, x: float, x_start: float, x_end: float) -> float:
        """x 위치에 따른 자연스러운 버킷 각도 계산 (끝단 각도 theta 기준)
        x가 멀 때: -50도, x가 가까울 때: -120도
        """
        # x 위치를 0~1 범위로 정규화 (x_start=0, x_end=1)
        if abs(x_start - x_end) < 1e-6:
            return math.radians(-50.0)
        
        progress = (x - x_start) / (x_end - x_start)
        progress = max(0.0, min(1.0, progress))  # 0~1 범위로 제한
        
        # 시작각 -50도에서 끝각 -120도로 변화 (끝단 theta 각도)
        start_angle = -50.0
        end_angle = -120.0
        
        # 부드러운 곡선 변화 (ease-in-out)
        smooth_progress = 0.5 * (1 - math.cos(progress * math.pi))
        angle = start_angle + (end_angle - start_angle) * smooth_progress
        
        return math.radians(angle)


    def _validate_and_adjust_target(self, x: float, z: float, theta: float) -> Tuple[float, float, float]:
        """목표 위치를 IK로 검증하고 필요시 안전한 범위로 조정"""
        # 먼저 목표 위치가 도달 가능한지 IK로 검증
        test_ik = self.model.inverse(x, z, theta, self.joints)
        
        if test_ik is not None:
            # 도달 가능하면 그대로 반환
            return x, z, theta
        
        # 도달 불가능하면 안전한 위치로 조정
        # 1. z 높이를 점진적으로 조정 (지면에서 위로)
        for z_offset in [0.1, 0.2, 0.3, 0.5]:
            adjusted_z = z + z_offset
            test_ik = self.model.inverse(x, adjusted_z, theta, self.joints)
            if test_ik is not None:
                return x, adjusted_z, theta
        
        # 2. x 위치를 베이스 쪽으로 조정
        for x_offset in [0.1, 0.2, 0.5, 1.0]:
            adjusted_x = x + x_offset if x < 0 else x - x_offset
            test_ik = self.model.inverse(adjusted_x, z, theta, self.joints)
            if test_ik is not None:
                return adjusted_x, z, theta
        
        # 3. 버킷 각도를 더 보수적으로 조정
        for theta_offset in [math.radians(10), math.radians(20), math.radians(30)]:
            adjusted_theta = theta + theta_offset
            test_ik = self.model.inverse(x, z, adjusted_theta, self.joints)
            if test_ik is not None:
                return x, z, adjusted_theta
        
        # 모든 조정이 실패하면 현재 위치 유지
        try:
            current_x, current_z, current_theta = self.end_pose()
            return current_x, current_z, current_theta
        except:
            # end_pose 호출도 실패하면 기본값 반환
            return x, z, theta

    def _demo_tick(self) -> None:
        # 3단계 평탄화 작업: 1) y=0 지점으로 접근+전진 2) x축 감소 이동 3) 초기 위치 복귀
        self._demo_t += 1
        
        # 시작 위치와 작업 설정
        start_x, start_z, start_theta = self._start_pose
        work_height = 0.0  # y축 값 0 (지면 높이)
        forward_distance = 1.0  # 1미터 전진
        work_x_start = start_x + forward_distance  # 작업 시작 x 위치
        
        # 작업 범위 계산 (고정값으로 설정)
        work_x_end = 4.5  # X축 4.5m까지만 이동
        
        # 각 단계별 지속 시간 (프레임 수)
        phase_duration = 150  # 각 단계당 2.5초 (60fps 기준)
        
        if self._demo_phase == 0:  # 1단계: y=0 지점으로 접근 + 1미터 전진 (하나의 모션)
            progress = min(1.0, self._demo_t / phase_duration)
            # 부드러운 이동을 위한 ease-in-out 함수
            smooth_progress = 0.5 * (1 - math.cos(progress * math.pi))
            
            # 목표: y=0, x는 현재위치+1m
            target_x = work_x_start
            target_z = work_height
            target_theta = self._calculate_bucket_angle(work_x_start, work_x_start, work_x_end)
            
            x = start_x + (target_x - start_x) * smooth_progress
            z = start_z + (target_z - start_z) * smooth_progress
            theta = start_theta + (target_theta - start_theta) * smooth_progress
            
            # IK 검증 및 조정
            x, z, theta = self._validate_and_adjust_target(x, z, theta)
            
            if progress >= 1.0:
                self._demo_phase = 1
                self._demo_t = 0
                
        elif self._demo_phase == 1:  # 2단계: x축 감소 방향으로 평탄화 작업
            progress = min(1.0, self._demo_t / (phase_duration * 1.2))  # 평탄화는 조금 더 천천히
            
            # x축으로 후진하면서 평탄화
            x = work_x_start + (work_x_end - work_x_start) * progress
            z = work_height  # y축 높이 유지 (지면)
            # x 위치에 따른 동적 버킷 각도
            theta = self._calculate_bucket_angle(x, work_x_start, work_x_end)
            
            # IK 검증 및 조정
            x, z, theta = self._validate_and_adjust_target(x, z, theta)
            
            if progress >= 1.0:
                self._demo_phase = 2
                self._demo_t = 0
                
        elif self._demo_phase == 2:  # 3단계: 초기 위치로 복귀
            progress = min(1.0, self._demo_t / phase_duration)
            smooth_progress = 0.5 * (1 - math.cos(progress * math.pi))
            
            # 현재 위치에서 원래 위치로 부드럽게 복귀
            current_x = work_x_end
            current_z = work_height
            current_theta = self._calculate_bucket_angle(work_x_end, work_x_start, work_x_end)
            
            x = current_x + (start_x - current_x) * smooth_progress
            z = current_z + (start_z - current_z) * smooth_progress
            theta = current_theta + (start_theta - current_theta) * smooth_progress
            
            # IK 검증 및 조정
            x, z, theta = self._validate_and_adjust_target(x, z, theta)
            
            if progress >= 1.0:
                self._stop_demo()
                return

        # 검증된 목표 위치로 역기구학 계산
        ik = self.model.inverse(x, z, theta, self.joints)
        if ik is not None:
            self.joints = ik
            self._update_view()
        else:
            # 이론적으로는 검증을 거쳤으므로 이 경우는 발생하지 않아야 함
            pass  # 조용히 실패 처리
        
        # 안전 장치: 너무 오래 실행되면 자동 종료
        if self._demo_t > phase_duration * 4:
            self._stop_demo()

    def end_pose(self) -> Tuple[float, float, float]:
        p0, p1, p2, p3 = self.model.forward(self.joints)
        theta = self.joints.boom_rad + self.joints.arm_rad + self.joints.bucket_rad
        return (p3[0], p3[1], theta)

    def _update_view(self) -> None:
        p0, p1, p2, p3 = self.model.forward(self.joints)
        self.view.update_excavator(p0, p1, p2, p3)
        # Sync FK controls with current joints
        if not self.ik_btn.isChecked():
            self.fk_boom.set_value(math.degrees(self.joints.boom_rad))
            self.fk_arm.set_value(math.degrees(self.joints.arm_rad))
            self.fk_bucket.set_value(math.degrees(self.joints.bucket_rad))
        else:
            x, z, th = self.end_pose()
            self.ik_x.set_value(x)
            self.ik_z.set_value(z)
            self.ik_theta.set_value(math.degrees(th))


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
