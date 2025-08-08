# Prompt: 파이썬으로 굴착기 IK/FK UI 구현 (use TODO list)

아래 프롬프트를 그대로 붙여넣어 실행하세요. Cursor/코딩 에이전트가 단계별 TODO를 수행하며, 실행 가능한 PyQt5 앱을 생성합니다.

---
You are an AI coding assistant working in Cursor. Build a runnable Python UI app that visualizes a 2D excavator arm and supports both forward kinematics (FK) and inverse kinematics (IK).

Follow the checklist strictly, checking off items as you complete them, and run/verify after code edits. Keep edits minimal and focused. If you say you will do something, do it in the same turn.

## TODO
- [ ] Set up Python project metadata
- [ ] Create `requirements.txt` with pinned dependencies
- [ ] Implement `excavator_ui.py` (PyQt5) with:
  - [ ] Top: 2D schematic with links for boom, arm, bucket (swing/base yaw is optional; default 0)
  - [ ] Middle: mode selector buttons — `역기구학 (IK)`, `기구학 (FK)`
  - [ ] Bottom: sliders
    - [ ] IK 모드: `x (m)`, `y (m)`, `theta (deg)`
    - [ ] FK 모드: `붐 (deg)`, `암 (deg)`, `버킷 (deg)`
  - [ ] Real-time updates: moving sliders updates the top drawing immediately
  - [ ] Use `QPainter.Antialiasing` for smooth rendering
  - [ ] Use Qt signals instead of parent traversal for slider updates
- [ ] Implement kinematics (`ExcavatorKinematics`)
  - [ ] Forward kinematics for 3-DOF planar chain
  - [ ] Inverse kinematics for pose `(x, y, theta)` with elbow-up/down solutions and selection by proximity to current state
  - [ ] Clamp to joint limits
  - [ ] Parameters approximating a 30-ton excavator:
    - [ ] Link lengths (m): `boom=6.15`, `arm=3.20`, `bucket=1.50`
    - [ ] Joint limits (deg): `boom [0, 75]`, `arm [-130, 130]`, `bucket [-120, 120]`
- [ ] Provide run instructions (`python excavator_ui.py`)
- [ ] Run and verify launch without errors; fix issues (e.g., rendering flags, signal wiring)
- [ ] Final pass: code cleanliness, readable names, minimal nesting, consistent formatting

## Requirements
- Python 3.10+
- PyQt5 (Qt 5.15+)

## Deliverables
- `requirements.txt`
- `excavator_ui.py`

## Acceptance Criteria
- App launches without errors.
- Switching between IK/FK updates the bottom control panel accordingly.
- Moving sliders updates the excavator drawing in real time.
- FK: adjusting `붐/암/버킷` reflects correct geometry.
- IK: adjusting `x/y/theta` solves to feasible joint angles (within limits) and updates the drawing.
- Joint limits respected (angles clamped or rejected gracefully).
- Code uses Qt signals for value propagation (no brittle `parentWidget()` chains).
- Rendering is anti-aliased for smoother visuals.

## Non-Goals / Notes
- Swing(base yaw) is optional; default to 0 (no yaw) for a clean 2D planar view.
- Exact OEM specs are not required; use the provided 30t approximations.
- Keep dependencies minimal (PyQt5 only).

## Run
- Install deps: `python -m pip install -r requirements.txt`
- Run app: `python excavator_ui.py`

---
