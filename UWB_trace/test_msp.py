import tkinter as tk
from tkinter import ttk
import threading
import socket
import struct
import time
import re
import math
import json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from flask import Flask, jsonify

# ==========================================
# ⚙️ 核心战术网络配置
# ==========================================
MODULE_IP = "192.168.4.1"
MODULE_PORT = 14550
MSP_SET_RAW_RC = 200
MSP_ATTITUDE = 108  # 🎯 索要飞机实时姿态(包含机头朝向)

uwb_pattern = re.compile(r'A0:\s*([\d.]+)\s*A1:\s*([\d.]+)\s*A2:\s*([\d.]+)\s*A3:\s*([\d.]+)')

# ==========================================
# 🧠 全局状态与自动驾驶共享内存
# ==========================================
state = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'yaw_angle': 0.0,
    'r0': 0.0, 'r1': 0.0, 'r2': 0.0, 'r3': 0.0,
    # 📏 新增：UWB 系统误差零偏 (米)
    'offset_r0': 0.0, 'offset_r1': 0.0, 'offset_r2': 0.0, 'offset_r3': 0.0,
    'target_x': 2.0, 'target_y': 2.0, 'target_z': 1.0,
    'throttle': 1000,
    'is_armed': False,
    'auto_mode': False,
    'hover_throttle': 1300,
    'rc_roll': 1500, 'rc_pitch': 1500, 'rc_yaw': 1500, 'rc_throttle': 1000,
    'anchors': np.array([
        [0.0, 0.0, 0.5], [0.0, 4.0, 2.5],
        [4.0, 4.0, 0.5], [4.0, 0.0, 2.5]
    ])
}


# ==========================================
# 📐 PID 控制器核心算法
# ==========================================
class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
        self.limit = output_limit
        self.integral = 0;
        self.last_error = 0

    def compute(self, error, dt):
        if dt <= 0.0: return 0
        p_term = self.kp * error
        self.integral = max(-self.limit, min(self.limit, self.integral + error * dt))
        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error
        return max(-self.limit, min(self.limit, p_term + self.ki * self.integral + d_term))


pid_x = PIDController(kp=40, ki=0, kd=10, output_limit=100)
pid_y = PIDController(kp=40, ki=0, kd=10, output_limit=100)
pid_z = PIDController(kp=60, ki=5, kd=20, output_limit=200)


def error_function(guess_pos, measured_distances):
    error = 0
    for i in range(4):
        calc_dist = np.sqrt(np.sum((guess_pos - state['anchors'][i]) ** 2))
        error += (calc_dist - measured_distances[i]) ** 2
    return error


def calculate_xyz(r0, r1, r2, r3):
    initial_guess = [state['target_x'], state['target_y'], state['target_z']]
    result = minimize(error_function, initial_guess, args=([r0, r1, r2, r3],), method='Nelder-Mead')
    return result.x[0], result.x[1], result.x[2]


def generate_rc_packet(roll, pitch, yaw, throttle, aux1=1000):
    payload = struct.pack("<HHHHHHHH", int(roll), int(pitch), int(throttle), int(yaw), aux1, 1000, 1000, 1000)
    size = len(payload)
    body = struct.pack(f"<BB{size}s", size, MSP_SET_RAW_RC, payload)
    checksum = size ^ MSP_SET_RAW_RC
    for byte in payload: checksum ^= byte
    return b'$M<' + body + struct.pack("<B", checksum)


def request_msp_packet(cmd):
    return struct.pack('<3sBBB', b'$M<', 0, cmd, cmd)


def network_thread_logic():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 14550))
    sock.setblocking(False)

    last_rc_time = 0
    last_pid_time = time.time()
    last_query_time = 0

    while True:
        current_time = time.time()

        # 1. 📥 接收双路数据 (UWB文本 + 飞控二进制)
        try:
            raw_data, _ = sock.recvfrom(2048)

            # --- 解析 UWB 测距 ---
            try:
                text_line = raw_data.decode('gbk', errors='ignore').strip()
                match = uwb_pattern.search(text_line)
                if match:
                    raw_r0, raw_r1, raw_r2, raw_r3 = map(float, match.groups())
                    # 🎯 核心：在此处扣除校准误差！(保证最短距离为 0.01m 防止数学引擎除零崩溃)
                    state['r0'] = max(0.01, raw_r0 - state['offset_r0'])
                    state['r1'] = max(0.01, raw_r1 - state['offset_r1'])
                    state['r2'] = max(0.01, raw_r2 - state['offset_r2'])
                    state['r3'] = max(0.01, raw_r3 - state['offset_r3'])

                    state['x'], state['y'], state['z'] = calculate_xyz(state['r0'], state['r1'], state['r2'],
                                                                       state['r3'])
            except:
                pass

            # --- 解析 飞控 MSP 回传 ---
            if b'$M>' in raw_data:
                idx = raw_data.find(b'$M>')
                if len(raw_data) >= idx + 6:
                    size = raw_data[idx + 3]
                    cmd = raw_data[idx + 4]
                    payload = raw_data[idx + 5: idx + 5 + size]

                    if cmd == MSP_ATTITUDE and len(payload) >= 6:
                        roll_raw, pitch_raw, yaw_raw = struct.unpack('<hhh', payload[0:6])
                        state['yaw_angle'] = yaw_raw

        except BlockingIOError:
            pass

        # 2. 🧠 执行 PID 与下发 (50Hz)
        if current_time - last_rc_time >= 0.02:
            dt = current_time - last_pid_time
            last_pid_time = current_time

            if state['auto_mode'] and state['is_armed']:
                err_x = state['target_x'] - state['x']
                err_y = state['target_y'] - state['y']
                err_z = state['target_z'] - state['z']

                # 盲飞策略 (假设机头朝向前方 +Y)
                pitch_out = pid_x.compute(err_y, dt)
                state['rc_pitch'] = 1500 - pitch_out

                roll_out = pid_y.compute(err_x, dt)
                state['rc_roll'] = 1500 + roll_out

                throttle_out = pid_z.compute(err_z, dt)
                state['rc_throttle'] = max(1000, min(1700, state['hover_throttle'] + throttle_out))
                state['rc_yaw'] = 1500
            else:
                state['rc_roll'], state['rc_pitch'], state['rc_yaw'] = 1500, 1500, 1500
                state['rc_throttle'] = state['throttle']

            current_aux1 = 1500 if state['is_armed'] else 1000
            packet = generate_rc_packet(state['rc_roll'], state['rc_pitch'], state['rc_yaw'], state['rc_throttle'],
                                        aux1=current_aux1)
            sock.sendto(packet, (MODULE_IP, MODULE_PORT))
            last_rc_time = current_time

        # 3. 📤 索要飞机姿态 (10Hz)
        if current_time - last_query_time >= 0.1:
            sock.sendto(request_msp_packet(MSP_ATTITUDE), (MODULE_IP, MODULE_PORT))
            last_query_time = current_time

        time.sleep(0.001)


# ==========================================
# 🖥️ GUI 界面引擎
# ==========================================
class DroneGCS(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🛸 UAV 战术雷达 (全知天眼版 + 校准系统)")
        self.geometry("1100x900")
        self.left_frame = ttk.Frame(self);
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.right_frame = ttk.LabelFrame(self, text="📡 战术雷达与轨迹监控 (带红箭头，可拖拽目标与基站)", padding=5)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.dragging = None

        self.create_panels()
        self.create_radar()
        self.update_gui_loop()

    def create_panels(self):
        ctrl_frame = ttk.LabelFrame(self.left_frame, text="🚀 核心动力系统", padding=10)
        ctrl_frame.pack(fill="x", pady=5)
        self.btn_arm = tk.Button(ctrl_frame, text="🔴 锁定状态 (点击解锁)", bg="#ffcccc",
                                 font=("Microsoft YaHei", 12, "bold"), command=self.toggle_arm)
        self.btn_arm.pack(fill="x", pady=5)
        self.btn_auto = tk.Button(ctrl_frame, text="🤖 开启自动驾驶 (Auto-Pilot)", bg="#e0e0e0",
                                  font=("Microsoft YaHei", 11, "bold"), command=self.toggle_auto)
        self.btn_auto.pack(fill="x", pady=5)

        ttk.Label(ctrl_frame, text="手动油门推杆:").pack(pady=(5, 0))
        self.lbl_throttle = ttk.Label(ctrl_frame, text="1000", font=("Consolas", 12, "bold"), foreground="red")
        self.lbl_throttle.pack()
        self.slider_thr = ttk.Scale(ctrl_frame, from_=1000, to=2000, orient="horizontal",
                                    command=lambda v: self.update_val('throttle', v))
        self.slider_thr.set(1000);
        self.slider_thr.pack(fill="x")

        rc_frame = ttk.LabelFrame(self.left_frame, text="📊 飞机状态与摇杆量", padding=10)
        rc_frame.pack(fill="x", pady=5)
        self.lbl_rc = ttk.Label(rc_frame, text="Roll: 1500 | Pitch: 1500", font=("Consolas", 10, "bold"),
                                foreground="purple")
        self.lbl_rc.pack()

        # 🎯 机头角度展示
        self.lbl_yaw = ttk.Label(rc_frame, text="机头朝向 (Yaw): 0.0°", font=("Consolas", 12, "bold"),
                                 foreground="brown")
        self.lbl_yaw.pack(pady=5)

        nav_frame = ttk.LabelFrame(self.left_frame, text="🎯 导航与测距", padding=10)
        nav_frame.pack(fill="x", pady=5)
        self.lbl_pos = ttk.Label(nav_frame, text="X: 0.00 | Y: 0.00 | Z: 0.00", font=("Consolas", 12, "bold"),
                                 foreground="blue")
        self.lbl_pos.pack(pady=5)

        # 🎯 新增：显示 4 个基站的实时测距
        self.lbl_raw_dist = ttk.Label(nav_frame, text="A0: 0.00m | A1: 0.00m | A2: 0.00m | A3: 0.00m",
                                      font=("Consolas", 10, "bold"), foreground="#d35400")
        self.lbl_raw_dist.pack(pady=2)

        ttk.Label(nav_frame, text="设置目标 X/Y/Z (m):").pack()
        entry_frame = ttk.Frame(nav_frame);
        entry_frame.pack()
        self.e_x = ttk.Entry(entry_frame, width=5);
        self.e_x.insert(0, str(state['target_x']));
        self.e_x.pack(side="left", padx=2)
        self.e_y = ttk.Entry(entry_frame, width=5);
        self.e_y.insert(0, str(state['target_y']));
        self.e_y.pack(side="left", padx=2)
        self.e_z = ttk.Entry(entry_frame, width=5);
        self.e_z.insert(0, str(state['target_z']));
        self.e_z.pack(side="left", padx=2)

        # 📏 新增：基站位置与UWB零偏校准综合面板
        calib_frame = ttk.LabelFrame(self.left_frame, text="📍 基站位置与 UWB 零偏校准", padding=10)
        calib_frame.pack(fill="x", pady=5)

        ttk.Label(calib_frame, text="节点").grid(row=0, column=0)
        ttk.Label(calib_frame, text="X (m)").grid(row=0, column=1)
        ttk.Label(calib_frame, text="Y (m)").grid(row=0, column=2)
        ttk.Label(calib_frame, text="Z (m)").grid(row=0, column=3)
        ttk.Label(calib_frame, text="误差 (-m)").grid(row=0, column=4)

        self.anchor_entries = []
        self.offset_entries = []
        for i in range(4):
            ttk.Label(calib_frame, text=f"A{i}:").grid(row=i + 1, column=0)
            row_xyz = []
            for j in range(3):
                ent = ttk.Entry(calib_frame, width=5)
                ent.insert(0, str(state['anchors'][i][j]))
                ent.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                row_xyz.append(ent)
            self.anchor_entries.append(row_xyz)

            # 校准误差框
            ent_off = ttk.Entry(calib_frame, width=5)
            ent_off.insert(0, str(state[f'offset_r{i}']))
            ent_off.grid(row=i + 1, column=4, padx=2, pady=2)
            self.offset_entries.append(ent_off)

        ttk.Button(calib_frame, text="刷新物理基站与校准参数", command=self.update_calib).grid(row=5, column=0,
                                                                                               columnspan=5, pady=8)

    def create_radar(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

    # 🖱️ 雷达拖拽逻辑增强：支持拖拽基站！
    def on_mouse_press(self, event):
        if event.xdata is None or event.ydata is None: return

        # 1. 检查是否点击了目标点
        if np.hypot(event.xdata - state['target_x'], event.ydata - state['target_y']) < 0.3:
            self.dragging = 'target'
            return

        # 2. 检查是否点击了某个基站
        for i in range(4):
            if np.hypot(event.xdata - state['anchors'][i][0], event.ydata - state['anchors'][i][1]) < 0.3:
                self.dragging = f'anchor_{i}'
                return

    def on_mouse_motion(self, event):
        if self.dragging is None or event.xdata is None or event.ydata is None: return

        # 拖拽目标点
        if self.dragging == 'target':
            state['target_x'] = event.xdata;
            state['target_y'] = event.ydata
            self.e_x.delete(0, tk.END);
            self.e_x.insert(0, f"{event.xdata:.2f}")
            self.e_y.delete(0, tk.END);
            self.e_y.insert(0, f"{event.ydata:.2f}")

        # 拖拽基站
        elif self.dragging.startswith('anchor_'):
            idx = int(self.dragging.split('_')[1])
            state['anchors'][idx][0] = event.xdata
            state['anchors'][idx][1] = event.ydata
            if hasattr(self, 'anchor_entries'):
                self.anchor_entries[idx][0].delete(0, tk.END);
                self.anchor_entries[idx][0].insert(0, f"{event.xdata:.2f}")
                self.anchor_entries[idx][1].delete(0, tk.END);
                self.anchor_entries[idx][1].insert(0, f"{event.ydata:.2f}")

    def on_mouse_release(self, event):
        self.dragging = None

    def toggle_arm(self):
        if not state['is_armed']:
            if state['throttle'] > 1050: return
            state['is_armed'] = True
            self.btn_arm.config(text="🟢 已解锁 (Armed)", bg="#ccffcc", fg="green")
        else:
            state['is_armed'] = False;
            state['auto_mode'] = False
            self.btn_auto.config(text="🤖 开启自动驾驶", bg="#e0e0e0", fg="black")
            state['throttle'] = 1000;
            self.slider_thr.set(1000)
            self.btn_arm.config(text="🔴 锁定状态 (点击解锁)", bg="#ffcccc", fg="black")

    def toggle_auto(self):
        if not state['is_armed']: return
        state['auto_mode'] = not state['auto_mode']
        if state['auto_mode']:
            self.btn_auto.config(text="⚡ 自动驾驶中 (点击接管)", bg="#ffd700", fg="red")
        else:
            self.btn_auto.config(text="🤖 开启自动驾驶", bg="#e0e0e0", fg="black")

    def update_val(self, key, val):
        state[key] = int(float(val))
        if key == 'throttle': self.lbl_throttle.config(text=str(state['throttle']))

        # 🛠️ 校准参数同步更新

    def update_calib(self):
        try:
            for i in range(4):
                state['anchors'][i][0] = float(self.anchor_entries[i][0].get())
                state['anchors'][i][1] = float(self.anchor_entries[i][1].get())
                state['anchors'][i][2] = float(self.anchor_entries[i][2].get())
                state[f'offset_r{i}'] = float(self.offset_entries[i].get())
            print("✅ 基站坐标与 UWB 校准参数已更新！")
        except ValueError:
            print("❌ 参数输入错误，请输入有效数字！")

    def update_gui_loop(self):
        self.lbl_pos.config(text=f"X: {state['x']:5.2f}m | Y: {state['y']:5.2f}m | Z: {state['z']:5.2f}m")

        # 🎯 新增：实时刷新 4 个基站的距离，如果断联会直接显示 0.01 左右的极小值
        self.lbl_raw_dist.config(
            text=f"A0: {state['r0']:4.2f}m | A1: {state['r1']:4.2f}m | A2: {state['r2']:4.2f}m | A3: {state['r3']:4.2f}m")

        self.lbl_rc.config(text=f"Roll: {int(state['rc_roll'])} | Pitch: {int(state['rc_pitch'])}")
        self.lbl_yaw.config(text=f"机头朝向 (Yaw): {state['yaw_angle']}°")

        self.ax.clear()
        self.ax.set_xlim(-1, 6);
        self.ax.set_ylim(-1, 6)
        self.ax.grid(True, linestyle='--', alpha=0.6);
        self.ax.set_aspect('equal')

        xs = state['anchors'][:, 0];
        ys = state['anchors'][:, 1]
        self.ax.scatter(xs, ys, c='blue', marker='s', s=100)
        self.ax.scatter(state['target_x'], state['target_y'], c='red', marker='*', s=200)
        self.ax.scatter(state['x'], state['y'], c='green', marker='o', s=150, edgecolors='black')

        # 画出红色的机头方向箭头！
        yaw_rad = math.radians(state['yaw_angle'])
        arrow_length = 0.5
        dx = arrow_length * math.sin(yaw_rad)
        dy = arrow_length * math.cos(yaw_rad)
        self.ax.arrow(state['x'], state['y'], dx, dy, head_width=0.15, head_length=0.2, fc='red', ec='red')

        self.canvas.draw()
        self.after(100, self.update_gui_loop)


# ==========================================
# 🌐 Flask REST API — 外部获取 state
# ==========================================
api_app = Flask(__name__)


@api_app.route("/api/state", methods=["GET"])
def get_state():
    """返回完整的无人机状态 JSON"""
    # 将 numpy 数组转为列表以便 JSON 序列化
    serializable = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    return jsonify(serializable)


@api_app.route("/api/state/<key>", methods=["GET"])
def get_state_key(key):
    """返回 state 中的单个字段，例如 /api/state/x"""
    if key not in state:
        return jsonify({"error": f"key '{key}' not found"}), 404
    val = state[key]
    if isinstance(val, np.ndarray):
        return jsonify({key: val.tolist()})
    return jsonify({key: val})


@api_app.route("/api/state", methods=["POST"])
def set_state():
    """修改 state 中的字段（必须为 JSON body）"""
    from flask import request
    data = request.get_json(force=True)
    updated = {}
    for k, v in data.items():
        if k in state:
            if isinstance(state[k], np.ndarray) and isinstance(v, list):
                state[k] = np.array(v, dtype=float)
            else:
                state[k] = v
            updated[k] = str(state[k])
    return jsonify({"updated": updated})


def run_api_server():
    api_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    threading.Thread(target=network_thread_logic, daemon=True).start()
    threading.Thread(target=run_api_server, daemon=True).start()
    app = DroneGCS()
    app.mainloop()