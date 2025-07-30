import cv2
import torch
import time
import serial
import threading
import math
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import subprocess
import re
import pandas as pd

class PestDetectionApp:
    def __init__(self, root):
        self.root = root
        self.servo2_current_pos = 0
        self.servo2_target_positions = [0, 180]
        self.servo2_pos_index = 0

        self.vertical_scan_cycles_completed_without_detection = 0
        self.pest_detected_during_current_vertical_scan = False

        # --- PERBAIKAN: Inisialisasi servo_running di sini ---
        self.servo_running = True # Inisialisasi di awal agar tersedia untuk semua thread

        self.ser = None # Inisialisasi self.ser di sini
        self.setup_serial()
        self.setup_detector()
        self.setup_gui()

    def setup_serial(self):
        try:
            # Pastikan port COM sesuai dengan Arduino Anda
            self.ser = serial.Serial('COM5', 9600, timeout=1) # Tambah timeout untuk readline
            time.sleep(2)
            print("[SERIAL] Serial connection established.")
            # Mulai thread untuk membaca data serial dari Arduino
            self.serial_read_thread = threading.Thread(target=self.read_serial_data)
            self.serial_read_thread.daemon = True # Daemon thread akan berhenti ketika program utama berhenti
            self.serial_read_thread.start()
        except Exception as e:
            print(f"[SERIAL ERROR] Connection failed: {e}")
            self.ser = None
            messagebox.showerror("Serial Error", f"Gagal terhubung ke port serial COM5: {e}\nPastikan Arduino terhubung dan tidak ada program lain yang menggunakan port ini.")

    def send_servo(self, servo_id, position):
        if 0 <= position <= 180 and self.ser and self.ser.is_open:
            try:
                data = bytearray()
                data.append(servo_id)
                data.extend(position.to_bytes(2, 'big'))
                self.ser.write(data)
                print(f"[SERVO-{servo_id}] Moving to {position}°")
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [SERVO-{servo_id}] Mengirim perintah ke {position}°\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))
            except Exception as e:
                print(f"[ERROR] Failed to move servo-{servo_id}: {e}")
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Gagal menggerakkan servo-{servo_id}: {e}\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))

    def read_serial_data(self):
        """Membaca data yang dikirim oleh Arduino melalui serial."""
        while self.ser and self.ser.is_open and self.servo_running: # Pastikan loop berhenti saat aplikasi ditutup
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"[ARDUINO MSG] {line}")
                        self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ARDUINO] {line}\n"))
                        self.root.after(0, lambda: self.log_text.yview(tk.END))
                time.sleep(0.05) # Jeda singkat agar tidak terlalu membebani CPU
            except serial.SerialException as e:
                print(f"[SERIAL READ ERROR] Koneksi serial terputus atau error: {e}")
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Koneksi serial terputus atau error: {e}\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))
                if self.ser and self.ser.is_open: # Cek lagi sebelum menutup
                    self.ser.close()
                self.ser = None
                break # Keluar dari loop jika terjadi error serial
            except Exception as e:
                print(f"[ERROR] Exception during serial read: {e}")
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Kesalahan saat membaca serial: {e}\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))


    def smooth_move_servo2(self, target_pos):
        """Smoothly move horizontal servo to target position"""
        self.pause_servo = True
        self.send_servo(1, 90)
        time.sleep(0.5)

        current = self.servo2_current_pos
        step = 1 if target_pos > current else -1

        if step > 0:
            for pos in range(current, target_pos + 1, step):
                if not self.servo_running: return
                self.send_servo(2, pos)
                time.sleep(0.01)
        else:
            for pos in range(current, target_pos - 1, step):
                if not self.servo_running: return
                self.send_servo(2, pos)
                time.sleep(0.01)

        self.servo2_current_pos = target_pos
        self.pause_servo = False

    def setup_detector(self):
        self.device = 'cpu'
        print(f"[INFO] Using device: {self.device}")
        self.class_names = ['Hama Putih Palsu', 'Penggerek Batang Padi', 'Wereng Coklat']
        self.confidence_threshold = 0.4
        try:
            self.model = YOLO("pest.pt").to(self.device)
        except Exception as e:
            print(f"[MODEL ERROR] Gagal memuat model YOLO: {e}")
            messagebox.showerror("Model Error", f"Gagal memuat model 'pest.pt': {e}\nPastikan file model ada dan PyTorch/YOLO terinstal dengan benar.")
            self.root.destroy()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[CAMERA ERROR] Gagal membuka kamera.")
            messagebox.showerror("Camera Error", "Gagal membuka kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
            self.root.destroy()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.ROTATION_HEIGHT_CM = 42.5
        self.HORIZONTAL_DISTANCE_CM = 28
        self.servo1_angle = 90

        self.output_folder = "output_detect"
        os.makedirs(self.output_folder, exist_ok=True)

        self.pause_servo = False
        self.prev_frame_time = 0

    def setup_gui(self):
        self.root.title("Deteksi Hama + Kontrol Servo")
        self.root.geometry("660x500")
        self.root.resizable(False, False)
        self.root.tk_setPalette(background='#FFFFFF', foreground='#000000')
        self.root.configure(bg='#FFFFFF')

        self.tab_control = ttk.Notebook(self.root)

        self.setup_webcam_tab()
        self.setup_log_tab()
        self.setup_output_tab()
        self.setup_control_tab()

        self.tab_control.pack(expand=1, fill="both")

        self.scan_thread = threading.Thread(target=self.automatic_scanning)
        self.scan_thread.daemon = True
        self.scan_thread.start()

        self.detection_thread = threading.Thread(target=self.update_frame)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def setup_webcam_tab(self):
        self.webcam_tab = tk.Frame(self.tab_control)
        self.tab_control.add(self.webcam_tab, text="Webcam")

        self.webcam_frame = tk.Frame(self.webcam_tab, bg='#FFFFFF')
        self.webcam_frame.pack(padx=10, pady=10)

        self.panel = tk.Label(self.webcam_frame, bg='#FFFFFF')
        self.panel.pack()

    def setup_log_tab(self):
        self.log_tab = tk.Frame(self.tab_control)
        self.tab_control.add(self.log_tab, text="Log")

        self.log_text = scrolledtext.ScrolledText(
            self.log_tab,
            width=80,
            height=15,
            bg="#FFFFFF",
            fg="#000000",
            font=("Arial", 10)
        )
        self.log_text.pack(padx=10, pady=10)
        self.log_text.insert(tk.END, "Log Deteksi Hama:\n")

        self.btn_save_log = tk.Button(
            self.log_tab,
            text="Simpan Log ke Excel",
            command=self.save_log_to_excel,
            bg="#8BC34A",
            fg="white"
        )
        self.btn_save_log.pack(pady=5)

    def setup_output_tab(self):
        self.output_tab = tk.Frame(self.tab_control)
        self.tab_control.add(self.output_tab, text="Output")

        self.output_frame = tk.Frame(self.output_tab, bg='#FFFFFF')
        self.output_frame.pack(padx=10, pady=10)

        self.btn_load_images = tk.Button(
            self.output_frame,
            text="Load Gambar Deteksi",
            command=self.list_output_images,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_load_images.pack(pady=5)

        self.btn_open_folder = tk.Button(
            self.output_frame,
            text="Buka Folder Output",
            command=self.open_output_folder,
            bg="#2196F3",
            fg="white"
        )
        self.btn_open_folder.pack(pady=5)

        self.btn_delete_all = tk.Button(
            self.output_frame,
            text="Hapus Semua Gambar",
            command=self.delete_all_images,
            bg="#F44336",
            fg="white"
        )
        self.btn_delete_all.pack(pady=5)

        self.image_list = scrolledtext.ScrolledText(
            self.output_frame,
            width=80,
            height=12,
            bg="#FFFFFF",
            fg="#000000",
            font=("Arial", 10)
        )
        self.image_list.pack(padx=10, pady=5)

    def setup_control_tab(self):
        self.control_tab = tk.Frame(self.tab_control)
        self.tab_control.add(self.control_tab, text="Kontrol Servo")

        self.control_frame = tk.Frame(self.control_tab, bg='#FFFFFF')
        self.control_frame.pack(padx=10, pady=10)

        tk.Label(self.control_frame, text="Kontrol Servo Horizontal:", bg='#FFFFFF').pack(pady=5)

        btn_frame = tk.Frame(self.control_frame, bg='#FFFFFF')
        btn_frame.pack(pady=10)

        self.btn_left = tk.Button(
            btn_frame,
            text="Kiri (0°)",
            command=lambda: self.smooth_move_servo2(0),
            bg="#2196F3",
            fg="white"
        )
        self.btn_left.pack(side=tk.LEFT, padx=5)

        self.btn_mid = tk.Button(
            btn_frame,
            text="Tengah (90°)",
            command=lambda: self.smooth_move_servo2(90),
            bg="#4CAF50",
            fg="white"
        )
        self.btn_mid.pack(side=tk.LEFT, padx=5)

        self.btn_right = tk.Button(
            btn_frame,
            text="Kanan (180°)",
            command=lambda: self.smooth_move_servo2(180),
            bg="#F44336",
            fg="white"
        )
        self.btn_right.pack(side=tk.LEFT, padx=5)

        tk.Label(self.control_frame, text="Posisi Servo Horizontal:", bg='#FFFFFF').pack(pady=5)
        self.servo2_slider = tk.Scale(
            self.control_frame,
            from_=0,
            to=180,
            orient=tk.HORIZONTAL,
            bg='#FFFFFF',
            command=self.on_servo2_slider_move
        )
        self.servo2_slider.set(90)
        self.servo2_slider.pack(pady=5)

    def on_servo2_slider_move(self, value):
        if not self.pause_servo:
            self.servo_running = False
            self.smooth_move_servo2(int(value))
            self.servo_running = True


    def automatic_scanning(self):
        """Automatic scanning pattern: Servo 2 moves only when no pest is detected for 2 vertical cycles."""
        while self.servo_running:
            target_pos_servo2 = self.servo2_target_positions[self.servo2_pos_index]
            self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Menggerakkan Servo 2 ke {target_pos_servo2}°\n"))
            self.root.after(0, lambda: self.log_text.yview(tk.END))
            self.smooth_move_servo2(target_pos_servo2)

            self.vertical_scan_cycles_completed_without_detection = 0

            while self.servo_running:
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Melakukan scanning vertikal di Servo 2 {target_pos_servo2}° (Siklus {self.vertical_scan_cycles_completed_without_detection + 1}/2)\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))

                self.pest_detected_during_current_vertical_scan = False
                self.scan_servo_vertikal()

                if not self.servo_running: return

                if self.pest_detected_during_current_vertical_scan:
                    self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DETEKSI HAMA] Hama terdeteksi di area fokus. Mereset siklus pemindaian vertikal.\n"))
                    self.root.after(0, lambda: self.log_text.yview(tk.END))
                    self.vertical_scan_cycles_completed_without_detection = 0
                else:
                    self.vertical_scan_cycles_completed_without_detection += 1
                    self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Siklus vertikal {self.vertical_scan_cycles_completed_without_detection} selesai tanpa deteksi.\n"))
                    self.root.after(0, lambda: self.log_text.yview(tk.END))

                    if self.vertical_scan_cycles_completed_without_detection >= 2:
                        self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Dua siklus vertikal selesai tanpa deteksi. Melanjutkan ke posisi Servo 2 berikutnya.\n"))
                        self.root.after(0, lambda: self.log_text.yview(tk.END))
                        break

                time.sleep(0.5)

            self.servo2_pos_index = (self.servo2_pos_index + 1) % len(self.servo2_target_positions)
            time.sleep(1)

    def scan_servo_vertikal(self):
        """Vertical scanning function for one complete cycle"""
        # Upward scan
        for pos in range(30, 131, 1):
            if not self.servo_running: return
            while self.pause_servo: time.sleep(0.1)
            self.servo1_angle = pos
            self.send_servo(1, pos)
            time.sleep(0.09)

        # Downward scan
        for pos in range(130, 29, -1):
            if not self.servo_running: return
            while self.pause_servo: time.sleep(0.1)
            self.servo1_angle = pos
            self.send_servo(1, pos)
            time.sleep(0.09)

    def pause_servo_temporarily(self, seconds):
        """Pause vertical scanning for specified time and stop horizontal servo (servo 2)"""
        if not self.pause_servo:
            self.pause_servo = True
            self.send_servo(1, self.servo1_angle)

            def resume():
                time.sleep(seconds)
                self.pause_servo = False
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Servo vertikal lanjut scanning.\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))
            threading.Thread(target=resume, daemon=True).start()

    def list_output_images(self):
        """List all images in output folder"""
        images = os.listdir(self.output_folder)
        self.image_list.delete(1.0, tk.END)
        for image in images:
            if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_list.insert(tk.END, image + "\n")

    def open_output_folder(self):
        """Open output folder in file explorer"""
        try:
            if os.name == 'nt':
                subprocess.Popen(f'explorer "{os.path.abspath(self.output_folder)}"')
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', os.path.abspath(self.output_folder)])
            else:
                messagebox.showwarning("Peringatan", "Pembukaan folder tidak didukung di OS ini.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuka folder: {e}")

    def delete_all_images(self):
        """Delete all images in output folder"""
        try:
            images = os.listdir(self.output_folder)
            if not images:
                messagebox.showwarning("Peringatan", "Tidak ada gambar untuk dihapus.")
                return

            confirm = messagebox.askyesno("Konfirmasi", "Apakah Anda yakin ingin menghapus SEMUA gambar deteksi?")
            if not confirm:
                return

            for image in images:
                if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(self.output_folder, image)
                    os.remove(file_path)

            messagebox.showinfo("Sukses", "Semua gambar telah dihapus.")
            self.list_output_images()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menghapus gambar: {e}")

    def save_log_to_excel(self):
        """Saves the content of the log_text widget to an Excel file."""
        log_content = self.log_text.get(1.0, tk.END)
        lines = log_content.strip().split('\n')

        data = []
        for line in lines:
            match = re.match(r"\[(.*?)\] \[(.*?)\] (.*)", line)
            if match:
                timestamp_str, log_type, message = match.groups()
                data.append([timestamp_str, log_type, message])
            else:
                data.append(['', '', line])

        if not data:
            messagebox.showwarning("Peringatan", "Log kosong, tidak ada yang bisa disimpan.")
            return

        df = pd.DataFrame(data, columns=['Timestamp', 'Tipe Log', 'Pesan'])

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Simpan Log ke Excel"
        )

        if file_path:
            try:
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Sukses", f"Log berhasil disimpan ke:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan log ke Excel:\n{e}")
        else:
            messagebox.showinfo("Info", "Penyimpanan log dibatalkan.")

    def update_frame(self):
        """Main detection and video update loop"""
        while self.servo_running:
            success, img = self.cap.read()
            if not success:
                print("[ERROR] Gagal baca dari kamera. Mencoba membuka ulang...")
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Gagal baca dari kamera. Mencoba membuka ulang...\n"))
                self.root.after(0, lambda: self.log_text.yview(tk.END))
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                time.sleep(1)
                continue

            h, w, _ = img.shape
            center_x, center_y = w // 2, h // 2

            box_width = int(w * 0.4)
            box_height = int(h * 0.4)

            box_x1 = center_x - box_width // 2
            box_y1 = center_y - box_height // 2
            box_x2 = center_x + box_width // 2
            box_y2 = center_y + box_height // 2

            new_frame_time = time.time()

            results = self.model(img, verbose=False)

            pests_in_focus_area = []

            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        cls = int(box.cls[0])
                        class_name = self.class_names[cls]

                        is_inside_box = (box_x1 <= cx <= box_x2 and
                                         box_y1 <= cy <= box_y2)

                        height_cm = None
                        if is_inside_box:
                            theta_rad = math.radians(self.servo1_angle - 90)
                            height_diff = math.tan(theta_rad) * self.HORIZONTAL_DISTANCE_CM
                            height_cm = round(max(0, self.ROTATION_HEIGHT_CM - height_diff), 1)

                            bbox_area = (x2 - x1) * (y2 - y1)
                            pests_in_focus_area.append({
                                'box': box,
                                'confidence': confidence,
                                'class_name': class_name,
                                'cx': cx,
                                'cy': cy,
                                'bbox_area': bbox_area,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'height_cm': height_cm
                            })

                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{class_name} {confidence:.2f}'

                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_x = max(5, x1)
                        label_y = max(label_size[1] + 10, y1 - 10)

                        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        if is_inside_box and height_cm is not None:
                            height_label = f'{height_cm:.1f} cm'
                            cv2.putText(img, height_label, (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            prioritized_pest = None
            if pests_in_focus_area:
                pests_in_focus_area.sort(key=lambda p: p['bbox_area'], reverse=True)
                prioritized_pest = pests_in_focus_area[0]

            if prioritized_pest and not self.pause_servo:
                self.pest_detected_during_current_vertical_scan = True

                log_msg = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DETEKSI PRIORITAS] "
                           f"{prioritized_pest['class_name']} (conf: {prioritized_pest['confidence']:.2f}) "
                           f"DI AREA FOKUS (Bounding Box Terbesar), Tinggi: {prioritized_pest['height_cm']:.1f} cm\n")
                self.root.after(0, lambda: self.log_text.insert(tk.END, log_msg))
                self.root.after(0, lambda: self.log_text.yview(tk.END))

                filename = os.path.join(self.output_folder, f"hama_{prioritized_pest['class_name'].replace(' ', '_')}_{int(time.time())}.jpg")
                cv2.imwrite(filename, img)

                self.pause_servo_temporarily(8)

            cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 165, 255), 3)

            fps = 1 / (new_frame_time - self.prev_frame_time + 1e-6)
            self.prev_frame_time = new_frame_time
            cv2.putText(img, f'FPS: {int(fps)}', (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.panel.configure(image=img)
            self.panel.image = img

            self.root.update_idletasks()

    def on_closing(self):
        """Cleanup when closing the application"""
        self.servo_running = False
        time.sleep(0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] Serial connection closed.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PestDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()