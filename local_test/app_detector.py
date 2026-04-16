import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from pathlib import Path

class YoloTkinterApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x600")
        
        # 1. Load the YOLOv5 Best Model
        print("Loading YOLOv5 model...")
        self.script_dir = Path(__file__).resolve().parent
        self.bundle_root = self.script_dir.parent
        weights_path = self.bundle_root / "model" / "best.pt"
        yolo_path = self.bundle_root / "external" / "yolov5"

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path}. "
                "Run training/export first or update the checkpoint path."
            )
        
        # Load local YOLOv5 using torch.hub
        self.model = torch.hub.load(str(yolo_path), 'custom', path=str(weights_path), source='local')
        # Set confidence threshold to 0.5 as requested
        self.model.conf = 0.5 
        
        # 2. Setup OpenCV Video Capture
        self.cap = cv2.VideoCapture(0) # 0 for default webcam
        
        if not self.cap.isOpened():
            print("Warning: Could not open video device.")
            
        # 3. Build UI Layout
        self.main_frame = tk.Frame(window, bg="#2b2b2b")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel: Video Feed
        self.video_frame = tk.Frame(self.main_frame, bg="#2b2b2b")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        # Right Panel: Detections & Greeting
        self.sidebar = tk.Frame(self.main_frame, bg="#2b2b2b", width=300)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, expand=True, padx=20, pady=20)
        
        # Greeting Label
        self.greeting_var = tk.StringVar()
        self.greeting_var.set("Waiting...")
        self.greeting_label = tk.Label(
            self.sidebar, 
            textvariable=self.greeting_var, 
            font=("Helvetica", 24, "bold"), 
            fg="#00d8ff", 
            bg="#2b2b2b"
        )
        self.greeting_label.pack(pady=30)
        
        # Last Detection Title
        self.last_det_title = tk.Label(
            self.sidebar, 
            text="Last Detection:", 
            font=("Helvetica", 14), 
            fg="white", 
            bg="#2b2b2b"
        )
        self.last_det_title.pack(pady=(20, 5))
        
        # Last Detection Image Crop
        self.crop_label = tk.Label(self.sidebar, bg="#2b2b2b")
        self.crop_label.pack(pady=10)
        
        # 4. Starting the video loop
        self.delay = 15 # Refresh rate in ms
        self.update_frame()
        
        # Cleanup routine
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # OpenCV captures BGR, convert to RGB for YOLO and Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame)
            
            # Extract highest confidence face
            df = results.pandas().xyxy[0] 
            # (thresholding is already governed by self.model.conf = 0.5, but we can double check)
            
            best_conf = 0
            best_name = None
            best_crop = None
            
            drawn_frame = rgb_frame.copy()
            
            for index, row in df.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = row['confidence']
                name = row['name']
                
                # Draw bounding box on the frame
                cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    drawn_frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                if conf > best_conf:
                    best_conf = conf
                    best_name = name
                    # Extract the face segment
                    best_crop = rgb_frame[y1:y2, x1:x2]

            # Update sidebar with the best detection of this frame
            if best_name:
                self.greeting_var.set(f"Hello {best_name}!")
                
                if best_crop is not None and best_crop.size > 0:
                    crop_img = Image.fromarray(best_crop)
                    # Resize while keeping aspect ratio
                    crop_img.thumbnail((250, 250))
                    crop_tk = ImageTk.PhotoImage(image=crop_img)
                    self.crop_label.imgtk = crop_tk
                    self.crop_label.configure(image=crop_tk)
            else:
                self.greeting_var.set("Waiting...")
                self.crop_label.configure(image="")
            
            # Update Main Video Frame
            img = Image.fromarray(drawn_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Loop recursively
        self.window.after(self.delay, self.update_frame)
        
    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

if __name__ == '__main__':
    # Initialize UI
    root = tk.Tk()
    app = YoloTkinterApp(root, "Identity Recognition App")
    root.mainloop()
