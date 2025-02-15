# Object Tracking with YOLOv8 and Deep SORT  

This project integrates **YOLOv8 for object detection** with **Deep SORT for object tracking**, enabling real-time tracking of people and objects. The system provides an interactive interface to **track individual objects**, **switch camera sources**, and **toggle between tracking only people, only objects, or both**.

---

## 📌 Requirements  

- **Python 3.7**  
- Install dependencies from `requirements.txt`:  
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 Deep SORT Integration  
We are working with this fork from the official Deep SORT implementation.

Download the Deep SORT feature extraction model:
[Feature Extraction Model](#)  

Place the downloaded model in the `model_data/` directory.

---

## 📂 Data  
You can download the dataset used for testing from the link below:  
[Download Dataset](#)

---

## ⚙️ Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Nikhil112024/object-tracking-yolov8-deep-sort.git
cd object-tracking-yolov8-deep-sort
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Object Tracking Script  
```bash
python main.py
```

---

## 🖥️ Controls & Features  

### 🟢 Track Button (Green → Red)  
- Click on an object or person in the video feed to track it.  
- Turns red when tracking is active.  

### 📷 Cam Button (Blue)  
- Click to switch the camera source.  
- Automatically cycles through available cameras.  

### 🔄 People/Object/Both Button (3 Colors)  
- Click to toggle between tracking only people, only objects, or both.  
- Changes color based on mode selection:  
  - 🟢 Green: Tracking only people  
  - 🔵 Blue: Tracking only objects  
  - 🟡 Yellow: Tracking both people and objects  

---

## 🏗️ Project Structure  
```
📂 object-tracking-yolov8-deep-sort
│── 📜 README.md               # Project Documentation  
│── 📄 requirements.txt        # Required Python Libraries  
│── 📂 model_data/             # YOLO & Deep SORT Model Files  
│── 📂 data/                   # Sample Test Data  
│── 📄 main.py                 # Main Tracking Code  
│── 📄 tracker.py              # Deep SORT Tracking Module  
│── 📄 utils.py                # Utility Functions  
│── 📂 outputs/                # Processed Videos & Images  
```

---

## 📜 About  

👨‍💻 **Developer:** Nikhil Kumar  
📍 **Location:** Dehradun, Uttarakhand  
📧 **Email:** nikhilkumarjuyal777@gmail.com  
🔗 **GitHub:** [github.com/Nikhil112024](https://github.com/Nikhil112024)

---

## 📜 Acknowledgments  
- **YOLOv8** by Ultralytics for real-time object detection.  
- **Deep SORT** for multi-object tracking.  
