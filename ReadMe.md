# Audio Classification for Parkinson’s Disease Detection

## 📖 Description

โครงการนี้พัฒนาเพื่อจำแนกเสียงพูดของผู้ป่วยพาร์กินสัน (PD) และกลุ่มควบคุมสุขภาพดี (HC)  
โดยใช้ **ภาพ Mel-spectrogram** ที่แปลงมาจากเสียง “อา” (voice “Ahh”) ผ่านโมเดล Deep Learning ได้แก่  
**EfficientNet-B0** และ **MobileNetV3-Small**  
ต้องใช้ **Cuda Toolkit** เพื่อใช้ GPU ในการประมวลผลช่วยเพิ่มความเร็วในการ Training

---

## 🗂️ Project Structure

| Filename / Folder      | Description                                                                                                                                                                                                                                                           | Location                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Train**              | โฟลเดอร์เก็บภาพสำหรับ Train/Test กับโมเดล ทั้ง Control และ PD                                                                                                                                                                                                         | `Desktop\Audio Classification\DataSet`               |
| **Effciennet_Model**   | โมเดล EfficientNet ที่ฝึกมาแล้วและมีค่าความแม่นยำที่ดีที่สุด                                                                                                                                                                                                          | `Desktop\Audio Classification\Model\Effciennet_Best` |
| **MobileNet_Model**    | โมเดล MobileNet ที่ฝึกมาแล้วและมีค่าความแม่นยำที่ดีที่สุด                                                                                                                                                                                                             | `Desktop\Audio Classification\Model\Mobilenet_Best`  |
| **RawData**            | โฟลเดอร์เก็บไฟล์เสียง “อา” จากแอป CheckPD ทั้ง Control และ PD                                                                                                                                                                                                         | `Desktop\Audio Classification`                       |
| **audio_ex.py**        | แปลงไฟล์เสียงของกลุ่มผู้ป่วยพาร์กินสัน (PD) และกลุ่ม HC ให้เป็นภาพ Mel-spectrogram โดยโหลดไฟล์เสียงแต่ละไฟล์ สกัดคุณลักษณะ Mel-spectrogram ด้วยคลาส `AudioFeatureExtractor` และบันทึกผลลัพธ์เป็นภาพสำหรับเทรนโมเดล                                                    | `Desktop\Audio Classification`                       |
| **feature_extract.py** | คลาสสำหรับโหลดและประมวลผลไฟล์เสียง เพื่อทำ Feature Extraction เช่น Spectrogram และ Mel-spectrogram, ทำ normalization, การแสดงผล และบันทึกภาพ spectrogram                                                                                                              | `Desktop\Audio Classification`                       |
| **model2.py**          | โค้ดฝึกและประเมินโมเดล Deep Learning (EfficientNet-B0 และ MobileNetV3-Small) จำแนกภาพ Mel-spectrogram ของเสียงจากกลุ่มผู้ป่วยพาร์กินสัน (PD) และ Control (HC) พร้อมบันทึก Accuracy, F1-score, Confusion Matrix, ROC Curve, และ Classification Report สำหรับแต่ละ fold | `Desktop\Audio Classification`                       |
| **Results/**           | เก็บผลการประเมินของแต่ละโมเดล เช่น Confusion Matrix, ROC Curve, Classification Report                                                                                                                                                                                 | `Desktop\Audio Classification\Results`               |
| **requirements.txt**   | รายการไลบรารีที่ต้องติดตั้งก่อนรันโปรเจกต์                                                                                                                                                                                                                            | `Desktop\Audio Classification`                       |

---

## ⚙️ Installation

### 1️⃣ สร้าง Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # สำหรับ Windows
# หรือ
source venv/bin/activate  # สำหรับ macOS/Linux
```

### 2️⃣ ติดตั้ง Dependencies (รองรับ CUDA)

ตรวจสอบเวอร์ชัน CUDA

```bash
nvidia-smi
```

จากนั้นติดตั้ง PyTorch ที่รองรับ CUDA เวอร์ชันเดียวกัน:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

และติดตั้ง dependencies อื่น:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1️⃣ สร้าง Mel-spectrogram จากไฟล์เสียง

```bash
python audio_ex.py
```

### 2️⃣ เทรนและประเมินโมเดล (ใช้ GPU)

```bash
python model2.py
```
