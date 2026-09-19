# ML Lab 06 - Neural Network

## รายละเอียด
โปรเจกต์นี้เป็นการประยุกต์ใช้ Neural Network สำหรับการจำแนกข้อมูล
โดยใช้ Breast Cancer Wisconsin Dataset จาก Scikit-learn

วัตถุประสงค์ของ Lab นี้คือ
- โหลดและเตรียม Dataset
- แบ่งข้อมูลเป็น Training และ Testing
- Standardize ข้อมูล
- สร้างและฝึก Neural Network
- เปรียบเทียบผลลัพธ์จากจำนวน Epoch ที่แตกต่างกัน
- เปรียบเทียบโครงสร้าง Neural Network ที่แตกต่างกัน
- แสดงผล Accuracy และ Loss

---

## Dataset

ใช้ Breast Cancer Wisconsin Dataset

- จำนวนข้อมูล: 569 samples
- จำนวน Features: 30
- จำนวน Classes: 2
  - Malignant
  - Benign

Dataset ถูกโหลดจาก `sklearn.datasets`

---

## โครงสร้างโปรเจกต์

```text
ML-LAB-06/
│
├── main.py
├── neural_network.py
├── visualization.py
├── output.txt
├── loss.png
├── accuracy_epochs.png
├── configuration.png
└── README.md
