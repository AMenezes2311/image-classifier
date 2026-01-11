## 📱 Project Overview

This project is a **lightweight computer vision application** demonstrating **transfer learning in practice** using **MobileNetV2**. The focus is on building an efficient, responsive pipeline suitable for real-time or near–real-time inference, rather than maximizing model complexity.

---

## 🧠 Key Features

- **Transfer Learning with MobileNetV2**  
  Uses a pretrained MobileNetV2 backbone to achieve strong performance while maintaining low latency and computational cost.

- **Fast Image Preprocessing**  
  Implements efficient preprocessing using **OpenCV** and **NumPy** to minimize input-to-inference overhead.

- **Cached Model Loading**  
  Models are loaded once and cached in memory, enabling smooth user experience and avoiding repeated initialization costs.

- **Lightweight & Responsive Design**  
  Designed to run efficiently on consumer hardware, emphasizing practical deployment considerations.

---

## 🔁 Inference Pipeline

1. Load and cache pretrained MobileNetV2 model  
2. Preprocess input images using OpenCV and NumPy  
3. Run inference using the fine-tuned model  
4. Return predictions with minimal latency  

##📜 License & Usage Modification: Not permitted.

Redistribution: Only allowed with proper attribution and without any changes to the original files.

Commercial Use: Only with prior written consent.

📌 Attribution All credits for the creation, design, and development of this project go to:

Andre Menezes 📧 Contact: andremenezes231@hotmail.com 🌐 Website: https://andremenezes.dev

If this project is used, cited, or referenced in any form (including partial code, design elements, or documentation), you must provide clear and visible attribution to the original author(s).

⚠️ Disclaimer This project is provided without any warranty of any kind, either expressed or implied. Use at your own risk.

📂 File Integrity Do not alter, rename, or remove any files, directories, or documentation included in this project. Checksum or signature verification may be used to ensure file authenticity.

© 2025 Andre Menezes. All Rights Reserved.
