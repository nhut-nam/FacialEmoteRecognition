# FacialEmoteRecognition: EmoNeXt & ConvNeXt with Differential Deep Metric Learning

## 📌 Giới thiệu
Đây là repo cho bài toán **nhận diện cảm xúc khuôn mặt** trên dataset **FER-2013**.  
Chúng tôi triển khai và so sánh hai hướng tiếp cận chính:

- **EmoNeXt**: Mô hình cải tiến dựa trên ConvNeXt, được tối ưu hoá cho bài toán phân loại cảm xúc.  
- **ConvNeXt with Differential Deep Metric Learning**: Kết hợp ConvNeXt backbone với **Differential Deep Metric Learning** để tăng khả năng phân tách giữa các lớp cảm xúc.

---

## 🏗️ Kiến trúc mô hình
### 1. EmoNeXt
- Dựa trên ConvNeXt architecture.  
- Cải tiến với các attention module (ví dụ: CBAM, SMAL) để tăng khả năng trích xuất đặc trưng khuôn mặt.  
- Hướng tới việc đạt **độ chính xác cao hơn ConvNeXt gốc** trên FER-2013.  

### 2. ConvNeXt + Differential Deep Metric Learning
- Sử dụng ConvNeXt làm backbone feature extractor.  
- Thêm **Differential Deep Metric Learning loss** nhằm cải thiện khoảng cách giữa embedding vector của các lớp.  
- Giúp mô hình học representation tốt hơn, tăng khả năng phân loại các cảm xúc gần nhau.  

---

## 📂 Dataset
- **FER-2013**: bộ dữ liệu gồm hơn 35,000 ảnh khuôn mặt xám (48x48) thuộc 7 cảm xúc:  
  `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`.  
- Dataset được tiền xử lý bao gồm: cân bằng lớp, data augmentation (flip, rotation, normalization).  

---

## ⚙️ Cài đặt
Clone repo:
```bash
git clone https://github.com/nhut-nam/FacialEmoteRecognition.git
cd FacialEmoteRecognition
