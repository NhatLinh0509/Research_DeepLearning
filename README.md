BLIP Flickr Captioning
📌 Giới thiệu
Repository này chứa toàn bộ mã nguồn và quá trình huấn luyện model BLIP để tạo caption cho ảnh, sử dụng bộ dữ liệu Flickr30k và Flickr8k.

Flickr30k: 2 model huấn luyện với dữ liệu 30.000 ảnh.

Flickr8k: 1 model huấn luyện với dữ liệu 8.000 ảnh.

📂 Model đã huấn luyện không được lưu trực tiếp tại đây do giới hạn dung lượng, mà được lưu trên Hugging Face Hub.

🔗 Link 3 model trên Hugging Face:

https://huggingface.co/nhatlinh59/blip-flickr-captioning/tree/main

🏆 Model tốt nhất
Model blip-finetuned-part9.zip cho kết quả tốt nhất trong toàn bộ quá trình huấn luyện.
Toàn bộ ví dụ kết quả dưới đây được tạo từ model này.

📂 Nội dung repo
/notebooks: Notebook huấn luyện model.

/scripts: Các script xử lý dữ liệu, train, và đánh giá.

README.md: Tài liệu này.

🛠 Cách sử dụng
python
Sao chép
Chỉnh sửa
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("nhatlinh59/blip-flickr30k-model1")
model = BlipForConditionalGeneration.from_pretrained("nhatlinh59/blip-flickr30k-model1")

img_url = "https://example.com/sample.jpg"
image = Image.open(requests.get(img_url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)
📸 Kết quả mẫu
Ví dụ 1
<img width="1054" height="546" alt="image" src="https://github.com/user-attachments/assets/cb806f81-0dd2-4c76-a1a6-c49aa1b5d475" />

a group of asian people posing for a picture.

Ví dụ 2
<img width="621" height="517" alt="image" src="https://github.com/user-attachments/assets/9b9e4fe3-c184-4c72-a313-fcc047a5c95d" />

a little girl in a gray sweater and jeans is playing on a bed.

📊 Thông tin huấn luyện
Thuật toán: Fine-tuning BLIP (Bootstrapping Language-Image Pre-training) với Transformer Decoder.

Batch size: 4

Optimizer: AdamW

Loss: Cross-Entropy Loss

Epochs: Chia dữ liệu thành nhiều phần, huấn luyện theo từng phần để tránh quá tải bộ nhớ.

