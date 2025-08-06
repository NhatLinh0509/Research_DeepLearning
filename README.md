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
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("nhatlinh59/blip-finetuned-part9")
model = BlipForConditionalGeneration.from_pretrained("nhatlinh59/blip-finetuned-part9")

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

img_dir = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
batch_size = 4
epoch_per_part = 1

for i in range(num_parts):
    print(f"\n🔁 Huấn luyện phần {i}")
    dataset = Flickr30kDataset(f"splits/train_part_{i}.csv", img_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train()
    for epoch in range(epoch_per_part):
        loop = tqdm(dataloader)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = input_ids.clone()

            outputs = model(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_description(f"Part {i}")
            loop.set_postfix(loss=loss.item())

    path = f"/kaggle/working/blip-finetuned-part{i}"
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"✅ Đã lưu tại: {path}")

