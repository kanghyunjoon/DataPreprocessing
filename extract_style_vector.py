from style_encoder_model import StyleEncoder
from PIL import Image
import torch
import torchvision.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델 불러오기
encoder = StyleEncoder(embedding_dim=128).to(device)
encoder.load_state_dict(torch.load("D:/바탕화면/project/models/style_encoder_epoch16.pt"))
encoder.eval()

# 이미지 불러오기 및 전처리
img_paths = ["D:/바탕화면/project/user_input/img1.jpg", "D:/바탕화면/project/user_input/img2.jpg", "D:/바탕화면/project/user_input/img3.jpg"]
images = [Image.open(p).convert("L") for p in img_paths]
transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
tensor_imgs = torch.stack([transform(img) for img in images]).to(device)

# 스타일 벡터 추출
with torch.no_grad():
    style_vector = encoder(tensor_imgs)

torch.save(style_vector, "D:/바탕화면/project/user_input/style_vector.pt")
