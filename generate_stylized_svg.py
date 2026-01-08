
import os
import torch
import pickle
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Resize, Lambda, ToTensor

from stylized_svg_model import StyleConditionedTransformer
from style_encoder_model import StyleEncoder
from deepsvg.utils.utils import remove_all_attrs

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STYLE_ENCODER_PATH = "models/style_encoder_epoch16.pt"
GENERATOR_PATH = "models/generator_epoch16.pt"
ID_TO_TOKEN_PATH = "tokenizer/id_to_token.pkl"
OUTPUT_PKL_PATH = "generated_tokens/generated_tokens.pkl"
USER_IMG_DIR = "user_images"
VOCAB_SIZE = 512
EMBED_DIM = 128
D_MODEL = 256
MAX_LEN = 128

# 전처리 정의
transform = Compose([
    Resize((128, 128)),  # 정사각형으로 비율 고정
    Lambda(lambda img: ImageOps.pad(img, (128, 128), color=255)),  # 흰색 패딩
    ToTensor()
])

# 모델 로드
encoder = StyleEncoder(embedding_dim=EMBED_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(STYLE_ENCODER_PATH, map_location=DEVICE))
encoder.eval()

generator = StyleConditionedTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    style_dim=EMBED_DIM
).to(DEVICE)
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
generator.eval()

# 스타일 벡터 추출
user_img_paths = sorted([os.path.join(USER_IMG_DIR, fname) for fname in os.listdir(USER_IMG_DIR) if fname.endswith(".png") or fname.endswith(".jpg")])
images = [Image.open(p).convert("L") for p in user_img_paths]
images = torch.stack([transform(img) for img in images]).to(DEVICE)
style_vector = encoder(images).mean(dim=0, keepdim=True)  # 평균 스타일 벡터

# 토큰 시퀀스 생성
start_token = torch.tensor([[1]], dtype=torch.long).to(DEVICE)
generated_tokens = generator.generate(style_vector, start_token, max_len=MAX_LEN)

# 저장
os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
with open(OUTPUT_PKL_PATH, "wb") as f:
    pickle.dump([generated_tokens[0].tolist()], f)

print(f"✅ 생성된 토큰 시퀀스를 {OUTPUT_PKL_PATH}에 저장했습니다.")
