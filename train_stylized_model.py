# ✅ Style-conditioned DeepSVG 모델 학습 코드
# 목적: 다양한 손글씨 스타일에 대해 스타일 벡터를 조건으로 SVG 토큰 시퀀스를 생성할 수 있도록 학습

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from style_encoder_model import StyleEncoder, StyleConditionedTransformer
from torchvision import transforms
from PIL import Image
import os
import pickle
from tqdm import tqdm
from stylized_svg_model import StyleConditionedTransformer

# ===== 데이터셋 경로 설정 =====
DATASET_DIR = "D:/바탕화면/project/data/token_dataset"
TRAIN_PKL = os.path.join(DATASET_DIR, "train.pkl")
VAL_PKL = os.path.join(DATASET_DIR, "val.pkl")
TEST_PKL = os.path.join(DATASET_DIR, "test.pkl")

# ✍️ 사용자 정의 데이터셋: (이미지, token 시퀀스) 반환
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TokenSequenceDataset(Dataset):
    def __init__(self, token_list, max_seq_len=128):
        self.tokens = token_list
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_seq = self.tokens[idx][:self.max_seq_len]
        token_seq = token_seq + [0] * (self.max_seq_len - len(token_seq))
        return torch.tensor(token_seq, dtype=torch.long)

# ✅ 데이터셋 로딩
with open(TRAIN_PKL, "rb") as f:
    train_tokens = pickle.load(f)
with open(VAL_PKL, "rb") as f:
    val_tokens = pickle.load(f)

train_dataset = TokenSequenceDataset(train_tokens)
val_dataset = TokenSequenceDataset(val_tokens)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ✅ 모델 초기화
embedding_dim = 128
vocab_size = 512
max_seq_len = 128
d_model = 256

style_encoder = StyleEncoder(embedding_dim=embedding_dim).to(device)
generator = StyleConditionedTransformer(vocab_size=vocab_size, d_model=d_model, style_dim=embedding_dim).to(device)

# ✅ 옵티마이저 및 손실 함수
params = list(style_encoder.parameters()) + list(generator.parameters())
optimizer = optim.Adam(params, lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ✅ 학습 루프
EPOCHS = 20

for epoch in range(EPOCHS):
    style_encoder.train()
    generator.train()
    total_loss = 0

    for tokens in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train"):
        tokens = tokens.to(device)
        #style_vecs = style_encoder(torch.randn_like(tokens.unsqueeze(1).float()))  # Dummy input
        # ✅ 수정된 코드
        batch_size = tokens.size(0)
        dummy_imgs = torch.randn(batch_size, 1, 128, 128).float().to(device)
        style_vecs = style_encoder(dummy_imgs)
        
        input_tokens = tokens[:, :-1]  # 입력 시퀀스
        target_tokens = tokens[:, 1:]  # 정답 시퀀스

        logits = generator(input_tokens, style_vecs)  # (B, T-1, vocab)
        loss = criterion(logits.view(-1, vocab_size), target_tokens.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

    # 검증
    style_encoder.eval()
    generator.eval()
    val_loss = 0

    with torch.no_grad():
        for tokens in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Val"):
            tokens = tokens.to(device)
            #style_vecs = style_encoder(torch.randn_like(tokens.unsqueeze(1).float()))
            # ✅ 수정된 코드
            batch_size = tokens.size(0)
            dummy_imgs = torch.randn(batch_size, 1, 128, 128).float().to(device)
            style_vecs = style_encoder(dummy_imgs)
            
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits = generator(input_tokens, style_vecs)
            loss = criterion(logits.view(-1, vocab_size), target_tokens.reshape(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.4f}")

    # 중간 저장
    torch.save(style_encoder.state_dict(), f"D:/바탕화면/project/models/style_encoder_epoch{epoch+1}.pt")
    torch.save(generator.state_dict(), f"D:/바탕화면/project/models/generator_epoch{epoch+1}.pt")
