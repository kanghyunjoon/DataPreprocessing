import os
import pickle
from svg_tokenizer import SVGTokenizer  # 수정한 svg_tokenizer.py
from deepsvg.utils.utils import remove_all_attrs

# 설정
VOCAB_SIZE = 512
generated_token_pkl = "D:/바탕화면/project/generated_tokens/generated_tokens.pkl"
id_to_token_path = "D:/바탕화면/project/tokenizer/id_to_token.pkl"
output_dir = "D:/바탕화면/project/generated_svgs"
os.makedirs(output_dir, exist_ok=True)

# tokenizer 생성 및 vocab 로드
tokenizer = SVGTokenizer(vocab_size=VOCAB_SIZE)
with open(id_to_token_path, "rb") as f:
    tokenizer.id_to_token = pickle.load(f)
    tokenizer.token_to_id = {v: k for k, v in tokenizer.id_to_token.items()}

# 토큰 시퀀스 로드
with open(generated_token_pkl, "rb") as f:
    sequences = pickle.load(f)

# SVG 복원
for i, seq in enumerate(sequences):
    try:
        path_d = tokenizer.decode(seq)
        path_d_cleaned = remove_all_attrs(path_d)
        svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'><path d='{path_d_cleaned}' fill='black'/></svg>"
        with open(os.path.join(output_dir, f"gen_{i:03}.svg"), "w", encoding="utf-8") as f:
            f.write(svg)
    except Exception as e:
        print(f"⚠️ {i}번 시퀀스 복원 실패: {e}")

print(f"✅ SVG {len(sequences)}개 복원 완료 → {output_dir}")
