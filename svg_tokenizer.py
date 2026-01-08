# ✅ PNG 이미지 → SVG → 토큰화 → .pkl 저장
# 스타일 학습용 데이터셋을 위한 자동 변환 스크립트

import os
import subprocess
import pickle
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from deepsvg.svglib.svg_parse import svg_file_to_paths
from deepsvg.svglib.svg_preprocess import svg_preprocess
from deepsvg.svglib.svg_tokenizer import SVGTokenizer
from typing import List
# 설정
DATA_ROOT = "D:/바탕화면/project/data/raw_png_val"  # 손글씨 이미지들이 들어있는 루트 디렉토리
OUT_EXT = ".svg"  # 중간 저장 확장자
MAX_LEN = 128 
TOKENIZER_VOCAB_SIZE = 512

class SVGTokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

        # 예시: 최소한의 토큰 사전 정의 (실제 학습된 값으로 대체 필요)
        self.id_to_token = {
            0: "<PAD>",
            1: "M 10 10",
            2: "L 20 20",
            3: "L 30 10",
            4: "Z"
        }
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    def decode(self, token_ids: List[int]) -> str:
        PAD_IDX = 0
        try:
            return "\n".join(self.id_to_token[idx] for idx in token_ids if idx != PAD_IDX)
        except KeyError as e:
            raise KeyError(f"토큰 ID {e}는 id_to_token 사전에 없습니다.")

def png_to_svg(input_path, pgm_path, svg_path):
    """PNG → PGM 저장 후 Potrace SVG 변환"""
    img = Image.open(input_path).convert("L")  # grayscale
    img.save(pgm_path)  # Potrace가 읽을 수 있는 포맷으로 저장
    subprocess.run(["potrace", str(pgm_path), "-s", "-o", str(svg_path)], check=True)

def process_one_png(png_path: Path, tokenizer: SVGTokenizer):
    svg_path = png_path.with_suffix(OUT_EXT)
    pgm_path = png_path.with_suffix(".pgm")
    pkl_path = png_path.with_suffix(".pkl")

    if os.path.exists(svg_path):
        print(f"이미 존재함: {svg_path.name}")
        return

    try:
        # PNG → PGM → SVG
        png_to_svg(png_path, pgm_path, svg_path)

        # SVG → 토큰
        path_strings = svg_file_to_paths(svg_path)
        norm_data = svg_preprocess(path_strings)
        tokens = tokenizer.tokenize(norm_data, max_len=MAX_LEN)

        # 저장
        with open(pkl_path, "wb") as f:
            pickle.dump(tokens, f)

        print(f"✅ {png_path.name} → {pkl_path.name}")

    except Exception as e:
        print(f"[ERROR] {png_path.name}: {e}")

    finally:
        if os.path.exists(pgm_path):
            os.remove(pgm_path)  # 중간 파일 제거

def batch_process_all():
    tokenizer = SVGTokenizer(vocab_size=TOKENIZER_VOCAB_SIZE)
    all_pngs = list(Path(DATA_ROOT).rglob("*.png"))

    print(f"총 {len(all_pngs)}개의 PNG 이미지 처리 시작...")
    for png_path in tqdm(all_pngs):
        try:
            process_one_png(png_path, tokenizer)
        except subprocess.CalledProcessError:
            print(f"[Potrace 실패] {png_path}")

if __name__ == "__main__":
    batch_process_all()
