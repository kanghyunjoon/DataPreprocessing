### 파이프라인

svg_tokenizer.py 를 통해 png를 토큰으로 변환 후  potrace로 svg생성 -> 결과값train.pkl, val.pkl, test.pkl

train_stylized_model.py로 학습

사용자 손글씨 이미지 기반 스타일벡터 추출
extract_style_vector.py 작성

사용자 손글씨 이미지를 받아 grayscale 변환 후 모델에 입력
style_vector 추출 완료

style_vector 기반 토큰 시퀀스 생성
generate_stylized_svg.py

토큰 시퀀스를 SVG로 변환 시도
tokens_to_svg.py
