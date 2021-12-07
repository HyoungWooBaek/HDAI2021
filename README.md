# HDAI2021

## 모델 학습
모델 학습시 train.py내부의 데이터셋 파일 경로 변경 필요

    python train.py --dataset All --model unet --loss gan
  
## 모델 테스트

    python inference.py --dataset A2C --model unet --parameter model_parameter_path
    
