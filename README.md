# GradLoss : GradCAM++을 활용한 손실함수

GradCAM++을 적용하여 모델이 특징을 잘 학습하도록 도와주는 손실함수를 구현한 프로젝트입니다.

## 프로젝트 구조

```
project/
└── src/
    ├── Models/              # 모델 체크포인트, 로그 저장
    ├── COCO_datasets/
    │   ├── coco2017/
    │   │   ├── train2017/
    │   │   ├── val2017/
    │   │   └── annotations/
    │   └── download_coco.py
    ├── GradLoss/
    │   └── gradloss.py
    └── tests/
        ├── ce_train.py
        ├── grad_train.py
        └── evaluator.py
```



## 예시 실행

1. COCO 데이터셋 다운로드 및 추출:
```bash
python src/COCO_datasets/download_coco.py
```

2. 모델 학습:
```bash
python src/tests/ce_train.py
python src/tests/gradloss_train.py
```

3. 모델 평가:
```bash
python src/tests/evaluator.py
```

4. 학습 진행 상황 확인:
```bash
tensorboard --logdir=run_logs
```

## 체크포인트

체크포인트는 20 에폭마다 Models 디렉토리에 저장됩니다:
- 기본 학습: `basic_checkpoint_epoch_[N].pth`
- GradCAM 학습: `grad_checkpoint_epoch_[N].pth`

## 성능 지표

Tensorboard에 다음 항목들이 기록됩니다:
- 학습/검증 손실
- 정확도 지표
- 혼동 행렬
- GradCAM 시각화