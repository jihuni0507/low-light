# Injection Network Regularization Configuration

## 개요

학습된 injection 네트워크의 문제점을 해결하기 위해 두 가지 regularization이 추가되었습니다:

1. **Spatial Smoothness Regularization**: 인접한 가우시안의 SH 파라미터 차이를 최소화하여 시각적 불연속성 제거
2. **Magnitude Regularization**: SH 계수 크기를 제약하여 초록색 편향 및 color oversaturation 방지

## 학습 단계 설정

### `train_injection_network.py` 사용 시:

```bash
python train/train_injection_network.py \
  --dataset_yaml dataset.yaml \
  --sh_degree 3 \
  --epochs 20 \
  --lambda_image 1.0 \
  --lambda_ssim 0.2 \
  --lambda_spatial_smooth 0.01 \
  --lambda_magnitude 0.001 \
  --spatial_smooth_k 16 \
  --device cuda
```

### 파라미터 설명:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `lambda_spatial_smooth` | 0.01 | 공간적 smoothness 정규화 가중치 (0 = 비활성화) |
| `spatial_smooth_k` | 16 | 각 가우시안의 인접 가우시안 개수 (KNN) |
| `lambda_magnitude` | 0.001 | SH 크기 정규화 가중치 (0 = 비활성화) |

**권장값:**
- `lambda_spatial_smooth`: 0.005 ~ 0.05 (높을수록 smooth하지만 세부 손실 가능)
- `lambda_magnitude`: 0.0005 ~ 0.005 (높을수록 색감 안정적)

## 인퍼런스 단계 설정

### `config.yaml`에 `inference` 섹션 추가:

```yaml
pipeline:
  run_train_base: false
  run_clip_encode: true
  run_injection: true

injection:
  gaussian_ply: "path/to/source.ply"
  checkpoint: "output/injection/injection_network.pt"
  output_ply: "output/injection_updated.ply"

# 새로 추가: 인퍼런스 시 regularization 설정
inference:
  enable_spatial_smoothing: true        # 공간적 smoothing 적용
  smoothing_lambda: 0.1                 # Smoothing 강도 (0.0 ~ 1.0)
  smoothing_neighbors: 16               # 고려할 인접 가우시안 수
  smoothing_iterations: 3               # Iterative smoothing 횟수
  
  enable_magnitude_clipping: true       # SH 크기 제한
  max_magnitude: 0.5                    # 최대 SH 계수 크기
  
  enable_channel_normalization: true    # 채널별 정규화 (색감 균형)
```

### 파라미터 설명:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `enable_spatial_smoothing` | true | 인접 가우시안 간의 smoothness 적용 |
| `smoothing_lambda` | 0.1 | 원본 값과 smooth화된 값의 blend 비율 |
| `smoothing_neighbors` | 16 | 각 가우시안의 인접 가우시안 개수 |
| `smoothing_iterations` | 3 | Smoothing 반복 횟수 |
| `enable_magnitude_clipping` | true | SH 계수 크기 제한 |
| `max_magnitude` | 0.5 | 클리핑할 최대 크기 |
| `enable_channel_normalization` | true | 채널별 정규화로 색감 균형 |

**권장값:**
- `smoothing_lambda`: 0.05 ~ 0.2 (초록색 편향이 심하면 높게)
- `smoothing_iterations`: 1 ~ 5 (1회도 충분한 경우 많음)
- `max_magnitude`: 0.3 ~ 0.7 (높을수록 더 표현력 있지만 색감 이상 위험)

## 사용 예제

### 새로운 scene에서 초록색 편향이 많은 경우:

```yaml
inference:
  enable_spatial_smoothing: true
  smoothing_lambda: 0.15
  smoothing_neighbors: 24
  smoothing_iterations: 5
  
  enable_magnitude_clipping: true
  max_magnitude: 0.3
  
  enable_channel_normalization: true
```

### 세밀한 디테일을 유지하려는 경우:

```yaml
inference:
  enable_spatial_smoothing: true
  smoothing_lambda: 0.05
  smoothing_neighbors: 8
  smoothing_iterations: 1
  
  enable_magnitude_clipping: true
  max_magnitude: 0.6
  
  enable_channel_normalization: true
```

## 알고리즘 설명

### Spatial Smoothness Regularization

1. 각 가우시안의 3D 위치로부터 KNN을 통해 k개의 가장 가까운 이웃 찾기
2. 거리에 반비례하는 가중치 계산 (가까운 이웃에 더 높은 가중)
3. 이웃 가우시안들의 SH 파라미터의 가중 평균 계산
4. 현재 가우시안과 이웃 평균 간의 차이를 MSE로 최소화
5. 선택적으로 여러 번 반복 적용

**효과**: 공간적으로 인접한 가우시안들이 유사한 색감을 가지도록 강제하여, scene 전체의 visual coherence 향상

### Magnitude Regularization

1. 각 SH 계수의 L2 norm 계산
2. DC component (DC_0)과 나머지 성분들의 평균 크기 비교
3. 나머지 성분이 DC component보다 과도하게 크지 않도록 제약
4. 이를 통해 DC에 중심된 색감 유지

**효과**: 
- DC component (0차 SH)가 기본 색상을 결정하고, 고차 항이 이를 보조하도록 유지
- 특정 채널(예: 초록색)이 과도하게 강해지는 것을 방지

### Channel Normalization (인퍼런스에서만)

1. 각 채널(R, G, B)별로 평균과 표준편차 계산
2. 채널별로 정규화 (zero-mean, unit-variance)
3. 원본 mean/std로 다시 스케일링

**효과**: 색상 채널 간의 불균형 해소

## 트러블슈팅

### 문제: 초록색이 너무 강함
→ 해결책:
- `lambda_magnitude` 증가 (0.005 ~ 0.01)
- `max_magnitude` 감소 (0.3 ~ 0.4)
- `smoothing_lambda` 증가 (0.15 ~ 0.2)

### 문제: 이미지가 너무 연하고 흐릿해짐
→ 해결책:
- `smoothing_lambda` 감소 (0.05 ~ 0.08)
- `smoothing_iterations` 감소 (1~2)
- `max_magnitude` 증가 (0.6 ~ 0.8)

### 문제: 일부 부분이 이상하게 색이 바뀜
→ 해결책:
- `smoothing_neighbors` 증가 (24 ~ 32)
- `smoothing_iterations` 증가 (3 ~ 5)

## Loss 계산 예

```
total_loss = lambda_image * L1_loss 
           + lambda_ssim * SSIM_loss
           + lambda_spatial_smooth * spatial_smooth_loss
           + lambda_magnitude * magnitude_loss
```

## 메모리 및 성능

- Spatial smoothing은 KD-tree 구성에 O(n log n), query에 O(k log n) 소요
- 일반적으로 n=500k, k=16인 경우 인퍼런스 후 2~3초 추가 소요
- 학습 단계에서는 batch당 10~20% 정도 추가 시간 소요
