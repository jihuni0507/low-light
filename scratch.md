# 🟦 Analysis

## 10.27

### Dataset
직접 촬영한 영상에서 프레임을 추출한 COLMAP 기반 커스텀 데이터셋을 사용함
``` text
cabin, chair, fan, musharna, office, table
```

### Experiment

**🔸 Train GE-Net**

``` text
normal light scenes:

musharna: 0 samples
chair: 3 samples
other scene: 1 sample
```
체크포인트:  `./snpashots/1027_best.pth`

**🔸 Inference**

Invesre ISP를 통한 Dark synthesis 강도를 세가지로 나눔
 - `strength=0` (normal light), `strength=0.5` (dark), `strength=0.9` (darker)

---

### Results
`strength=0.5`로 dark synthesis한 scene으로 학습한 GE-net 체크포인트를
정상 이미지에 적용했더니 가우시안들이 하얗게 타버림

 - 충분히 밝다면 더이상 밝게 하지 않는 loss도 추가해야 함
 - 또는 inverse_isp 과정 자체가 네트워크 학습을 불안정하게 하는 요소일 수 있음 (인위적인 색 열화 + 노이즈 재현에 의해)

✅ Baseline 가우시안에서부터 geometry를 제대로 잡지 못했지만 정상 이미지, dark synthesis된 이미지 모두 일관되게 enhance하는 결과가 나타남
→ 실제 GE-Net이 low-light restoration 역할을 수행하고 있음


❌ Baseline 가우시안에서부터 geometry가 크게 어긋남

❌ blue 컬러 편향이 나타남

❌ 인퍼런스 시 결과가 입력 이미지의 밝기에 비례함 (어두울수록 결과도 어두움)

# 이후 가능한 방향성
## 10.27
**⭐ 1️⃣ 실제 low-light 데이터셋 활용**

- Inverse ISP로 어둡게 열화시킨 이미지를 사용하지 않고 실제 저조도 환경에서 촬영한 scene 사용

- ❌ Low-light 시나리오를 가정하므로 필수적인 과정이지만, COLMAP이 제대로 작동하지 않아 카메라 포즈가 부정확해질 수 있음

**⭐⭐ 2️⃣ GE-Net 학습 시 입력 이미지의 Dark synthesis 강도를 늘려가며 학습**

- 정상 이미지로 학습한 GE-Net은 `strength`가 `0, 0.5, 0.9`인 scene 모두 enhance할 수 있었음
- Dark Synthesis한 이미지로 학습한 GE-Net을 적용할 경우 세가지 조건 모두 가우시안이 하얗게 타버리는 결과가 나타남

따라서 GE-Net을 학습할 때, 처음에는 정상 이미지를 입력하다가 에포크가 진행됨에 따라 `0.5`, `0.9`만큼 dark synthesis한 입력 이미지도 차례대로 학습할 수 있도록 설정

**⭐3️⃣ MVSplat으로 baseline 가우시안 재구성 시 `near`, `far`값 직접 라벨링**
- 실제 MVSplat 저자도 적절한 `near`, `far`값을 찾기 위해 임의로 값을 설정해보고 최적의 값을 찾아냈다고 함
- 특히 `near`값이 작아 가우시안이 카메라 앞에 몰리는 현상을 방지해야함 (`near_good_fan.png`, `near_too_low_table.png` 참고)
- 초기 geometry를 잡는 데 도움이 될 것으로 기대됨

**4️⃣ Loss항 추가**

- 현재 가우시안의 위치, 크기는 구정하고 SH 계수에만 loss를 부여하고 있음 위치 수정한다면 rendering loss까지 (tradeoff 자유도는 높아지지만 GT가 주어짐)

**5️⃣ (선택) 데이터셋 / 학습 scene의 개수를 늘려 generalization 확보**
- 외부 데이터셋 (RE10K, LLFF 등) 학습 시에 보지 못했던 새로운 scene에 대해서도 `baseline_gaussian`이 주어졌다면 enhance할 수 있게끔 학습
- ❌ 굳이 필요한가? + 학논연 레벨을 넘어감 + 파인튜닝 많이 해야할 것

---
## 1028

- `enhance_gaussians.py`에서 `r` 해석 시도해볼것 (되면 좋고)
- 이미지 -> Zero-DCE 먼저 하고 MVSplat 한것 과 결과 비교

- (선택) 외부 데이터셋

