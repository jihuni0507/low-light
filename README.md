# low-light

## Implementation Logic
``` Text
1. Reconstruct Gaussian splats with StereoGS
2. Generate text/image condition vector by CLIP encoder
3. Input Gaussians and condition vector into injection network
4. Freeze Gaussians' geometry, update only SH features
5. Loss - computed between rendered SH output and inverse-ISP pseudo-GT images

```
<!-- ## Overall Pipeline
<img width="995" height="963" alt="image" src="https://github.com/user-attachments/assets/f24ab313-1b85-4c14-bfd5-0f7054fb2636" /> -->

## 📋 TODO
- RE10K, LOL 같은 외부 데이터 다운받아서 바로 Dataloader에 로딩될 수 있도록 `dataset.yaml` 구축 로직 구현
- 현재 `Dataset` 구조에서 이미지만 바로 어둡게 하는 로직 구현
- 추후 loss 항목 별로 확인할 수 있도록 로그 추가
- PLY 생성시 reference 뷰 두개 인덱스가 파일명에 들어가도록 수정
- `near`, `far` 관련 현재 두개: 고정시켜두면 loss가 높고 유동적으로 하면 형태가 무너짐(실제로 더 밝아지긴 함) <- 코드 수정해야함


## Instructions

### 🔶 Environment Setting

``` Bash
Set-Location -Path "your/project/directory"
$env:PYTHONPATH = "your/project/directory"
```

### 🔶 `main.py`
`config.yaml` 파일을 통해 실험 세부 설정을 할 수 있다.

**🚫!!`yaml`파일 구성에 따라 달라질 수 있음!!🚫**
- 실험 시나리오 별 `yaml`파일 생성하거나
- Base `yaml` + 세부 `yaml` 모듈화 시도 중

### 🔶 Dataset 구성

``` txt
low-light/
├── dataset.yaml
├── train/
│   ├── train_enhance_network.py
│   └── initialize_gaussians.py
├── datasets/
│   └── dataset.py
├── render/
│   └── renderer_splats.py
└── data/
    ├── scene_001/
    │   ├── point_cloud.ply
    │   ├── sparse/
    │   │   └── 0/
    │   │       ├── cameras.txt
    │   │       └── images.txt
    │   └── images/
    │       ├── lowlight_01.png
    │       ├── lowlight_02.png
    │       └── ...
    └── scene_002/
        ├── point_cloud.ply
        ├── sparse/
        │   └── 0/
        │       ├── cameras.txt
        │       └── images.txt
        └── images/
            ├── lowlight_01.png
            ├── lowlight_02.png
            └── ...
```

#### **🔶 Synthesizing Dark Images - Inverse ISP**
`models/inverse_isp.py`에 구현되어 있으며, 사용 예시는 다음과 같다
``` python
from models.inverse_isp import InverseISP

InverseISP().from_path(
    "source.png",
    "target_low_light.png",
    seed=42,
    strength=0.5,
)
```
---
```bash
python models/inverse_isp.py ./datasets/example/ ./datasets/example_darken --seed 42 --strength 0.5
```
`--strength`는 0~1 범위이며, 0은 원본을 유지하고 1은 거의 검은색에 가까운
강한 어두움 효과를 적용한다. 생략하면 기본값 1이 적용된다.

`dataset.yaml` 예시:

```yaml
dataset:
  root: ./data
  samples:
    - id: cabin_undistorted_0
      source_gaussian_ply: ./cabin_undistorted/base_gaussians_2_8.ply
      colmap_model: ./cabin_undistorted/sparse
      input_views:
        - image: ./cabin_undistorted/images/image002.jpg
          camera_id: 1  # images.txt의 IMAGE_ID
        - image: ./cabin_undistorted/images/image008.jpg
          camera_id: 8
      render_view:
        camera_id: 5

    - id: chair_undistorted_0
      source_gaussian_ply: ./chair_undistorted/base_gaussians_3_9.ply
      colmap_model: ./chair_undistorted/sparse
      input_views:
        - image: ./chair_undistorted/images/image003.jpg
          camera_id: 3
        - image: ./chair_undistorted/images/image009.jpg
          camera_id: 9
      render_view:
        camera_id: 5
```
`source_gaussian_ply` 경로를 지정할 때, 매칭된 두 개의 이미지 번호를 명시하도록 한다.


데이터를 등록할 때 다음 조건을 확인한다.

- 모든 경로는 `dataset.root`를 기준으로 해석한다.
- source PLY의 SH degree와 trainer의 `--sh_degree`가 동일해야 한다.
- `camera_json` 또는 COLMAP TXT 모델 경로인 `colmap_model` 중 하나를 지정한다.
- `input_views[].camera_id`와 `render_view.camera_id`는 이미지의 COLMAP `IMAGE_ID`를 사용한다.
- 입력 이미지 경로는 undistort된 COLMAP 이미지의 실제 파일 경로를 지정하고, 이미지 크기는 COLMAP 카메라 크기와 같아야 한다.

JSON 대신 `sparse/cameras.txt`와 `sparse/images.txt`를 직접 읽을 수 있다.
undistort된 이미지의 `PINHOLE` 및 `SIMPLE_PINHOLE` 카메라 모델을 지원하며,
COLMAP의 world-to-camera pose는 렌더러 입력인 camera-to-world pose로 변환한다.

COLMAP의 카메라 `CAMERA_ID`가 아니라 각 등록 이미지의 `IMAGE_ID`를 YAML의
`camera_id`에 지정한다. 동일한 CAMERA_ID를 여러 이미지가 공유할 수 있기
때문이다.

현재 `dataset.yaml`은 데이터 포맷을 정의하는 단계이며, dataset 전체를 읽는
DataLoader는 `train/injection_dataset.py`에 구현되어 있다. `len(dataset)`은
view 수가 아니라 scene 수이며, 실제 학습에서는 예측 SH로 source geometry를
render한 결과와 각 샘플의 pseudo-GT image 사이의 loss를 계산한다.

**🔹 Dataset과 DataLoader 사용 예시:**

```python
from train.injection_dataset import build_injection_dataloader, encode_prompt_batch

dataloader = build_injection_dataloader(
  "dataset.yaml",
  sh_degree=3,
  batch_size=2,
  shuffle=True,
  num_workers=0,
  device="cpu",
)

for batch in dataloader:
  condition = encode_prompt_batch(batch, clip_encoder, device="cuda")
  for sample, sample_condition in zip(batch, condition):
    input_images = sample["input_images"]  # 항상 5개이며 하나의 scene에 해당
    source_features = sample["source_features"].to("cuda")
    target_images = [image.to("cuda") for image in sample["target_images"]]
    predicted = injection_network(source_features, sample_condition)
```

Gaussian 개수가 sample마다 다를 수 있으므로 기본 `collate_fn`은 tensor를
하나로 쌓지 않고 sample dictionary의 list를 반환한다. 따라서 각 sample의
loss를 계산한 뒤 평균을 내는 방식으로 학습한다. 모든 sample의 Gaussian 수가
같은 경우에도 이 방식은 사용할 수 있으며, 추후 padding 기반 batch 연산으로
최적화할 수 있다.

### 🔶 Train

학습 결과는 `output/injection/injection_network.pt`에 저장된다. 저장된 가중치를
사용해 source PLY에 appearance를 적용하려면:
``` Bash
python main.py --config config.yaml --stage injection --train False
```
Inference는 체크포인트를 불러와 SH feature를 갱신하고
`output/injection/injected_gaussians.ply`에 저장한다.

**5) Gaussian 렌더링**

`render/renderer_splats.py`는 PLY에서 위치, log-scale, quaternion, opacity
logit, SH 계수를 읽고, 카메라 JSON의 camera-to-world pose와 intrinsics를 사용해
이미지를 생성한다. StereoGS의 Python Camera/모델 객체는 사용하지 않는다.
카메라 JSON은 카메라 목록(list) 또는 `{"cameras": [...]}` 형식이며, 각 항목은
`id`, `width`, `height`, `fx`, `fy`, `rotation`(3×3 camera-to-world 회전),
`position`(camera-to-world 위치)을 포함해야 한다. `cx`, `cy`는 선택 항목이고,
생략하면 이미지 중심을 사용한다.

``` Bash
python render/renderer_splats.py \
  --ply output/injection/injected_gaussians.ply \
  --camera_json path/to/cameras.json \
  --mode cameras \
  --output_dir output/rendered/cameras \
  --video output/rendered/cameras.mp4 \
  --device cuda
```

- `render_gaussians()` 함수는 PLY 대신 가우시안 텐서 mapping도 받을 수 있다.
- `xyz`, `scaling`, `rotation`, `opacity`와 SH `features`(또는
`features_dc`/`features_rest`)를 넘기며, 기본적으로 scale은 log-scale,
opacity는 logit으로 해석한다.
- 이미 RGB 색상을 가진 경우에는 SH 대신 `colors` 텐서를 넘길 수 있다.
- 렌더러는 CUDA rasterizer를 사용하므로 `diff_gaussian_rasterization`이 설치된 환경에서 실행한다.


**6) Pseudo-GT image loss**

`train_injection_network.py`는 다음 pseudo-GT image loss를 사용한다.

```text
L = lambda_image * L1(render(SH_pred), pseudo_GT_image)
  + lambda_ssim * (1 - SSIM(render(SH_pred), pseudo_GT_image))
```

---

### 🔶 Baseline Gaussian Reconstruction using MVSplat

Baseline gaussian 구성 전 scene의 `near`, `far` 자동으로 구한 뒤 gaussian 재구성

``` powershell
for scene_name in "cabin" "chair" "fan" "musharna" "office" "table"; do
  echo "Processing scene: $scene_name"

python -m src.scripts.infer_colmap_auto_bounds \
  --colmap-model ./datasets/colmap_based/${scene_name}_undistorted/sparse \
  --images ./datasets/colmap_based/${scene_name}_undistorted/images \
  --checkpoint checkpoints/re10k.ckpt \
  --context-indices 2 8 \
  --target-index 5 \
  --output ../low-light/data/${scene_name}_undistorted/base_render.png \
  --ply-output ../low-light/data/${scene_name}_undistorted/base_gaussians.ply
done
```

```txt
chair: Estimated near=5.64178, far=14.2082
```

`near`, `far` 지정 후 MVSplat 에서 baseline gaussian 재구성 할때:
``` powershell
for scene_name in "cabin" "chair" "fan" "musharna" "office" "table"; do
  echo "Processing scene: $scene_name"
  
  python -m src.scripts.infer_colmap \
    --colmap-model ./datasets/colmap_based/${scene_name}_undistorted/sparse \
    --images ./datasets/colmap_based/${scene_name}_undistorted/images \
    --checkpoint checkpoints/re10k.ckpt \
    --context-indices 2 8 \
    --target-index 5 \
    --near 30 \
    --far 100 \
    --output ../low-light/data/${scene_name}_undistorted/base_render.png \
    --ply-output ../low-light/data/${scene_name}_undistorted/base_gaussians.ply
done
```
---

### 🔶 Gaussian Enhance Network

#### 🔹 Train

프로젝트 루트에서 다음과 같이 학습한다. tqdm은 epoch별 배치 진행률과 현재/평균
loss를 표시한다.

```bash
python train/train_enhance_network.py --dataset_yaml dataset.yaml
```

기본 저장 위치는 `snapshots/`이다. `latest.pth`는 매 epoch 덮어써 저장하고,
`snapshot_iter` epoch 간격 및 마지막 epoch에는 `epoch_XXXX.pth` 체크포인트도
저장한다. 저장 경로와 간격은 `--snapshots_folder`, `--snapshot_iter`로 바꿀 수
있다. 저장된 가중치를 불러와 이어서 학습하려면:

```bash
python train/train_enhance_network.py \
  --dataset_yaml dataset.yaml \
  --load_pretrain \
  --pretrain_dir snapshots/latest.pth
```

#### 🔹 Inference

학습된 네트워크를 baseline Gaussian PLY에 적용하고 지정한 카메라 뷰를 렌더링하려면:

```bash
python infer_enhance.py \
  --baseline_gaussian data/cabin_undistorted/base_gaussians.ply \
  --checkpoint snapshots/latest.pth \
  --colmap_model data/cabin_undistorted/sparse \
  --camera_ids 7 \
  --images_dir data/cabin_undistorted/images \
  --output_dir output/cabin_7
```

MVSplat에서 Gaussian PLY를 내보낼 때 위치를 중심 이동·정규화·회전하므로,
원본 COLMAP pose를 그대로 넣으면 PLY와 좌표계가 달라 검은 이미지가 나올 수 있다.
수정된 `mvsplat/src/scripts/infer_colmap.py`는 PLY와 함께 변환된 카메라 pose를
`base_gaussians.cameras.json`에 저장한다. 따라서 기존 baseline PLY는 MVSplat
추론 명령을 다시 실행해 sidecar를 만든 뒤 이 명령을 사용한다. `--colmap_model`은
COLMAP 모델 디렉터리(`cameras.txt`, `images.txt` 포함)를 지정하며, 렌더링에는
PLY 옆 sidecar를 사용한다. 원본 카메라 이미지 디렉터리는 기본적으로 COLMAP
모델 폴더의 부모에 있는 `images`를 사용하며, 다른 경로라면 `--images_dir`로
지정한다. 카메라 이미지의 실제 크기를 확인해 해당 해상도로 렌더링하고, 크기가
카메라 메타데이터와 다르면 intrinsics도 함께 비례 조정한다. JSON 카메라를 직접
제공하려면 `--camera_json`을 쓰고, 원본 이미지 크기 참조를 위해 `--images_dir`도
지정할 수 있다. JSON의 pose가 PLY 좌표계 기준인지 확인해야 한다.

`--camera_ids`를 생략하면 지정한 카메라 입력의 모든 뷰를 렌더링한다. 추론 결과는
`output_dir/enhanced_gaussians.ply`와 `output_dir/render_<camera_id>.png`에
저장된다. 기본 checkpoint는 `snapshots/latest.pth`이며, `--output_ply`,
`--chunk_size`, `--device`로 저장 위치·추론 chunk 크기·CUDA 장치를 지정할 수 있다.
추론 입력 색상은 source PLY의 DC SH를 RGB로 변환해 사용하며, geometry와 나머지
SH 계수는 변경하지 않는다. 렌더링에는 `render/renderer_splats.py`와 동일하게
CUDA `diff_gaussian_rasterization` 환경이 필요하다.

---
