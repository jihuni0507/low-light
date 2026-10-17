# low-light

## Implementation Logic
``` Text
1. Reconstruct Gaussian splats with StereoGS
2. Generate text/image condition vector by CLIP encoder
3. Input Gaussians and condition vector into injection network
4. Freeze Gaussians' geometry, update only SH features
5. Loss - computed between rendered SH output and inverse-ISP pseudo-GT images

```
## Overall Pipeline
<img width="995" height="963" alt="image" src="https://github.com/user-attachments/assets/f24ab313-1b85-4c14-bfd5-0f7054fb2636" />


## 사용방식
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

### 🔶 Dataset 구성 가이드

Injection network 학습용 데이터셋은 밝은 source Gaussian과 inverse ISP로
생성한 5개의 어두운 pseudo-GT view를 하나의 scene으로 묶는다.

```text
bright source Gaussian PLY + 5 inverse-ISP pseudo-GT views + prompt
                         |
                         v
       dark predicted Gaussian
```

각 샘플은 다음 정보를 가진다.

| 항목 | 설명 |
| --- | --- |
| `views` | inverse ISP로 생성한 어두운 pseudo-GT 이미지 5장의 경로 |
| `camera_json` | StereoGS가 저장한 `cameras.json` 경로 |
| `camera_ids` | 렌더링할 camera id 목록. 생략하면 첫 5개 사용 |
| `prompt` | 적용할 appearance 변환 방향 텍스트 |

    각 샘플은 자신의 밝은 입력 Gaussian PLY와 대응하는 pseudo-GT view 5장을
  함께 가진다. 서로 다른 장면을 추가할 때는 샘플마다 `source_gaussian_ply`를
  별도로 기록한다.

#### 🔸 어두운 이미지 생성
```bash
python models/inverse_isp.py ./datasets/example/ ./datasets/example_darken --seed 42 --strength 0.5
```
`--strength`는 0~1 범위이며, 0은 원본을 유지하고 1은 거의 검은색에 가까운
강한 어두움 효과를 적용한다. 생략하면 기본값 1이 적용된다.

권장 디렉터리 구조:

```text
low-light/
├── dataset.yaml
└── datasets/
    └── inputs/
      ├── 1.png
      ├── 2.png
      ├── 3.png
      ├── 4.png
      └── 5.png
```

`dataset.yaml` 예시:

```yaml
dataset:
  name: low_light_injection
  root: "."
  num_views: 5
  samples:
    - id: office1_low_light
      source_gaussian_ply: "../StereoGS/output/LLFF/office1_5views/point_cloud/iteration_30000/point_cloud.ply"
      views:
        - input_image: "datasets/inputs/office1_dark_view1.png"
        - input_image: "datasets/inputs/office1_dark_view2.png"
        - input_image: "datasets/inputs/office1_dark_view3.png"
        - input_image: "datasets/inputs/office1_dark_view4.png"
        - input_image: "datasets/inputs/office1_dark_view5.png"
      camera_json: "../StereoGS/output/LLFF/office1_5views/cameras.json"
      camera_ids: [0, 1, 2, 3, 4]
      prompt: "low-light"
```

데이터를 등록할 때 다음 조건을 확인한다.

- 모든 경로는 `dataset.root`를 기준으로 해석한다.
- source PLY의 SH degree와 trainer의 `--sh_degree`가 동일해야 한다.
- pseudo-GT 이미지는 source reconstruction camera와 동일한 view 순서여야 한다.
- Rendered image loss에는 source reconstruction의 StereoGS `cameras.json`이 필요하다.
- 각 sample에는 appearance 변환 방향을 나타내는 `prompt`가 있어야 한다.
- 각 prompt는 CLIP 텍스트 인코더를 통과해 조건 벡터가 된다.
- 각 scene은 반드시 정확히 5개의 view를 가져야 한다.

현재 `dataset.yaml`은 데이터 포맷을 정의하는 단계이며, dataset 전체를 읽는
DataLoader는 `train/injection_dataset.py`에 구현되어 있다. `len(dataset)`은
view 수가 아니라 scene 수이며, 실제 학습에서는 예측 SH로 source geometry를
render한 결과와 각 샘플의 pseudo-GT image 사이의 loss를 계산한다.

Dataset과 DataLoader 사용 예시:

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

**1) 전체 파이프라인 실행**
``` Bash
export PYTHONPATH=$PWD
python main.py --config config.yaml
```

**2) CLIP만 단독 실행**
``` Bash
export PYTHONPATH=$PWD
python main.py --config config.yaml --stage clip
```~~

**3) StereoGS base reconstruction**
``` Bash
cd ../StereoGS
python train.py \
  -s /path/to/scene \
  -m ./output/llff/fern_5views \
  --dataset_name LLFF \
  --n_views 5 \
  --resolution 8 \
  --eval \
  --use_mvs_init \
  --mvs_model_name MVSAnywhere \
  --sh_degree 3
```

StereoGS saves the handoff file at
`<model_path>/point_cloud/iteration_<N>/point_cloud.ply`. Set
`injection.gaussian_ply` in `config.yaml` to that file before running the
injection stage. The `--sh_degree 3` option is required because this project
expects 45 non-DC SH values per Gaussian.

**4) SH injection 학습**
``` Bash
export PYTHONPATH=$PWD
python train/train_injection_network.py \
  --dataset_yaml dataset.yaml \
  --sh_degree 3 \
  --device cuda \
  --lambda_sh 0 \
  --lambda_image 1.0 \
  --lambda_ssim 0.2
```

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

`render_gaussians()` 함수는 PLY 대신 가우시안 텐서 mapping도 받을 수 있다.
`xyz`, `scaling`, `rotation`, `opacity`와 SH `features`(또는
`features_dc`/`features_rest`)를 넘기며, 기본적으로 scale은 log-scale,
opacity는 logit으로 해석한다. 이미 RGB 색상을 가진 경우에는 SH 대신
`colors` 텐서를 넘길 수 있다. 렌더러는 CUDA rasterizer를 사용하므로
`diff_gaussian_rasterization`이 설치된 환경에서 실행한다.

원형 orbit 경로로 여러 프레임을 생성하려면:

``` Bash
python render/renderer_splats.py \
  --ply output/injection/injected_gaussians.ply \
  --camera_json path/to/cameras.json \
  --mode orbit \
  --frames 120 \
  --fps 30 \
  --output_dir output/rendered/orbit \
  --video output/rendered/orbit.mp4 \
  --device cuda
```

**6) Pseudo-GT image loss**

`train_injection_network.py`는 다음 pseudo-GT image loss를 사용한다.

```text
L = lambda_image * L1(render(SH_pred), pseudo_GT_image)
  + lambda_ssim * (1 - SSIM(render(SH_pred), pseudo_GT_image))
```

StereoGS의 `render.py`를 별도 프로세스로 실행하지 않고 동일한
`gaussian_renderer.render()`를 직접 호출한다. 그래야 렌더링 loss의 gradient가
injection network까지 전달된다.

``` Bash
python train/train_injection_network.py \
  --dataset_yaml dataset.yaml \
  --lambda_sh 0.0 \
  --lambda_image 1.0 \
  --lambda_ssim 0.2
```

---
**7) Text-Conditioned Training**
``` Bash
python main.py --config config.yaml clip.text_prompt="low-light scene"
```
또는 `config.yaml`을 직접 수정할 수도 있다.

**8) Image-Conditioned Training**
``` Bash
python - <<'PY'
import yaml
from pathlib import Path

p = Path("config.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["clip"]["image_path"] = "assets/example.png"
cfg["pipeline"]["run_clip_encode"] = True
cfg["pipeline"]["run_injection"] = True
p.write_text(yaml.safe_dump(cfg, allow_unicode=True))
PY

python main.py --config config.yaml
```

---
### 🔶 Gaussian Reconstruction using StereoGS
Recommended commmand
```bash
python train.py \
  -s /path/to/scene \
  -m ./output/LLFF/scene_3views \
  --dataset_name LLFF \
  --n_views 3 \
  --resolution 8 \
  --eval \
  --sh_degree 3 \
  --use_mvs_init \
  --mvs_model_name MVSAnywhere \
  --opacity_decay \
  --opacity_decay_factor 0.99 \
  --grad_sensitivity 0.5 \
  --generate_binocular_view \
  --start_generate_binocular_view_iters 20000 \
  --cam_trans_dist 4.0 \
  --binocular_model_type FoundationStereo \
  --binocular_depth \
  --lambda_binocular 1.0 \
  --dropout_factor 0.3 \
  --init_pcd_downsample 0.1
```
Required structure of `/path/to/acene`
``` bash
scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```
Example for `office1`:
``` bash
python train.py \
  -s /home/work/test2/datasets/office1 \
  -m ./output/LLFF/office1_5views \
  --dataset_name LLFF \
  --n_views 3 \
  --resolution 8 \
  --eval \
  --sh_degree 3
```
---
### 🔶 `train_injection_network.py`
Dataset YAML에 등록한 밝은 source Gaussian, inverse-ISP pseudo-GT 이미지,
이미지별 prompt로 학습한다.

#### 🔸 Dataset Training

``` Bash
export PYTHONPATH=$PWD
python train/train_injection_network.py \
  --dataset_yaml dataset.yaml \
  --sh_degree 3 \
  --batch_size 2 \
  --epochs 10 \
  --device cuda
```

`batch_size`는 한 optimization step에 사용할 dataset sample 수다. 각 sample의
Gaussian 개수는 다를 수 있으므로 DataLoader는 sample list를 반환하고, 학습
루프는 sample별 render loss를 계산한 뒤 batch 평균으로 업데이트한다. Prompt는
각 batch에서 CLIP으로 한 번에 인코딩한다.

새 학습 스크립트는 dataset 기반 입력만 지원한다. 단일 PLY와 단일 캡션을
사용하는 기존 방식 대신, 모든 학습 sample을 `dataset.yaml`에 등록한 뒤
`--dataset_yaml`로 전달한다.

---
### 🔶 Synthesizing Dark Images - Inverse ISP
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
