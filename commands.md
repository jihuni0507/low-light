# Bash Commands for RTX 5090 Environments
## 🔶 Light Reconstruction with StereoGS
#### Sparse View를 보고 light reconstruction 및 `.ply` 파일을 생성하는 단계이다.

##### 4 RTX 5090
``` bash
SCENES="musharna elaina"
GPU_ID=0

for SCENENAME in $SCENES; do 
  for var in {1..19}; do
    echo "========================================"
    echo "Processing SCENE ${SCENENAME}_${var} on GPU ${GPU_ID}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
      -s /workspace/datasets/${SCENENAME}_${var} \
      -m /workspace/StereoGS/output/LLFF/${SCENENAME}_${var}_5views \
      --dataset_name LLFF \
      --n_views 5 \
      --resolution 8 \
      --eval \
      --sh_degree 3 &

    GPU_ID=$(( (GPU_ID + 1) % 4 ))

    if [ $GPU_ID -eq 0 ]; then
      wait
    fi
  done
done

wait
echo "All training jobs are complete!"
```

## 🔶 Training Injection Network

``` bash
cd /home/work/test2/low-light
export PYTHONPATH=$PWD

python train/train_injection_network.py \
  --dataset_yaml dataset.yaml \
  --sh_degree 3 \
  --device cuda \
  --epochs 10 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --hidden_dim 256 \
  --gaussian_chunk_size 16384 \
  --lambda_sh 0 \
  --lambda_image 1.0 \
  --lambda_ssim 0.2 \
  --lambda_well_exposure 0.1 \
  --lambda_well_saturated 0.1 \
  --lambda_gain 0.001 \
  --lambda_spatial_smooth 0.01 \
  --lambda_magnitude 0.001 \
  --spatial_smooth_k 16 \
  --save_dir ./output/injection
```

 - 초기에는 `lambda_well_exposure, lambda_well_saturated, lambda_gain` 값을 작게 주고 이후에 loss를 강하게 줄 수 있다
 - `lambda_sh = 0`은 현새 pseudo-GT 구조에서 필수

 ## 🔶 Inference
 #### 학습에서 보지 못한 scene에 대한 가우시안에 trained injection network의 가중치를 적용해 어둡게 만든다
 ``` bash
cd /home/work/test2/low-light

python main.py \
  --config config.yaml \
  --stage injection \
  --train False
 ```

 코드 실행 이후 아래 경로에서 darkened gaussians 파일이 생성된다.
 ``` bash
./output/injection/injected_gaussians.ply
 ```

  ## 🔶 Rendering
  #### `cameras.json`을 기준으로 렌더링
  ```bash
  python renderes/renderer_splats.py \
  --ply output/injection/injected_gaussians_3.ply \
  --camera_json /workspace/StereoGS/output/LLFF/elaina_2_5views/cameras.json \
  --mode cameras \
  --output_dir output/rendered/cameras_2 \
  --video output/rendered/cameras_2.mp4 \
  --sh_degree 3
  ```
