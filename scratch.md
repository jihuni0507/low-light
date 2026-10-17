### 🟦 Checklist

- [ ] **Dark Image 생성 로직 점검** - `enhance_dark.py`, `generate_dark.py` 로직 및 성능 비교
- [ ] JPG 이미지 darken 이후 zero-dce로 복원했을 때 결과가 이상적이지 않음, PNG 이미지로 했을 때도 그런지 확인

- [ ] **Per-scene Optimization** 가장 먼저 10장 이상의 enhanced image 바탕 3DGS로 견고하게 geometry를 재건한 이후에(geometry는 잘 되었다는 가정) 원래대로 두장의 이미지만 입력으로 넣었다고 생각하고 가우시안 컬러 수정하는 네트워크 검증

### TODO

아니요. rendered_img = None을 실제 렌더 호출로 바꾸는 것은 필요한 작업이지만, 그것만으로 학습이 완성되지는 않습니다. 현재 확인한 train_enhance_network.py에는 렌더링 외에도 아직 연결해야 할 부분이 있습니다.

최소한 학습 단계마다 다음 흐름이 이어져야 합니다.

배치에서 scene의 PLY 경로와 세 번째 view의 카메라 정보를 받습니다.
그 PLY를 읽어 base_gaussians를 준비합니다.
GE_net으로 Gaussian의 색 속성을 보정합니다.
보정된 속성을 사용해 세 번째 카메라에서 미분 가능한 renderer를 호출합니다.
렌더 결과로 unsupervised loss를 계산하고, Gaussian-domain loss와 합쳐 backward()합니다.
특히 렌더 결과가 GE_net까지 gradient를 전달해야 합니다. 보정된 색을 detach()하거나, gradient 경로가 끊기는 방식으로 새 Parameter에 복사하면 렌더링 loss가 네트워크를 학습시키지 못합니다. Geometry를 고정하더라도 색 속성에 대한 연산 그래프는 보존되어야 합니다.

그리고 두 입력 이미지가 의도한 네트워크 입력이라면, 지금처럼 GE_net(base_gaussians)만 호출해서는 그 이미지 정보가 네트워크에 전달되지 않습니다. 현재 구조에서는 이미지가 별도 loss에 쓰이거나, 네트워크의 conditioning 입력으로 추가되어야 실제로 활용됩니다.

따라서 Dataset에서 세 번째 카메라 정보를 반환하고, PLY 로드·보정 Gaussian을 renderer에 전달하는 경로까지 먼저 연결해야 합니다. 이 작업들이 이미 되어 있고 rendered_img만 비어 있다면 그 호출부가 마지막 주요 연결점일 수 있지만, 현재 확인된 스크립트에는 base_gaussians = None 등 다른 미구현 부분도 남아 있습니다.
