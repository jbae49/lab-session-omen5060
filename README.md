# RGB 영상에서 G1 모방 정책까지 — OMEN 16 / RTX 5060 실습

이 저장소는 처음 3D 사람 모델과 로봇 리타깃을 접하는 학부생을 위한 실습이다.
정식 파이프라인은 RGB 영상에서 사람의 3D 움직임을 복원한 뒤 Unitree G1 29DoF
동작으로 바꾸지만, **이번 수업에서는 미리 복원된 사람 motion부터 시작한다.**

지원 기준은 다음 한 가지다.

- HP OMEN 16 (2025), Ryzen AI, NVIDIA RTX 5060 Laptop GPU 8GB
- Ubuntu 22.04 네이티브 설치 권장
- NVIDIA driver, Conda/Miniconda, 인터넷 연결 필요
- Windows 네이티브는 지원하지 않는다. WSL2는 가능하지만 렌더링 설정이 더 어렵다.

## 이번 수업에서 할 것: PromptHMR 설치 없이 리타깃부터

PromptHMR 전체 환경은 Detectron2, SAM2, ViTPose, 여러 checkpoint와 Google Drive
다운로드가 필요하다. 수업 시간에는 설치 문제에 시간을 쓰지 않도록, 교사가
PromptHMR/SMPL-X로 미리 복원한 다음 파일을 제공한다.

```text
sample_data/input/jump_smplx_joint_targets_30fps.npz
```

이 파일에는 GMR에 필요한 사람 관절 14개의 frame별 위치와 방향만 들어 있다.
SMPL/SMPL-X body-model 파일이나 PromptHMR checkpoint를 포함하지 않으며, 이
수업 경로에서는 둘 다 필요하지 않다.

### 저장소에서 제공하는 사람 원본 자료

```text
sample_data/input/jump.mov
    원본 RGB 영상

sample_data/expected/jump_prompthmr_overlay.mp4
    원본 영상 위에 SMPL-X mesh를 겹쳐 녹화한 확인 영상

sample_data/expected/jump_smplx_gmr_input.npz
    PromptHMR에서 복원한 raw SMPL-X pose/betas/root motion

sample_data/input/jump_smplx_joint_targets_30fps.npz
    위 SMPL-X motion을 body model 없이 GMR에 넣을 수 있도록 만든 사람 joint motion
```

**G1 qpos, G1 50Hz reference, G1 녹화 영상은 정답으로 제공하지 않는다.** 학생이
직접 GMR을 실행해 `outputs/jump_retarget_lab/` 아래에 생성해야 한다.

### 수업용 설치

```bash
sudo apt update
sudo apt install -y git ffmpeg build-essential libgl1 libegl1

bash scripts/install_retarget_lab.sh --execute
```

### 수업용 실행 — 한 줄

```bash
bash scripts/run_retarget_lab.sh
```

이 명령은 제공된 사람 joint NPZ를 GMR로 G1 29DoF에 리타깃하고, 50Hz MjLab
reference와 지면이 포함된 MP4를 만든다. 명령이 끝나면 다음 영상을 반드시 직접
재생해 보자!

```text
outputs/jump_retarget_lab/jump_retarget_lab_g1_retarget.mp4
```

정상 결과의 핵심은 `qpos.shape == (T,36)`, 관절 29개, NaN/Inf 없음이다. 영상에서
로봇의 앞뒤 방향, 무릎, 팔, 발의 지면 관통과 floating을 확인한다.

여기까지가 필수 수업 범위다. 아래 내용은 원래 영상에서 사람 motion을 만드는
**정식 전체 파이프라인 설명과 선택 확장 실습**이다.

---

## 정식 전체 파이프라인은 어떻게 구성되는가? (선택)

## 우리가 만들 결과

```text
RGB video
  → PromptHMR: 사람 선택·추적 + SMPL-X 3D 사람 motion
  → GMR: SMPL-X → Unitree G1 29DoF retarget
  → MJLab: 50 Hz reference motion 생성
  → PPO: G1 motion-imitation policy 학습
  → checkpoint + rollout MP4
```

### 용어를 먼저 이해하기

**SMPL**은 사람의 체형과 관절 자세를 숫자로 나타내는 3D body model이다.
**SMPL-X**는 SMPL에 손과 얼굴 등 더 자세한 구조를 추가한 모델이다. PromptHMR은
SMPL-X를 출력한다. G1에는 손가락 actuator가 없으므로 이 실습의 로봇 동작은
손목까지만 사용한다.

**리타깃(retargeting)**은 사람 관절 회전을 단순 복사하는 일이 아니다. 사람과
로봇의 팔다리 길이와 관절 구조가 다르므로, GMR이 사람의 pelvis, knee, ankle,
shoulder, elbow, wrist 위치를 따라가도록 매 frame inverse kinematics를 푼다.

**29DoF G1**은 다리 12 + 허리 3 + 팔 14개의 actuator를 사용한다. floating
base의 위치 3개와 quaternion 4개를 더하면 GMR `qpos` 한 frame의 폭은
`7 + 29 = 36`이다. 이 실습의 29DoF checkpoint를 23DoF 로봇에 사용할 수 없다.

## 1. 노트북 준비

전원 어댑터를 연결하고 OMEN Gaming Hub에서 Performance mode와 discrete GPU를
선택한다. 절전/자동 sleep을 끈다. 터미널에서 확인한다.

```bash
sudo apt update
sudo apt install -y git wget unzip ffmpeg build-essential libgl1 libegl1

nvidia-smi
conda --version
ffmpeg -version
git --version
```

Conda가 없다면 [Miniconda 공식 설치 안내](https://docs.conda.io/projects/miniconda/en/latest/)를
따른다. NVIDIA driver는 Ubuntu가 RTX 5060에 권장하는 최신 proprietary driver를
사용한다. 별도의 시스템 CUDA toolkit을 먼저 설치할 필요는 없다.

RTX 5060은 Blackwell(`sm_120`) GPU다. 따라서 예전 PromptHMR 설치법의
PyTorch 2.6/cu126 wheel을 사용하지 않는다. 이 저장소의 설치 스크립트는
PyTorch 2.7.1/cu128을 사용하고 Detectron2를 이 GPU용으로 source build한다.

권장 여유 공간은 35GB 이상이다. Git 저장소 자체는 작지만 Conda 환경과 neural
network weight, SMPL/SMPL-X model은 설치 후 별도로 커진다.

## 2. PromptHMR 포함 전체 환경 설치 — 선택 사항

이 단계는 수업용 리타깃 실습에는 필요 없다. 원본 영상에서 SMPL-X를 직접
복원해 보고 싶은 경우에만 저장소 최상위 폴더에서 실행한다.

```bash
bash scripts/install_omen5060.sh --execute
```

다음 세 환경을 서로 분리해서 만든다.

| 환경 | 역할 |
|---|---|
| `phmr_omen5060` | 영상에서 SMPL-X 복원 |
| `gmr_omen5060` | SMPL-X를 G1으로 리타깃 |
| `unitree_mjlab_omen5060` | reference 변환, PPO 학습, 렌더링 |

설치는 인터넷 속도와 source build에 따라 오래 걸릴 수 있다. 오류가 나면 마지막
20줄만 지우지 말고 전체 오류를 저장해 조교에게 전달한다.

설치 도중 오류가 나도 Conda 환경을 삭제할 필요가 없다. 먼저 저장소를 업데이트한
뒤 같은 설치 명령을 다시 실행하면 기존 환경과 이미 받은 핵심 weight를 재사용한다.

```bash
git pull
bash scripts/install_omen5060.sh --execute
```

## 3. 전체 파이프라인용 SMPL과 SMPL-X 다운로드 — 선택 사항

이 두 모델은 라이선스 때문에 GitHub에 넣거나 친구에게 파일을 전달할 수 없다.
각 학생이 공식 사이트에서 계정을 만들고 동의해야 한다.

1. [SMPL-X 공식 다운로드](https://smpl-x.is.tue.mpg.de/download.php)에서
   **SMPL-X v1.1 model archive** (`models_smplx_v1_1.zip`)를 선택한다.
2. [SMPL 공식 다운로드](https://smpl.is.tue.mpg.de/download.php)에서
   **SMPL for Python v1.1.0** (`SMPL_python_v.1.1.0.zip`)을 선택한다.
3. 아직 ZIP을 받지 않았다면 아래 권장 방법을 실행하고 본인의 계정 정보를 입력한다.

```bash
cd repos/PromptHMR
bash scripts/fetch_smplx.sh
cd ../..
bash scripts/copy_body_models_to_gmr.sh
```

### 브라우저로 ZIP을 이미 받은 경우: 정확히 어디에 넣나?

ZIP 파일 자체를 저장소 안에 넣는 것이 아니다. 압축 안의 모델 파일을 아래 최종
경로로 복사한다.

```text
SMPL-X ZIP의 SMPLX_NEUTRAL.npz → repos/PromptHMR/data/body_models/smplx/SMPLX_NEUTRAL.npz
SMPL-X ZIP의 SMPLX_MALE.npz    → repos/PromptHMR/data/body_models/smplx/SMPLX_MALE.npz
SMPL-X ZIP의 SMPLX_FEMALE.npz  → repos/PromptHMR/data/body_models/smplx/SMPLX_FEMALE.npz

SMPL ZIP의 neutral PKL → repos/PromptHMR/data/body_models/smpl/SMPL_NEUTRAL.pkl
SMPL ZIP의 male PKL    → repos/PromptHMR/data/body_models/smpl/SMPL_MALE.pkl
SMPL ZIP의 female PKL  → repos/PromptHMR/data/body_models/smpl/SMPL_FEMALE.pkl
```

Linux 명령으로 한 번에 풀고 복사하려면 `/path/to/...zip` 두 곳을 실제 다운로드
경로로 바꿔 실행한다.

```bash
BODY_MODEL_TMP="$(mktemp -d)"
unzip /path/to/models_smplx_v1_1.zip -d "$BODY_MODEL_TMP/smplx"
unzip /path/to/SMPL_python_v.1.1.0.zip -d "$BODY_MODEL_TMP/smpl"

mkdir -p repos/PromptHMR/data/body_models/{smplx,smpl}
cp "$BODY_MODEL_TMP"/smplx/models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz \
  repos/PromptHMR/data/body_models/smplx/
cp "$BODY_MODEL_TMP"/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
  repos/PromptHMR/data/body_models/smpl/SMPL_NEUTRAL.pkl
cp "$BODY_MODEL_TMP"/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl \
  repos/PromptHMR/data/body_models/smpl/SMPL_MALE.pkl
cp "$BODY_MODEL_TMP"/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl \
  repos/PromptHMR/data/body_models/smpl/SMPL_FEMALE.pkl
```

수동 압축 해제 후에도 PromptHMR 보조 변환 파일을 받아야 한다.

```bash
cd repos/PromptHMR
conda run -n phmr_omen5060 gdown --folder -O ./data/ \
  'https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7'
conda run -n phmr_omen5060 gdown -O ./data/body_models/smplx/ \
  1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk
cd ../..
bash scripts/copy_body_models_to_gmr.sh
```

마지막 명령이 SMPL-X 세 파일을 `repos/GMR/assets/body_models/smplx/`에도 실제
파일로 복사한다. ZIP은 다운로드 폴더에 그대로 두거나 삭제해도 실행에는 영향이 없다.

PromptHMR은 SMPL-X pose를 만들기 때문에 SMPL-X가 핵심이다. SMPL도 scene export와
SMPL-X/SMPL 변환 보조 데이터에 사용되므로 둘 다 준비한다. 이 저장소는 symbolic
link를 만들지 않고 GMR 폴더에 실제 파일을 복사한다.

정상 구조:

```text
repos/PromptHMR/data/body_models/
├── smpl/SMPL_NEUTRAL.pkl
├── smpl/SMPL_MALE.pkl
├── smpl/SMPL_FEMALE.pkl
├── smplx/SMPLX_NEUTRAL.npz
├── smplx/SMPLX_MALE.npz
├── smplx/SMPLX_FEMALE.npz
├── smplx2smpl.pkl
└── smplx2smpl_joints.npy
```

더 자세한 파일 설명은 [모델과 라이선스](docs/MODELS_AND_LICENSES.md)에 있다.

## 4. 전체 파이프라인을 한 줄로 검사하기

SMPL/SMPL-X 다운로드까지 끝났으면 다음 한 줄을 실행한다.

```bash
bash scripts/test_everything_omen5060.sh
```

이 명령은 다음을 실제로 모두 수행한다.

1. 세 Conda 환경과 필수 파일 검사
2. `jump.mov`에서 지정한 한 사람을 PromptHMR로 복원
3. 사람 mesh가 영상에 겹쳐진 overlay MP4 생성
4. GMR로 G1 29DoF 리타깃
5. G1 reference MP4 생성
6. 32개 병렬 환경으로 PPO 20 iteration smoke training
7. 방금 생긴 checkpoint로 rollout 생성
8. rollout MP4 생성

명령이 성공하면 마지막에 세 경로가 출력된다. 반드시 VLC나 파일 탐색기로
직접 열어보자!

```text
outputs/omen5060_acceptance/prompthmr/overlay.mp4
outputs/omen5060_acceptance/gmr/omen5060_acceptance_g1_reference.mp4
outputs/omen5060_acceptance/rl/omen5060_acceptance_smoke_rollout.mp4
```

20 iteration은 학습 품질 평가용이 아니라 설치와 데이터 연결을 검사하는
**smoke test**다. 마지막 로봇이 아직 잘 따라하지 못해도 오류가 아니다. 첫 번째
영상에서는 mesh가 원하는 사람 위에 있는지, 두 번째 영상에서는 G1의 앞뒤와 발이
맞는지를 먼저 확인한다.

출력 폴더를 새로 만들고 싶다면 다음처럼 이름을 바꾼다.

```bash
NAME=my_test bash scripts/test_everything_omen5060.sh
```

## 5. 각 단계를 따로 배우기

### A. PromptHMR: 여러 사람 중 한 명 고르기

PromptHMR이 모든 사람을 무조건 완벽히 복원하는 것은 아니다. 첫 선택 frame에서
detector가 사람 후보를 만들고, 우리가 고른 사람을 SAM2가 앞뒤 frame으로 추적한다.
`jump` 샘플에서는 화면 앞 점퍼의 몸통 좌표 `(426.5, 294)`를 사용한다.

```bash
conda run --no-capture-output -n phmr_omen5060 \
  python scripts/01_extract_prompthmr.py \
  --video sample_data/input/jump.mov \
  --output outputs/jump/prompthmr \
  --static-camera --start-frame 0 \
  --target-x 426.5 --target-y 294 \
  --num-people 1 --debug-masks
```

8GB VRAM을 위해 기본값은 입력 높이 640, segmentation batch 2, PromptHMR batch 1이다.
모델 자체의 896 입력 crop은 바꾸지 않는다. OOM이 나면 다른 GPU 프로그램과
브라우저를 닫고 다시 실행한다.

먼저 `outputs/jump/prompthmr/init_masks.jpg`를 연다. 색칠된 사람이 점퍼인지
확인한다. 잘못된 사람이라면 학습까지 진행하지 않는다.

사람이 첫 frame에서 가려졌다면 잘 보이는 frame을 골라 `--start-frame 120`처럼
바꾸고 그 frame의 몸통 `(x,y)`를 준다. tracker는 그 frame에서 앞과 뒤 방향으로
전파한다. 긴 가림, 화면 밖 이탈, 비슷하게 겹친 사람은 identity switch를 일으킬 수
있으므로 overlay 확인이 필수다.

카메라가 고정된 수업 영상에는 `--static-camera`를 사용한다. 이 RTX 5060 수업
profile은 static camera만 설치·검증한다. 움직이는 카메라는 DROID-SLAM/DPVO 계열의
추가 CUDA extension과 별도 검증이 필요해 이 최소 저장소 범위에 넣지 않았다.

### B. 사람 mesh overlay 만들기

```bash
conda run --no-capture-output -n phmr_omen5060 \
  python scripts/05_render_prompthmr_overlay.py \
  --video sample_data/input/jump.mov \
  --results outputs/jump/prompthmr/results.pkl \
  --output outputs/jump/prompthmr/overlay.mp4
```

이제 `overlay.mp4`를 직접 재생하자. 다음을 눈으로 찾는다.

- mesh가 다른 사람에게 순간이동하지 않는가?
- 앞/뒤 방향이 뒤집히지 않았는가?
- 무릎과 발이 원본 사람과 비슷한가?
- 가려진 구간 뒤에 같은 사람으로 돌아오는가?

### C. GMR로 G1 29DoF 리타깃

```bash
conda run --no-capture-output -n phmr_omen5060 \
  python scripts/02_export_smplx_for_gmr.py \
  --input outputs/jump/prompthmr/results.pkl \
  --output outputs/jump/gmr/jump_smplx.npz \
  --video sample_data/input/jump.mov

conda run --no-capture-output -n gmr_omen5060 \
  python scripts/03_retarget_gmr.py \
  --input outputs/jump/gmr/jump_smplx.npz \
  --output outputs/jump/gmr/jump_g1_29dof_qpos.npz --fps 30
```

정상 출력은 `(T, 36)` qpos이고 순서는
`root_xyz(3), root_quaternion_wxyz(4), joint_angles(29)`다.

### D. 50 Hz reference와 G1 영상 만들기

```bash
conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python scripts/04_prepare_mjlab_motion.py \
  --input outputs/jump/gmr/jump_g1_29dof_qpos.npz \
  --output outputs/jump/mjlab/jump_g1_29dof_50hz.npz --fps 50

cd repos/unitree_rl_mjlab
MUJOCO_GL=egl conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python scripts/render_motion_npz.py \
  --motion-file ../../outputs/jump/mjlab/jump_g1_29dof_50hz.npz \
  --output ../../outputs/jump/gmr/jump_g1_reference.mp4
cd ../..
```

`jump_g1_reference.mp4`를 꼭 재생한다. 발이 지면을 뚫거나, root가 떠 있거나,
관절이 뒤집히면 reference부터 수정해야 한다. PPO가 나쁜 reference를 좋은 춤으로
자동 수정해 주지는 않는다.

### E. RTX 5060에서 학습

```bash
# 설치 확인: 32 env, 20 iteration
bash scripts/train_mimic.sh smoke outputs/jump/mjlab/jump_g1_29dof_50hz.npz

# 수업 중 loss/reward 경향 확인: 128 env, 1,000 iteration
LOGGER=wandb bash scripts/train_mimic.sh quick \
  outputs/jump/mjlab/jump_g1_29dof_50hz.npz

# 장기 학습 시작값: 256 env, 60,000 iteration
LOGGER=wandb bash scripts/train_mimic.sh full \
  outputs/jump/mjlab/jump_g1_29dof_50hz.npz
```

8GB laptop GPU에서 4096 env는 기본값으로 사용하지 않는다. 256에서도 OOM이면
`scripts/train_mimic.sh`의 full env 수를 128로 낮춘다. 반대로 VRAM 사용량을 확인한
뒤 512를 시험할 수는 있지만, 발열과 thermal throttling을 함께 관찰해야 한다.

```bash
watch -n 2 nvidia-smi
```

이 저장소의 4090 실험은 4096 env에서 30k당 약 12~18시간이었지만, 8GB RTX 5060
노트북은 env 수가 훨씬 작고 전력 제한도 있어 그대로 환산할 수 없다. 첫 1,000
iteration 시간을 직접 측정한 뒤 팀별 예상 시간을 계산한다. 완전한 50k~60k 학습은
수업 한 회보다 오래 걸릴 수 있다.

## 6. 새 영상을 가져왔을 때

고정 카메라 영상, 사람이 전신으로 잘 보이는 짧은 clip부터 시작한다.

```bash
START_FRAME=120 TARGET_X=720 TARGET_Y=350 \
  bash scripts/run_jump_pipeline.sh /absolute/path/new_video.mp4 my_motion
```

이 wrapper는 새 결과를 `outputs/my_motion/`에 만들며 입력이나 기존 결과를
덮어쓰지 않는다. 종료할 때 PromptHMR overlay와 G1 reference MP4 경로를 출력한다.

체크 순서:

1. `init_masks.jpg`: 원하는 한 사람이 선택됐는가?
2. `overlay.mp4`: identity, 앞뒤, 발/무릎이 맞는가?
3. `*_g1_reference.mp4`: G1 관절과 지면 접촉이 자연스러운가?
4. `*.npz`: 29 joint이고 NaN/Inf가 없는가?
5. smoke training이 checkpoint를 저장하는가?
6. 그 뒤에만 quick/full 학습을 시작한다.

## 7. 정책의 입력과 출력

29DoF No-State-Estimation actor(`ClassLab-G1-Tracking-No-State-Estimation`)는 reference joint command, 현재 robot IMU angular
velocity, 현재 joint position/velocity, 이전 action, reference torso orientation을
사용한다. action은 29개 목표 joint-position offset이다. 학습 중 critic은 더 많은
privileged simulation state를 볼 수 있지만, actor는 외부 motion-capture camera를
센서로 사용하지 않는다.

`motion_anchor_ori_b`는 camera yaw가 아니다. 현재 robot base frame에서 표현한
reference anchor orientation이다. reference 좌표계, quaternion 순서, 초기 base
orientation이 잘못되면 이 값이 튀고 정책도 불안정해질 수 있다.

기본 reward에는 root/body position·orientation·linear/angular velocity tracking과
joint/action penalty가 있다. reference가 실제 로봇의 관절 범위와 접촉 물리에 맞는지
먼저 확인해야 reward 조정이 의미가 있다.

## 8. 저장소가 가벼운 이유

Git에는 실행에 필요한 source, G1 mesh, 작은 샘플, 검증용 checkpoint만 둔다.
다음은 포함하지 않고 설치 후 내려받거나 직접 생성한다.

- PromptHMR/SAM2/ViTPose neural-network weights
- 라이선스가 있는 SMPL/SMPL-X model
- Conda 환경과 Python cache
- 학습 log, W&B cache, 생성 MP4/NPZ, 새 checkpoint
- 본 실습과 무관한 robot asset과 bicycle experiment data

따라서 clone은 비교적 작지만 **설치 완료 후 디스크 사용량은 커지는 것이 정상**이다.

## 9. 문제 해결

`CUDA error: no kernel image ... sm_120`이 보이면 잘못된 예전 PyTorch/cu126 환경을
사용한 것이다. 환경 이름이 `phmr_omen5060`인지 확인한다.

`Getting requirements to build wheel ... No module named 'pip'`가 보이면 오래된
`chumpy`를 pip의 격리 build 환경에서 만들려 한 경우다. 최신 저장소의 설치
스크립트는 `chumpy`에 `--no-build-isolation`을 적용한다. 다음처럼 이어서 실행한다.

```bash
git pull
bash scripts/install_omen5060.sh --execute
```

직접 해당 단계만 복구해야 한다면 다음 명령을 사용할 수 있다.

```bash
conda run -n phmr_omen5060 python -m pip install --no-build-isolation \
  'git+https://github.com/Arthur151/chumpy.git@ec5a20f3adc1a105c7a907cfed9a33d7dccbe95f'
```

`CUDA out of memory`이면 브라우저/게임/다른 Python GPU process를 닫고 PromptHMR
batch 1, 학습 env 128로 다시 시도한다. 노트북을 재부팅한 직후 실행하는 것도 좋다.

EGL 렌더링 오류가 나면 NVIDIA driver와 `nvidia-smi`를 먼저 확인한다. WSL2라면
네이티브 Ubuntu와 EGL 동작이 다를 수 있다.

파일별 입출력은 [PIPELINE_IO](docs/PIPELINE_IO.md), 고정 버전은
[REPRODUCIBILITY](docs/REPRODUCIBILITY.md), 수업 진행안은
[LECTURE_PLAN](docs/LECTURE_PLAN.md)을 참고한다.
