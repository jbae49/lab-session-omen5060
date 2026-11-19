# 수업 후 확장: 새 RGB 영상에서 G1 정책 학습까지

이 폴더는 정규 수업의 리타깃 실습과 **완전히 별도인 선택 과정**이다. 수업용
`gmr_omen5060`, `unitree_mjlab_omen5060` 환경은 수정하지 않는다. 여기서는 다음의
독립된 Conda 환경을 사용한다.

```text
phmr_postclass_omen5060
gmr_postclass_omen5060
mjlab_postclass_omen5060
```

목표 파이프라인은 다음과 같다.

```text
새 RGB 영상
  → PromptHMR: 사람 선택·추적, SMPL-X motion 복원
  → overlay MP4로 사람과 mesh 확인
  → GMR: Unitree G1 29DoF 리타깃
  → MjLab: 50 Hz reference 생성 및 MP4 확인
  → PPO motion-imitation 학습
  → checkpoint와 policy rollout 확인
```

## 0. 지원 범위

- HP OMEN 16 (2025), RTX 5060 Laptop GPU 8GB, Ubuntu 22.04
- 짧은 고정 카메라 영상
- 한 명의 전신이 비교적 잘 보이는 영상
- Unitree G1 29DoF

움직이는 카메라, 장시간 가림, 여러 사람이 계속 교차하는 영상은 자동으로 잘 된다고
보장할 수 없다. 첫 실험은 5–15초 정도의 짧은 고정 카메라 영상으로 시작한다.

## 1. 시스템 준비

```bash
sudo apt update
sudo apt install -y git wget unzip ffmpeg build-essential libgl1 libegl1

nvidia-smi
conda --version
```

전원 어댑터를 연결하고 OMEN Performance mode와 discrete GPU를 사용한다. 설치 후
Conda 환경과 checkpoint 때문에 35GB 이상의 여유 공간을 권장한다.

## 2. 별도 전체 환경 설치

저장소 최상위 폴더에서 실행한다.

```bash
bash postclass_full_pipeline/install.sh --execute
```

이 명령은 세 개의 `*_postclass_omen5060` 환경을 만들고 PromptHMR, Detectron2,
SAM2, GMR, MjLab 및 공개 neural-network checkpoint를 설치한다. RTX 5060의
Blackwell `sm_120`을 위해 PyTorch 2.7.1/cu128과 source-built Detectron2를 쓴다.
네트워크와 source build 속도에 따라 오래 걸리며, 중간에 멈추면 같은 명령을 다시
실행해 이어갈 수 있다.

Google Drive quota 또는 접근 오류로 checkpoint 다운로드만 실패했다면 환경을
삭제하지 않는다. 잠시 후 다음 명령만 다시 실행한다.

```bash
bash postclass_full_pipeline/fetch_weights.sh
```

## 3. SMPL과 SMPL-X 모델 준비

이 모델은 라이선스 때문에 GitHub나 설치 스크립트가 대신 배포할 수 없다. 각자 공식
사이트에서 계정을 만들고 라이선스에 동의한다.

1. <https://smpl-x.is.tue.mpg.de/download.php>에서
   **SMPL-X v1.1 model archive** (`models_smplx_v1_1.zip`)
2. <https://smpl.is.tue.mpg.de/download.php>에서
   **SMPL for Python v1.1.0** (`SMPL_python_v.1.1.0.zip`)

브라우저로 받은 두 ZIP을 자동 배치하려면 다음처럼 실행한다.

```bash
bash postclass_full_pipeline/install_body_models.sh \
  ~/Downloads/models_smplx_v1_1.zip \
  ~/Downloads/SMPL_python_v.1.1.0.zip
```

ZIP 원본은 저장소 안으로 복사하지 않는다. 스크립트는 필요한 NPZ/PKL만
`repos/PromptHMR/data/body_models/`와 `repos/GMR/assets/body_models/`에 복사하며
symbolic link를 만들지 않는다.

## 4. 전체 환경 검사

```bash
bash postclass_full_pipeline/verify.sh
```

모든 항목이 `[OK]`여야 한다. `MISSING`이 나오면 표시된 정확한 경로를 확인한다.

## 5. 새 영상 준비와 사람 선택

영상은 저장소 밖에 있어도 된다. 먼저 원하는 사람이 잘 보이는 초기화 frame과 그
사람 몸통 중앙의 pixel 좌표 `(x, y)`를 찾는다. 영상 왼쪽 위가 `(0, 0)`이다.

예를 들어 초기화 frame 30에서 몸통 중앙이 `(720, 350)`이라면 다음처럼 실행한다.

```bash
bash postclass_full_pipeline/run_video.sh \
  /absolute/path/my_video.mp4 my_motion 30 720 350
```

인수 순서는 다음과 같다.

```text
run_video.sh VIDEO_PATH RESULT_NAME START_FRAME TARGET_X TARGET_Y
```

새 영상에서 좌표를 생략하지 못하도록 일부러 필수 인수로 만들었다. 샘플 `jump`의
좌표가 다른 영상에 실수로 적용되는 것을 막기 위해서다. 기존 파일은 덮어쓰지 않으며
같은 `RESULT_NAME` 폴더가 이미 있으면 실행을 중단한다.

## 6. 시각적으로 확인하기

완료 후 두 영상을 반드시 직접 재생한다.

```text
outputs/postclass/my_motion/prompthmr/overlay.mp4
outputs/postclass/my_motion/gmr/my_motion_g1_reference.mp4
```

overlay에서 확인할 것:

- mesh가 원하는 사람 위에 있는가?
- 가림 뒤에 다른 사람으로 순간이동하지 않는가?
- 앞뒤 방향과 무릎 굽힘이 원본과 맞는가?

G1 reference에서 확인할 것:

- 로봇이 앞뒤가 뒤집히지 않았는가?
- 발이 지면 아래로 관통하거나 계속 떠 있지 않은가?
- 팔과 다리가 관절 한계를 넘는 듯 꺾이지 않는가?

잘못된 reference를 PPO가 자동으로 고쳐 주지는 않는다. 문제가 있으면 학습 전에
사람 선택 frame/좌표나 입력 영상을 바꾸어 다시 생성한다.

## 7. 학습 시작

먼저 20 iteration smoke test로 연결만 검사한다.

```bash
bash postclass_full_pipeline/train.sh smoke my_motion
```

성공하면 256 env, 60,000 iteration 학습을 백그라운드로 시작한다.

```bash
bash postclass_full_pipeline/start_training.sh my_motion
```

터미널은 닫아도 되지만 노트북을 끄거나 재부팅하거나 suspend하면 학습도 멈춘다.
전원과 통풍을 유지한다. 진행 상태와 최신 checkpoint는 다음 명령으로 본다.

```bash
bash postclass_full_pipeline/training_status.sh
```

다음 날 최신 checkpoint로 정책 rollout MP4를 만든다.

```bash
bash postclass_full_pipeline/record_latest_policy.sh my_motion
```

명령 마지막에 출력되는 `outputs/postclass_policy_rollouts/.../policy_rollout.mp4`를
reference 영상과 나란히 비교한다.

기본은 계정이 필요 없는 TensorBoard 로깅이다.

```bash
conda run -n mjlab_postclass_omen5060 tensorboard \
  --logdir repos/unitree_rl_mjlab/logs/rsl_rl/postclass_g1_tracking --port 6006
```

브라우저에서 <http://localhost:6006>을 연다. 완전한 학습은 RTX 5060 노트북에서
수업 한 회보다 훨씬 오래 걸릴 수 있다. 첫 1,000 iteration의 실제 시간을 측정해
남은 시간을 추정한다.

## 8. 결과 위치

```text
outputs/postclass/<name>/prompthmr/results.pkl          PromptHMR raw 결과
outputs/postclass/<name>/prompthmr/overlay.mp4          사람 mesh 확인 영상
outputs/postclass/<name>/gmr/<name>_smplx.npz           GMR 입력 SMPL-X
outputs/postclass/<name>/gmr/<name>_g1_29dof_qpos.npz   GMR 리타깃 qpos
outputs/postclass/<name>/mjlab/<name>_g1_29dof_50hz.npz 학습 reference
repos/unitree_rl_mjlab/logs/rsl_rl/postclass_g1_tracking/ 학습 log/checkpoint
outputs/postclass_policy_rollouts/                        정책 rollout와 MP4
```

PromptHMR 출력은 SMPL-X다. GMR은 이를 사람과 구조가 다른 G1의 관절로 맞추며,
학습 actor의 출력 차원은 29다. 29DoF reference/checkpoint는 23DoF 로봇에 그대로
사용할 수 없다.

## 9. 더 자세한 설명

- 모델 파일과 라이선스: [`../docs/MODELS_AND_LICENSES.md`](../docs/MODELS_AND_LICENSES.md)
- 단계별 입출력: [`../docs/PIPELINE_IO.md`](../docs/PIPELINE_IO.md)
- 고정 버전: [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)

