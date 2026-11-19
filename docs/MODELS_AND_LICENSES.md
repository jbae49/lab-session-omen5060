# 학생용 모델 다운로드 안내

이 문서는 새 컴퓨터에서 시작할 때 직접 받아야 하는 모델을 설명한다. SMPL과
SMPL-X는 일반 Python package가 아니라 사람 mesh를 생성하는 데이터 파일이다.
`pip install smplx`만 실행해서는 모델 파일이 생기지 않는다.

## 1. 어떤 파일이 필요한가

### 이 실습 실행에 필요한 핵심 파일

| 파일 | 사용 위치 | 역할 |
|---|---|---|
| `SMPLX_NEUTRAL.npz` | PromptHMR, GMR | neutral SMPL-X mesh와 joint model |
| `SMPL_NEUTRAL.pkl` | PromptHMR | SMPL-X 결과를 SMPL vertex/joint로 변환할 때 사용 |
| `smplx2smpl.pkl` | PromptHMR | SMPL-X vertex를 SMPL vertex로 변환하는 matrix |
| `smplx2smpl_joints.npy` | PromptHMR | joint 변환 보조 데이터 |

이 README의 exporter는 gender를 `neutral`로 저장하므로 위 neutral 파일이 실제
`jump` 실행의 최소 조건이다. 그러나 새 데이터가 male/female gender metadata를
가질 수 있으므로 공식 archive에 포함된 다음 파일도 함께 준비한다.

```text
SMPLX_MALE.npz
SMPLX_FEMALE.npz
SMPL_MALE.pkl
SMPL_FEMALE.pkl
```

`.npz`와 `.pkl`은 서로 이름만 바꿔 쓸 수 없다. 이 실습에서는 SMPL-X는 NPZ,
SMPL은 PKL 형식을 사용한다.

## 2. 공식 다운로드 링크

- SMPL-X 홈페이지: <https://smpl-x.is.tue.mpg.de/>
- SMPL-X 다운로드: <https://smpl-x.is.tue.mpg.de/download.php>
- SMPL 홈페이지: <https://smpl.is.tue.mpg.de/>
- SMPL 다운로드: <https://smpl.is.tue.mpg.de/download.php>
- PromptHMR 공식 설치 설명: <https://github.com/yufu-wang/PromptHMR#installation>

각 사이트에 직접 가입하고 해당 license에 동의해야 다운로드 메뉴가 열린다.
비공식 Google Drive, Hugging Face 재업로드, 다른 학생이 전달한 파일은 사용하지
않는다. 파일 출처와 버전이 불명확하고 라이선스 조건을 위반할 수 있다.

다운로드 메뉴에서 고를 항목은 다음과 같다.

1. SMPL-X: **SMPL-X v1.1 model archive**
   - 공식 파일명: `models_smplx_v1_1.zip`
   - 필요한 형식: `SMPLX_*.npz`
2. SMPL: **SMPL for Python v1.1.0**
   - 공식 파일명: `SMPL_python_v.1.1.0.zip`
   - 필요한 형식: `basicmodel_*_v1.1.0.pkl`

## 3. 권장 설치: 공식 PromptHMR script

이 저장소의 최상위 폴더에서 다음을 실행한다.

```bash
cd repos/PromptHMR
bash scripts/fetch_smplx.sh
cd ../..
bash scripts/copy_body_models_to_gmr.sh
```

첫 번째 script는 SMPL-X 계정과 SMPL 계정을 차례로 묻는다. 공동 계정을 사용하지
말고 본인이 라이선스에 동의한 계정을 입력한다. 공용 화면을 녹화하거나 공유 중인
terminal에서는 비밀번호를 입력하지 않는다.

두 번째 script는 다운로드된 SMPL-X NPZ 세 개를 GMR 폴더로 실제 복사한다.
symbolic link를 만들지 않으므로 PromptHMR 폴더를 이동해도 GMR 파일이 깨지지 않는다.

## 4. 최종 디렉터리 구조

```text
repos/PromptHMR/data/body_models/
├── smpl/
│   ├── SMPL_NEUTRAL.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_FEMALE.pkl
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
├── smplx2smpl.pkl
└── smplx2smpl_joints.npy

repos/GMR/assets/body_models/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_MALE.npz
└── SMPLX_FEMALE.npz
```

다음 명령에서 `symbolic link`가 출력되면 안 된다.

```bash
find repos/PromptHMR/data/body_models repos/GMR/assets/body_models -type l -print
```

아무것도 출력되지 않으면 정상이다.

## 5. PromptHMR neural-network checkpoint

SMPL-X body model과 PromptHMR checkpoint는 다른 파일이다.

- body model: 관절 회전과 shape로 mesh를 만드는 수학적 모델
- checkpoint: RGB 영상에서 pose/shape를 예측하는 neural-network weight

`scripts/install_all.sh --execute`가 PromptHMR의 world-video 설치를 수행하면서
checkpoint와 detector/SAM2/ViTPose 관련 파일을 다운로드한다. 설치가 중단됐다면
환경을 활성화한 뒤 공식 download script를 다시 실행한다.

자동으로 준비되어야 하는 주요 파일은 다음과 같다.

```text
data/pretrain/phmr/checkpoint.ckpt
data/pretrain/phmr_vid/prhmr_release_002.ckpt
data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt
data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl
data/pretrain/vitpose-h-coco_25.pth
data/pretrain/camcalib_sa_biased_l2.ckpt
data/pretrain/droidcalib.pth
```

```bash
conda activate phmr_omen5060
cd repos/PromptHMR
bash scripts/fetch_data.sh
cd ../..
conda deactivate
```

`fetch_data.sh`가 `data/pretrain`이 이미 존재한다는 이유로 멈추면 기존 다운로드가
부분적으로 남은 상태일 수 있다. 먼저 `doctor.py` 결과를 확인하고, 임의로 파일을
삭제하지 말고 조교에게 확인한다.

## 6. 검증

```bash
conda run -n phmr_omen5060 python scripts/doctor.py --stage prompthmr
conda run -n gmr_omen5060 python scripts/doctor.py --stage gmr
```

모든 항목이 `[OK]`여야 다음 단계로 진행한다. `MISSING`이 하나라도 있으면 오류
메시지에 표시된 정확한 경로에 파일을 놓는다.

## 7. 라이선스 주의

SMPL/SMPL-X 사이트에 표시되는 라이선스가 이 문서보다 우선한다. 모델 archive를
Git에 commit하거나, 공개 cloud folder에 올리거나, 수업 외 제3자에게 재배포하지
않는다. 상업적 사용이 필요하면 별도 commercial license를 확인해야 한다.
