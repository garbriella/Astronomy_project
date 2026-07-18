# Astronomy_project

## AstroWeather Lab | 천문 이미지와 고층대기 분석기

AstroWeather Lab은 두 개의 독립적인 학습·탐구 영역을 하나의 Streamlit 앱으로 제공합니다.

1. **기본 과제 기능:** FITS 천문 이미지 업로드, 이미지 처리와 서울 기준 천체 위치 계산
2. **심화탐구 기능:** NOAA IGRA 2.2 라디오존데 자료를 이용한 대기 수직구조와 강수 전후 안정도 분석

FITS 자료와 NOAA 자료 사이에 직접적인 인과관계가 있다고 가정하지 않습니다.

## FITS 기본 기능

- `.fits`, `.fit`, `.fts`, `.fz` 파일 업로드
- Astropy를 이용한 전체 HDU 탐색과 이미지 HDU 직접 선택
- 3차원 이상 배열의 2차원 슬라이스 선택
- 이미지 크기, 자료형, 관측 대상, 노출 시간, 관측 시각, 필터와 장비 정보
- 평균·중앙값·표준편차·최솟값·최댓값·유효/결측 픽셀 통계
- 1%–99.5% 기본 명암 범위와 선형·로그·제곱근 변환
- 확대 가능한 이미지, 밝기 히스토그램, 중앙 가로·세로 밝기 프로파일
- FITS 헤더 RA/DEC 또는 수동 좌표를 이용한 서울 기준 현재/관측 당시 고도와 방위각
- 전체 FITS 헤더 표

### 샘플 FITS 파일 사용법

[Google Drive 샘플 FITS 공유 폴더](https://drive.google.com/drive/folders/1PH6OfSRX5SUtImmrFtKbd4P6cl3e92hw)에서 파일을 직접 다운로드한 뒤 앱의 **FITS 이미지 분석** 탭 업로드 창에 올립니다.

FITS 파일은 GitHub 저장소에 올릴 필요가 없습니다. 앱은 업로드된 파일을 현재 실행 세션에서만 처리하며 Google Drive API, 구글 로그인 또는 API 키를 사용하지 않습니다.

## NOAA 심화 기능

기본 관측소는 포항(`KSM00047138`)이며, NOAA 최근자료가 제공되는 다른 대한민국 IGRA 관측소도 선택할 수 있습니다.

- [NOAA IGRA 2.2 관측소 목록](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-station-list.txt)
- [NOAA IGRA 최근 관측자료](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/)
- [NOAA IGRA 2.2 원시자료 형식](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-data-format.txt)

앱은 최근자료 디렉터리에서 `관측소ID-data-beg연도.txt.zip` 파일명을 자동 탐색하고 ZIP을 메모리에서 해제합니다. 시작 연도는 코드에 고정하지 않습니다. 고층관측은 일정한 1시간 간격 자료가 아니므로 UTC와 KST를 병기한 **관측 시각별 자료**로 표현합니다.

주요 기능은 기온·이슬점·상대습도·바람의 수직 분포, Skew-T, 표면 기반 CAPE/CIN, LCL/LFC/EL, 가강수량, 안정도 지수, 관측 시각별 변화와 CSV 다운로드입니다.

- 최근자료가 있는 대한민국 IGRA 관측소 탐색 및 위치 지도
- IGRA 공식 고정폭 파싱, 누락값·QA 제거값·중복 기압 처리
- 관측소·최근자료 디렉터리·ZIP·관측 시각별 계산 결과 캐시
- 수동 또는 CSV 강수 사례 입력과 허용시간 내 전후 sounding 자동 선택
- 현재 브라우저 세션에만 저장되는 댓글과 피드백

심화탐구 주제는 다음과 같습니다.

> 강수 전후 포항 상공의 열역학적 안정도 변화 분석: CAPE 감소율과 CIN 회복 및 수증기량 변화를 중심으로

강수 시작 전과 종료 후 가장 가까운 유효 sounding을 허용시간 안에서 골라 전후 지표와 CAPE 감소율을 비교합니다.

강수 사례 CSV에는 다음 열이 필요합니다.

```text
event_name,start_kst,end_kst,precipitation_mm
사례1,2026-07-15 09:00,2026-07-15 15:00,18.5
```

## 로컬 실행

Python 3.11 이상을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

브라우저에서 Streamlit이 안내하는 로컬 주소(일반적으로 `http://localhost:8501`)를 엽니다.

테스트 실행:

```bash
pytest -q
```

## Streamlit Community Cloud 배포

1. GitHub 저장소를 Streamlit Community Cloud에서 선택합니다.
2. 진입 파일을 `main.py`로 지정합니다.
3. 배포 로그에서 `requirements.txt` 설치와 NOAA HTTPS 연결을 확인합니다.
4. FITS 파일은 저장소에 포함하지 않고 배포된 앱의 업로드 창으로 분석합니다.

개인 이메일, 로그인 정보, API 키 또는 비밀번호는 코드나 설정 파일에 넣지 마세요. NOAA 공개자료에는 인증정보가 필요하지 않습니다.

## 과학적 한계

CAPE가 크다는 사실만으로 강수 발생을 확정할 수 없습니다. CIN, 수증기, 강제상승, 전선, 기단 이동과 일사 변화를 함께 고려해야 합니다. 라디오존데 관측소와 실제 강수 지점 사이의 공간적 차이, 주로 약 12시간인 관측 간격, 결측과 관측오차도 분석을 제한합니다. CAPE 감소율은 간접지표이며 강수와의 인과관계를 증명하지 않습니다.

천체 위치는 FITS 헤더 좌표와 관측 시각의 정확도에 영향을 받습니다. 좌표계 또는 RA/DEC 단위가 헤더에 명확하지 않은 자료는 사용자가 원자료 설명을 확인해야 합니다.

## 파일 구조

```text
.
├── main.py                    # Streamlit 화면, 탭과 상태 관리
├── fits_utils.py              # FITS HDU·이미지·헤더·좌표 처리
├── weather_utils.py           # NOAA 다운로드, IGRA 파싱, 열역학 계산
├── plot_utils.py              # FITS·고층대기·Skew-T 그래프
├── requirements.txt           # 실행 및 테스트 의존성
├── .streamlit/config.toml     # 앱 테마
└── tests/
    ├── test_fits_utils.py
    └── test_igra_parser.py
```

댓글은 `st.session_state`에만 저장되므로 앱 재시작이나 세션 종료 후 사라질 수 있습니다.
