# GitHub → Zenodo 첫 공개 가이드 (v2.3)

이 문서는 GitHub를 처음 사용하는 경우를 기준으로, 현재 폴더를 **GitHub 공개 저장소 → GitHub Release → Zenodo DOI**로 연결하는 최소 절차를 정리한 것입니다.

## 0. 이번 공개에서 권장하는 이름

- GitHub repository name: `nonlinear-diffusion-ui`
- 첫 release tag: `v2.3`
- Release title: `Nonlinear Diffusion Interactive Model / Learning App v2.3`
- Visibility: `Public`
- Zenodo resource type: GitHub 연동 시 software record로 처리

## 1. GitHub 계정 만들기

1. https://github.com/ 에 접속합니다.
2. 계정이 없으면 **Sign up**으로 계정을 만듭니다.
3. 이메일 인증까지 마칩니다.

처음에는 Git 명령어나 GitHub Desktop을 사용하지 않아도 됩니다. 이번 첫 공개는 웹 브라우저만으로도 가능합니다.

## 2. 새 GitHub repository 만들기

1. GitHub 오른쪽 위의 `+` → **New repository**를 선택합니다.
2. Repository name에 `nonlinear-diffusion-ui`를 입력합니다.
3. 간단한 Description 예시:

   `Interactive Jupyter research software for forward and inverse nonlinear diffusion modeling using FDM, FVM, and FEM.`

4. **Public**을 선택합니다.
5. 중요: 이 배포 폴더에는 이미 `README.md`, `.gitignore`, `LICENSE`가 있으므로 GitHub 생성 화면에서 README / .gitignore / license를 새로 만들지 않는 편이 가장 단순합니다.
6. **Create repository**를 누릅니다.

## 3. 이 폴더의 파일을 GitHub에 업로드

새 repository가 비어 있는 화면에서:

1. **uploading an existing file** 또는 **Add file → Upload files**를 선택합니다.
2. 이 배포 폴더의 **내용물 전체**를 업로드합니다. 폴더 자체를 한 겹 더 넣기보다, `README.md`가 repository 최상위(root)에 보이게 하는 것이 좋습니다.
3. Commit message 예시:

   `Initial public release preparation for v2.3`

4. **Commit changes**를 누릅니다.

업로드 후 GitHub 첫 화면에서 README가 자동으로 보이고, 오른쪽에 **Cite this repository**가 나타나면 `CITATION.cff`가 정상 인식된 것입니다.

## 4. Zenodo 계정과 GitHub 연결

GitHub Release를 만들기 **전에** Zenodo에서 repository를 연결해 두는 것이 안전합니다.

1. https://zenodo.org/ 에 로그인합니다.
2. GitHub 계정이 Zenodo에 연결되어 있지 않으면 연결합니다.
3. Zenodo 상단 프로필 메뉴에서 **GitHub**를 엽니다.
4. **Sync now**를 눌러 repository 목록을 새로 읽습니다.
5. `nonlinear-diffusion-ui`를 찾아 연결 toggle을 켭니다.

Zenodo 공식 설명에 따르면, 연결된 repository의 새로운 GitHub Release가 자동으로 수집되어 보존됩니다.

## 5. GitHub에서 첫 Release 만들기

Repository의 GitHub 화면에서:

1. 오른쪽의 **Releases** → **Draft a new release**를 선택합니다.
2. **Choose a tag**에서 새 tag `v2.3`을 만듭니다.
3. Target은 `main` branch로 둡니다.
4. Release title:

   `Nonlinear Diffusion Interactive Model / Learning App v2.3`

5. Release notes 예시:

```text
First archival-ready public baseline.

Highlights:
- 1D/2D/3D forward nonlinear diffusion modeling
- isotropic FDM/FVM/FEM backends
- axis-aligned diagonal-anisotropic FVM
- 1D/2D inverse FVM workflow
- sensitivity / practical-identifiability diagnostics
- reproducibility-oriented Save Run / Load Run records

See README.md and CHANGELOG.md for details.
```

6. 이번 버전을 실제 공개 baseline으로 사용할 것이므로 **pre-release는 체크하지 않아도 됩니다.**
7. **Publish release**를 누릅니다.

## 6. Zenodo에서 DOI 확인

GitHub Release가 공개된 뒤:

1. Zenodo → 프로필 → **GitHub**로 돌아갑니다.
2. 연결된 repository의 `v2.3` release가 처리되었는지 확인합니다.
3. 처리가 끝나면 Zenodo record와 DOI가 생성됩니다.
4. 해당 record에서 title, author, version, license, description을 확인합니다.

이번 repository에는 `CITATION.cff`가 있으므로 Zenodo가 지원하는 software metadata를 읽을 수 있습니다. 특별한 Zenodo-specific metadata가 필요하지 않으므로 `.zenodo.json`은 의도적으로 넣지 않았습니다.

## 7. DOI가 생긴 뒤 할 일

### 가장 중요한 것

Zenodo가 부여한 **v2.3의 version-specific DOI**를 논문, Technical Note, 웹페이지에서 v2.3을 특정해 인용할 때 사용합니다.

### GitHub README에 DOI badge를 추가하고 싶다면

Zenodo record 페이지에서 제공하는 DOI badge Markdown을 복사하여 README 상단에 추가할 수 있습니다. 이 수정은 `v2.3` release 이후의 `main` branch 수정이므로, v2.3 archived ZIP 자체를 바꾸지는 않습니다.

### CITATION.cff의 DOI는 꼭 즉시 넣어야 하나?

아닙니다. 첫 GitHub→Zenodo release에서는 DOI가 release 뒤에 생기므로 현재 파일에는 DOI를 미리 적지 않았습니다. 다음 버전에서 필요에 따라 DOI/URL을 보강할 수 있습니다.

## 8. 다음 버전

예를 들어 기능이 추가되면:

- 작은 기능 개선: `v2.4`
- 구조가 크게 바뀌면: `v3.0`

처럼 새 GitHub Release를 만듭니다. Zenodo는 새 release를 별도 version record로 보존합니다.

## 9. 공개 직전 체크리스트

- [ ] Notebook가 repository root에서 실행되는가?
- [ ] `python tests/smoke_test.py`가 통과하는가?
- [ ] README의 기능 설명이 실제 구현과 일치하는가?
- [ ] 개인 경로, 비밀번호, 토큰, API key가 포함되어 있지 않은가?
- [ ] 저장된 notebook output에 개인 정보가 없는가?
- [ ] `LICENSE`가 의도한 공개 조건과 맞는가?
- [ ] `CITATION.cff`의 저자명과 버전이 맞는가?
- [ ] Zenodo에서 GitHub repository를 **release 전에 enable**했는가?
- [ ] GitHub tag가 `v2.3`으로 정확한가?

## 10. 이번 패키지에서 의도적으로 선택한 것

- License: **MIT**
- Citation metadata: **CITATION.cff only**
- `.zenodo.json`: 사용하지 않음
- Run outputs: 기본적으로 `runs/`를 Git에서 제외
- Notebook outputs: 공개본은 비어 있는 상태 유지
- Numerical backend algorithms: 공개 포장 과정에서 변경하지 않음

MIT는 매우 permissive한 software license입니다. 공개 전에 다른 라이선스(GPL 등)를 원한다면 `LICENSE`, `CITATION.cff`, README의 license 표기를 함께 바꾸어야 합니다.

## 공식 문서

- GitHub 새 repository: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- GitHub releases: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- Zenodo GitHub integration: https://help.zenodo.org/docs/github/
- Zenodo repository enable: https://help.zenodo.org/docs/github/enable-repository/
- Zenodo `CITATION.cff`: https://help.zenodo.org/docs/github/describe-software/citation-file/
- Zenodo GitHub release archive: https://help.zenodo.org/docs/github/archive-software/github-upload/
