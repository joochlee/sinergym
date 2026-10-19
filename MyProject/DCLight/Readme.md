# `./EP_original`
- 원래의 스케줄링 기반의 EP 시뮬레이션 모델링 파일 및 시뮬레이션 결과
<br>

# `./Train`
- RL 학습 코드가 저정된 폴더
- train_rl.py : sinergym 기반 RL 학습용 코드
- train_rl_with_input_pref.py : user preference (energy weight 값) 을 상태에 추가해서 학습한 코드

## `./Train/tb_logs`
- EP 기반 RL 수행중에 기록한 tensorboard 로그 저장 폴더
<br>

# `./Evaluation`
- RL 학습된 모델을 로딩하여 테스트하는 코드가 저장된 폴더
- load_rl_with_input_pref.py : user preference (energy weight 값) 을 학습한 모델을 로딩해서 preference 값을 바꿔가며 테스트하는 코드
- load_rl.py : sinergym 기반 SB3 모델로 학습된 모델을 로딩해서 테스트하는 코드

## `./Evaluation/trained_model`
- RL 학습이 끝난 모델을 저장해둔 폴더 (zip 파일)

## `./Evaluation/tb_logs`
- EP 기반 RL 수행중에 기록한 tensorboard 로그 저장 폴더
<br>
