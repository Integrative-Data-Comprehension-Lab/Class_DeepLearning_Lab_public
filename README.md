# 딥러닝응용 수업 실습자료

## 실험 준비

1. cloning public lab materials
```bash
cd ~/
git clone https://github.com/Integrative-Data-Comprehension-Lab/Class_DeepLearning_Lab_public.git
```

2. cloning your private repository
```bash
cd ~/
git clone https://YOUR_USERNAME:YOUR_TOKEN@github.com/\
YOUR_USERNAME/YOUR_PRIVATE_REPOSITORY_NAME.git
```

3. 실험 자료를 개인 레포지토리로 가져오기
```bash
cd ~/
cp Class_DeepLearning_Lab_public/README.md YOUR_PRIVATE_REPOSITORY_NAME/
cp -r Class_DeepLearning_Lab_public/lab_XX YOUR_PRIVATE_REPOSITORY_NAME/
```

4. 실험 진행 전 상태를 푸쉬하기
```bash
cd ~/YOUR_PRIVATE_REPOSITORY_NAME
git status
git add lab_XX
git commit -m "before lab_XX"
git push
```

## 과제 제출 방법
 - <mark>(주의)</mark> 폴더 구조, 파일 이름, 함수 이름, 또는 함수 인자를 변경할 경우 테스트 모듈이 정상 작동하지 않으니 주의할 것.
 - <mark>(주의)</mark> jupyter notebook에 테스트를 위해 기존에 없던 새로운 셀을 추가했다면 반드시 삭제할것. (코드 테스트가 너무 오래걸리거나 실패할 수 있음)
 
1. .ipynb파일은 .py 파일로 변환한다.
``` bash
cd ~/YOUR_PRIVATE_REPOSITORY_NAME
cd lab_XX
jupyter nbconvert FILE_NAME.ipynb --to script \
--TagRemovePreprocessor.enabled=True \
--TagRemovePreprocessor.remove_cell_tags execute_cell
```

2. 숙제 제출 전 코드를 최종 테스트 한다.
- 테스트가 실패하거나 시간이 너무 오래걸릴경우 코드에 문제가 있는지 다시 검토한다. (테스트는 1분 이내로 완료됨.)
```bash
cd ~/YOUR_PRIVATE_REPOSITORY_NAME
cd lab_XX
pytest
```

3. 깃 커밋 & 푸쉬 한다
- 과제와 관련된 파일을 모두 제출할 것 (checkpoint나 data 등 용량이 큰 파일들은 별도의 지침이 없는 한 푸쉬하지 마세요.). 
- 예: `.py`, `.ipynb`, `.sh`, ...

```bash
cd ~/YOUR_PRIVATE_REPOSITORY_NAME
git add lab_XX
git commit -m "finished lab_XX"
git push
```