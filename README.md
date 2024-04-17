영수증 인식 OCR 서비스 "RecoWRITE"  
2024.04.04 프로젝트 종료  
  
--- 
개요: 수기 문서의 DB화로 데이터 관리, 활용 효율성 증대

---
소스 데이터: 영수증 63장, 개별 단어 캡쳐 이미지 1500여장, AIHub 다양한 형태의 한글 문자 이미지 인식 데이터 中 필기체 글자, 단어 합 100만여장

---
데이터 분석 요구사항
-  인쇄체와 필기체 구분
-  한국어와 영어, 숫자 구분
-  다양한 필체 구분
-  양식을 벗어난 글자 인식
-  인식된 텍스트를 DB에 저장
-  저장된 데이터 검색, 편집

---
데이터 분석 담당 작업 내용
- 영수증 데이터 파일 전처리
  - 그레이스케일 변환, 대비 증가
- OCR 모델 학습
  - NONE-VGG-BiLSTM-CTC 구조 사용
  - 과적합 문제에 주의
  - 글자 단위 TEST best accuracy 93.51%
  - 수기 영수증 파일 best accuracy 74.99%
- Flask API 설계
  - /image/upload : 업로드한 이미지를 EXTRACT_TEXT_FROM_IMAGE 함수로 전달
  - /image/(filename) : 작업 완료된 이미지를 클라이언트에게 반환
- 텍스트 정제 및 구조화
   - OCR 교정 : 숫자 0,1과 알파벳 o,l를 혼동하는 문제를 해결하기 위한 함수 생성
   - 정규 표현식을 사용해서 규격화된 날짜 형식 추출
   - 환경변수에서 정의한 필터링 규칙을 적용해서 DB화에 불필요한 부분 제거
