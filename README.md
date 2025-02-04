# 수기 영수증 텍스트 데이터 추출 분석

## 프로젝트 개요

### 1.1 프로젝트 추진 배경
- 수기 영수증 처리로 인해 대량 데이터 관리에 어려움이 있으며, 이를 디지털 텍스트로 변환하여 업무 효율성을 높이고자 함.

### 1.2 수요 기업 소개
- **나인온스(9OZ)**: 온/오프라인 의류 패션 매장을 운영하며, 인쇄 및 수기 영수증 처리 문제 해결이 필요함.

### 1.3 필요성
- 대량의 수기 영수증을 디지털화하여 데이터 관리 및 활용의 효율성 향상 필요.

### 1.4 목적
- 문자 인식 기술을 활용하여 영수증 데이터를 추출 및 DB에 저장.
- 거래 업체 분석 및 검색 업무의 효율성 증대.

---

## 프로젝트 문제 설명

### 2.1 프로젝트 문제 기술
1. **영수증 처리 방식**: 도매 거래 시, 수기 및 인쇄 영수증 혼재.
2. **영수증 분실 문제**: 거래 내역 확인 불가로 서비스 문제 발생.
3. **데이터화 제약**: 거래 내역 통합 분석 어려움.

### 2.2 데이터셋 현황
1. **나인온스 & 부산 C&S 제공 영수증 63장**:
   - 거래 업체 정보 (날짜, 연락처, 계좌번호 등).
   - 품목, 단가, 수량, 금액, 총액 정보.
2. **편집 이미지 1,548장**: 영수증 개별 단어 분리.
3. **AIHub 데이터**:
   - 필기체 글자 344,412건.
   - 필기체 단어 756,813건.
![image](https://github.com/user-attachments/assets/7b8d7a13-c85d-4a8c-a9a8-944d910edf7f)

---

## 시스템 설계 내용

### 3.1 서비스 요구 사항
- 수기 영수증에서 `-` 항목을 `,000`으로 인식.
<img src="https://github.com/user-attachments/assets/365b9d1e-39df-404f-9ac3-0c8c9d0b4466" width="270">
<img src="https://github.com/user-attachments/assets/e27f9cd9-d191-4cc6-b2d0-d33990c65f7d" width="270">
<img src="https://github.com/user-attachments/assets/453f0378-bf98-4eda-9b5f-4dd2adf02829" width="270">

### 3.2 데이터 분석 요구 사항
1. 인쇄체와 필기체 구분.
2. 한국어, 영어, 숫자 구분.
3. 다양한 필체 인식.
4. 양식을 벗어난 글자 인식.
5. **OCR 및 YOLO** 활용: 필기체 텍스트 인식.
6. 인식된 텍스트 DB 저장.
7. DB 저장 데이터 검색 및 분석 제공
---
## 팀 구성원별 작업 내용
- **FE**: 문형호
- **BE**: 이지연
- **DA**: 허선행
![담당내용](https://github.com/user-attachments/assets/c69c615b-923b-4834-ae7f-d96571efa84d)

#### 영수증 업로드
<p align="center">
  <img src="https://github.com/user-attachments/assets/c8513bda-3f8a-4627-a19d-0bc04205185b" alt="2 영수증 업로드">
</p>

#### 내용 수정 및 저장
<p align="center">
  <img src="https://github.com/user-attachments/assets/7691e743-b0d8-4bcd-b054-af55e1c537c1" alt="3 인식내용 수정 후 저장">
</p>

#### 엑셀파일 저장
<p align="center">
  <img src="https://github.com/user-attachments/assets/973ed37a-34d9-43ab-95aa-e7f7bd66633f" alt="7 엑셀 파일로 저장">
</p>


## 추진 일정 경과
- **2024-02-28 ~ 2024-04-04**
![세부 일정](https://github.com/user-attachments/assets/093dba05-0191-42a4-8caa-b9ad1b6d43ee)
---

## 아쉬운 점
1. 회원 가입 시 인증 방식 미흡.
2. 영수증 업로드 및 수정 시 회원 간 알림 기능 부재.

---

## 개선 사항

### FE
1. OAuth 로그인 기능 이해 부족.
2. 반응형 페이지 미완성.
3. 컴포넌트 재사용성 미흡.

### BE
1. OAuth 로그인 연결 부족.
2. N+1 문제 미해결.
3. 테스트 코드 미완성.

### DA
1. 손글씨 모델 완성도 부족.
2. 다양한 모델 시도.
3. 학습 데이터 부족.
4. 학습 미흡.

---

## 참고 자료
- 프로젝트 관련 참고 자료 목록 (작성 시 추가).
![참고자료](https://github.com/user-attachments/assets/ed598933-1e2e-46bd-af94-f453dc549740)
