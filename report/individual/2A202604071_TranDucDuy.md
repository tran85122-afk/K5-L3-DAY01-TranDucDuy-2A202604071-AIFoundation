# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604071
- **Họ và tên:** Trần Đức Duy
- **SEED:** 4906

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD/1k tin |
|---|---|---|---|---|---|---|
| 1. BoW + LogReg | 1.000 | 0.828 | 0.906 | 0.857 | 0.07 | 0 |
| 2. BoW + MLP | 0.977 | 0.903 | 0.939 | 0.923 | 0.35 | 0 |
| 3a. Encoder + kNN | 0.827 | 0.978 | 0.897 | 1.000 | 130.5 | 0 |
| 3b. Encoder + LogReg | 0.957 | 0.957 | 0.957 | 1.000 | 42.0 | 0 |
| 3c. Encoder + MLP head | 0.935 | 0.935 | 0.935 | 0.974 | 59.4 | 0 |

- Phần 4 (LLM): em không có API key nên không chạy được 4.A, 4.B và stress test LLM. Code 4.1–4.4 chỉ được kiểm tra bằng API giả lập.
- Đường Pareto: BoW + LogReg, BoW + MLP, 3b.
- Đạt SLA p95 < 50 ms: BoW + LogReg, BoW + MLP, 3b.

## 3. Dự đoán và thực tế

1. **emb_search**: Em dự đoán BoW chỉ nhận ra rất ít từ của query. Thực tế 12/13 từ có trong vocab BoW. Dù vậy, 6/6 kết quả gần nhất theo embedding đều là spam (trúng thưởng, kỳ nghỉ), nên phần dự đoán "kết quả gần nhất là spam" là đúng.
2. **emb_tricky**: Em dự đoán Encoder + LogReg đúng nhiều hơn BoW. Thực tế hai bên hoà 5/6, cả hai cùng sai tin "Last chance! Your account rewards expire tonight, confirm details…". Em cho rằng kiểu spam "xác nhận tài khoản" này khác với các mẫu spam model đã học.
3. **tradeoff**: Em dự đoán chỉ có BoW + LogReg nằm trên đường Pareto. Thực tế còn có BoW + MLP (p95 0.35 ms, F1 SUB 0.923) và 3b (p95 42 ms, F1 SUB 1.0). Em đã chỉ nghĩ đến tốc độ mà quên rằng một cách chậm hơn vẫn nằm trên Pareto nếu ít lỗi hơn.

## 4. Phân tích 50 case khó

| Model | F1 SUB | F1 50 case khó | Acc 50 case khó |
|---|---|---|---|
| 1. BoW + LogReg | 0.857 | 0.333 | 0.52 |
| 2. BoW + MLP | 0.923 | 0.368 | 0.52 |
| 3a. Emb + kNN | 1.000 | 0.759 | 0.72 |
| 3b. Emb + LogReg | 1.000 | 0.667 | 0.66 |
| 3c. Emb + MLP head | 0.974 | 0.667 | 0.66 |

- Tụt nhiều nhất: **BoW + MLP** (0.923 → 0.368) và **BoW + LogReg** (0.857 → 0.333).
- 5 case mà tất cả model đều sai: #13, #14, #16, #19 (spam giọng tự nhiên) và #41 (tin OTP hợp lệ).
- **Case #14** "Hi, I'm working from home and making about £300 a day…" (thật là spam): tin không chứa các từ nằm trong top từ đẩy về spam của Phần 1 (call, txt, free, text, now, stop, www, reply, 150p…), lại viết như lời kể cá nhân.
- **Case #41** "Your OTP for login is 482913… call your bank." (thật là ham): tin chứa "call" và "your", hai từ nằm trong top từ đẩy về spam của Phần 1, cộng thêm dãy số. Em cho rằng đây là lý do model nhầm thành spam.
- Tập test thường cùng phân phối với train, nên F1 trên đó lạc quan hơn nhiều so với 50 case khó.

## 5. Thử thách cá nhân

**Đề #5:** Viết 4 tin nhắn của riêng em (2 spam né từ khoá, 2 ham chứa "free"/"win"), so dự đoán BoW với Embedding + LogReg và giải thích từng tin.


## 6. Phương án cho sản phẩm thật

- Yêu cầu: 10 triệu SMS/ngày (~116 tin/s), p95 < 50 ms/tin.
- Đạt độ trễ (đo trên Colab): BoW + LogReg (0.07 ms), BoW + MLP (0.35 ms), 3b (42 ms).
- Không đạt: 3c (59 ms), 3a kNN (130 ms). LLM: em chưa đo được vì không có API key.
- Em chọn **3b** vì F1 test cao nhất (0.957) và F1 trên 50 case khó (0.667) cao gấp khoảng 2 lần BoW.
- Đánh đổi:
  - p95 42 ms sát ngưỡng 50 ms.
  - Throughput cả lô chỉ khoảng 47 tin/s, thấp hơn 116 tin/s, nên cần chạy nhiều tiến trình song song hoặc dùng GPU.
- Chi phí nếu dùng LLM nhỏ cho toàn bộ (tính theo giá trong notebook, 100 token input + 1 token output mỗi tin): 10.000.000 × (100 × $0.10 + 1 × $0.50) / 1.000.000 = $105/ngày, khoảng $3.150/tháng.

## 7. AI log và điều đã học

| Phần | Em hỏi AI gì | AI trả lời / viết gì | Em sửa / kiểm chứng gì |
|---|---|---|---|
| 4 | Code call_openai, prompt, parse_label, llm_classify_batch | Code | Chạy test mock 4.1–4.4 |


**Điều em đã học:** 

- Model học gì, viết và chạy được vòng lặp feature → trọng số → loss → gradient descent.
- Tập test nhỏ thì không đáng tin.
- Điểm test cao chưa có nghĩa là model hiểu.
- Chọn phương án cho sản phẩm thật dựa trên số đo (F1, thời gian train, throughput, latency p50/p95, USD/1k tin) và đường Pareto (1 − F1) ↔ độ trễ, không dựa trên cảm giác.
