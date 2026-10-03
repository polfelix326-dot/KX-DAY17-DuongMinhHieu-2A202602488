# Báo Cáo Phân Tích Hệ Thống Memory & Kết Quả Benchmark

Báo cáo này tổng kết kết quả thực nghiệm và phân tích trade-off giữa hai kiến trúc agent: **Baseline Agent** (chỉ có short-term memory trong thread) và **Advanced Agent** (tích hợp persistent memory `User.md` và compact memory cho ngữ cảnh dài).

---

## 1. Bảng Kết Quả Benchmark

### 1.1. Standard Benchmark (`data/conversations.json` - 10 hội thoại, đa phiên)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 2,246 | 19,884 | **0.0%** | 15.0% | 0 | 0 |
| **Advanced Agent** | 4,258 | 35,277 | **100.0%** | **100.0%** | 435 | 0 |

### 1.2. Long-Context Stress Benchmark (`data/advanced_long_context.json` - Chuỗi dài ép nén)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 374 | 23,500 | **0.0%** | 15.0% | 0 | 0 |
| **Advanced Agent** | 1,168 | **12,747** | **100.0%** | **100.0%** | 344 | **4** |

---

## 2. Phân Tích Chuyên Sâu (Bước 8 trong Guide.md)

### 2.1. Vì sao Advanced Agent có Recall vượt trội so với Baseline Agent?
- **Baseline Agent** hoàn toàn bị giới hạn bởi phạm vi của session (`thread_id`). Khi sang thread mới (mô phỏng người dùng quay lại sau vài giờ hoặc vài ngày), baseline không có bất kỳ trạng thái nào và không thể nhớ bất kỳ fact nào từ phiên trước (Recall đạt 0.0%).
- **Advanced Agent** duy trì persistent profile (`User.md`) độc lập với vòng đời của các thread. Các thông tin ổn định như tên người dùng (`DũngCT`), nơi ở hiện tại (`Huế`/`Đà Nẵng`), nghề nghiệp (`MLOps engineer`), đồ uống (`cà phê sữa đá`), thú cưng (`corgi tên Bơ`) được lưu trữ bền vững trên đĩa và tự động nạp vào prompt context ở mọi phiên làm việc mới, mang lại **100% Cross-session recall**.

### 2.2. Vì sao Advanced Agent có thể tốn chi phí hơn ở hội thoại ngắn?
- Trong các hội thoại ngắn (như Standard Benchmark), mỗi lượt chat của Advanced Agent phải mang theo phần ngữ cảnh của file `User.md`.
- Với 10 hội thoại và các câu hỏi recall, tổng `Prompt tokens processed` của Advanced Agent (35,277) cao hơn Baseline (19,884).
- **Trade-off cốt lõi**: Để có khả năng nhớ dài hạn và cá nhân hóa trải nghiệm qua nhiều phiên, hệ thống phải chấp nhận một chi phí cố định (overhead) để duy trì và nạp profile vào prompt context.

### 2.3. Vì sao Compact Memory giúp Advanced Agent có lợi thế áp đảo ở hội thoại dài?
- Trong hội thoại rất dài (Stress Benchmark với 16 lượt dày đặc tin tức NASA, WMO, v.v.), Baseline Agent kéo theo toàn bộ lịch sử thô qua từng lượt. Chi phí ngữ cảnh tăng theo cấp số cộng của độ dài hội thoại ($O(N^2)$ prompt token load), khiến Baseline tiêu tốn tới **23,500 prompt tokens**.
- **Compact Memory** của Advanced Agent tự động kích hoạt khi token load vượt ngưỡng (`compact_threshold_tokens`). Nó chuyển các lượt trao đổi cũ thành một bản tóm tắt súc tích có giới hạn và chỉ giữ lại $K$ tin nhắn gần nhất (`compact_keep_messages`).
- Nhờ vậy, `Prompt tokens processed` của Advanced Agent chỉ còn **12,747 tokens** (giảm tới **45.7%** so với Baseline), dù vẫn duy trì 100% recall và chất lượng câu trả lời cao nhất.

### 2.4. Tốc độ tăng trưởng của Memory File và các rủi ro đi kèm
- Trong bài test, `User.md` tăng từ 0 lên khoảng **344 - 435 bytes** (chứa khoảng 8-10 facts chính).
- **Rủi ro thực tế trong production**:
  1. **Profile Bloat (phình to không kiểm soát)**: Nếu mọi chi tiết vụn vặt đều được trích xuất và nhét vào `User.md`, file sẽ phình to thành hàng chục KB, làm tăng vọt chi phí prompt overhead ở mọi lượt chat tương lai.
  2. **Stale/Outdated Facts**: Người dùng thay đổi thông tin (đổi nghề, đổi nơi ở) nhưng hệ thống vẫn giữ fact cũ hoặc lưu song song hai fact mâu thuẫn.
  3. **Noise Injection**: Agent nhầm lẫn giữa thông tin đùa giỡn, chuyến công tác tạm thời với thông tin cá nhân cốt lõi.

---

## 3. Các Tính Năng Bonus Đã Triển Khai (Mục tiêu 90-100 điểm)

1. **Confidence Threshold & Question Filtering**:
   - Agent nhận diện các lượt hỏi thăm/kiểm tra trí nhớ (inquiry questions) như `"Hiện tại mình đang ở đâu?"` hay `"Bạn có biết DũngCT không?"` để **không** trích xuất sai fact vào profile.
2. **Conflict Handling & Dynamic Fact Replacement**:
   - Tự động phát hiện và xử lý đính chính: khi người dùng chuyển từ `backend engineer` sang `MLOps engineer`, hoặc chuyển nơi ở từ `Đà Nẵng` sang `Huế` rồi sang `Đà Nẵng`, profile lập tức cập nhật giá trị mới nhất thay vì giữ song song cả hai.
3. **Noise Filtering**:
   - Lọc bỏ nhiễu cố tình như câu đùa chuyển sang `product manager` hoặc chuyến công tác 2 ngày tại `Hà Nội`.
   - Ngăn chặn nhầm lẫn tên người dùng với tên thú cưng (`corgi tên Bơ`) hay thực thể tin tức (`phi hành gia`).
4. **Structured Entity Extraction & Cumulative Merging**:
   - Tổ chức `User.md` thành các mục rõ ràng (Thông tin cá nhân, Nghề nghiệp, Sở thích & Phong cách).
   - Hỗ trợ gộp mối quan tâm kỹ thuật (`Python`, `MLOps`, `AI`) tích lũy qua thời gian.
