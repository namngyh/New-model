# Tóm tắt điều hành

Mô hình ước lượng mức VN-Index, khả năng tăng/giảm, trạng thái thị trường và khoảng rủi ro cho 20 phiên sau 2026-07-13. Đây là phân phối kịch bản có điều kiện, không phải dự đoán chắc chắn hay khuyến nghị đầu tư.

Median terminal là **1826.41**, tương ứng median return **1.44%**. Xác suất tăng là **60.34%**, giảm là **39.66%**. VaR 95% là **-7.59%**, ES 95% là **-9.97%**, và xác suất maximum drawdown vượt 5% là **35.03%**.

So với baseline: chưa có đủ bằng chứng để kết luận mô hình chính tốt hơn baseline mạnh nhất; ở h=20, random_walk_drift có rmse 0.057330. Một mô hình có RMSE tốt hơn vẫn có thể nhận diện Bear/Stress kém hoặc tạo interval quá rộng. Kết quả cần được cập nhật khi có dữ liệu mới và không nên dùng đơn lẻ để quyết định giao dịch.

Run experimental đổi RMSE h=20 từ **0.067470** xuống **0.057330**, coverage 95% từ **86.29%** lên **95.26%**, width từ **0.2088** lên **0.2944**, và VaR exceedance từ **13.45%** xuống **4.74%**. Chỉ **7/9** acceptance checks đạt, vì vậy `configs/default.yaml` vẫn giữ A0; kết quả mới nằm ở `configs/experimental.yaml`.
