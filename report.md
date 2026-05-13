# Báo Cáo GPU FinOps & Cost Optimization

**Sinh viên:** Vũ Văn Huân  
**MSSV:** 2A202600348  
**Ngày nộp:** 13/05/2026  
**Môn:** Day 25 - GPU FinOps & Cost Optimization Hands-on Lab

---

## 1. Giới thiệu

### 1.1 Mục tiêu của bài lab

Bài lab này nhằm cung cấp trải nghiệm thực hành toàn diện về GPU FinOps — lĩnh vực quản lý tài chính cho hạ tầng GPU trong môi trường cloud. Sinh viên được thực hành với cả mock cluster (mô phỏng qua Docker) và GPU thực tế (Tesla T4), bao gồm:

- Giám sát cluster GPU và theo dõi metrics thời gian thực
- Quản lý workload submission và billing/cost tracking
- Sử dụng Spot Instances để tối ưu chi phí
- Cấu hình và đánh giá cơ chế autoscaling (KEDA-like)
- Phân tích waste report và nhận recommendations tối ưu
- So sánh training FP32 vs Mixed Precision (AMP) trên GPU thực
- Phân tích multi-GPU scaling, dự báo chi phí dự án, và thiết kế chiến lược tối ưu

### 1.2 Tổng quan về GPU FinOps

GPU FinOps là sự kết hợp giữa Financial Operations và quản lý hạ tầng GPU. Trong bối cảnh AI/ML phát triển mạnh, chi phí GPU (đặc biệt là A100, H100) có thể chiếm phần lớn ngân sách dự án. Các nguyên tắc cốt lõi bao gồm:

- **Visibility**: Theo dõi và đo lường chi phí GPU theo thời gian thực
- **Optimization**: Sử dụng spot instances, mixed precision training, scheduling
- **Accountability**: Gán chi phí cho từng workload/project cụ thể
- **Automation**: Autoscaling để tránh lãng phí tài nguyên idle

---

## 2. Phân tích từng phần

### 2.1 Part 1-7: Phân tích kết quả từ Mock Cluster

#### 2.1.1 Part 1 — GPU Cluster Monitoring (Cells 3-4)

Mock cluster được triển khai với 4 nodes và 8 GPUs thuộc 3 loại:

| Node    | GPU 0  | GPU 1  |
|---------|--------|--------|
| node-00 | T4     | T4     |
| node-01 | A100   | A100   |
| node-02 | V100   | V100   |
| node-03 | T4     | T4     |

**Cluster Metrics Summary:**
- Total GPUs: 8, Busy: 0, Idle: 8
- Average Utilization: 7.6%
- Memory Used: 9.2 GB / 288.0 GB (3.2%)
- Total Power Draw: 291W
- Node Count: 4

**Nhận xét:** Ở trạng thái ban đầu, cluster gần như idle (7.6% utilization). Điều này phản ánh tình trạng phổ biến trong thực tế: GPU được provision nhưng không được sử dụng hiệu quả, dẫn đến lãng phí chi phí đáng kể.

#### 2.1.2 Part 2 — Workload Submission & Cost Tracking (Cells 5-6)

Sau khi submit 4 workloads (train-resnet-001, train-bert-002, inference-api-003, train-llm-004):

- Busy GPUs tăng từ 0 lên 5/8
- Utilization tăng từ 7.6% lên 47.1%
- Workload train-llm-004 sử dụng 2 GPUs (node-01 GPU 1, node-02 GPU 0)

**Billing Summary:**
- **Total Cost:** $1.1949
- **Total Savings:** $1.2927
- **Budget Used:** 1.2% / $100.00
- **Alert Status:** OK

| Workload          | Type      | Cost    | Saved    |
|-------------------|-----------|---------|----------|
| train-resnet-001  | ON-DEMAND | $0.0292 | $0.0000  |
| train-bert-002    | ON-DEMAND | $0.6117 | $0.0000  |
| inference-api-003 | SPOT      | $0.0035 | $0.0082  |
| train-llm-004     | SPOT      | $0.5505 | $1.2845  |

**Nhận xét:** Spot instances mang lại savings đáng kể — inference-api-003 tiết kiệm 70%, train-llm-004 tiết kiệm 70%. Hai on-demand workloads không có savings, cho thấy tiềm năng chuyển đổi sang spot.

#### 2.1.3 Part 3 — Spot Instance Management (Cells 7-9)

**Spot Pricing hiện tại:**

| GPU Type | On-Demand | Spot Price | Discount | Availability |
|----------|-----------|------------|----------|--------------|
| T4       | $0.35     | $0.2662    | 24.0%    | LOW          |
| A100     | $3.67     | $2.8676    | 21.9%    | HIGH         |
| V100     | $2.48     | $1.7464    | 29.6%    | MEDIUM       |

Kết quả request spot instances: 3/3 được granted (spot-t4-001, spot-t4-002, spot-a100-001).

**Spot Preemption Simulation:**
- Preempted instances: 0
- Still active: 3
- Spot cost: $0.0004 | On-demand equivalent: $0.0012
- **Total saved: $0.0009 (70.0%)**

**Nhận xét:** V100 có discount cao nhất (29.6%) nhưng availability ở mức MEDIUM. A100 có discount thấp hơn (21.9%) nhưng availability HIGH — phù hợp cho production workloads cần độ ổn định. Spot instances tiết kiệm ~70% so với on-demand, nhưng cần đánh đổi với rủi ro preemption.

#### 2.1.4 Part 4 — Autoscaling (Cells 10-11)

**Autoscaling Policy ban đầu:**
- scale_up_threshold: 80% → điều chỉnh xuống 70%
- scale_down_threshold: 20% → điều chỉnh xuống 25%
- cooldown_seconds: 60
- max_nodes: 8, min_nodes: 1
- preferred_gpu_type: T4
- cost_aware: True

**5 Autoscaling Evaluation Cycles:**
Tất cả 5 cycles đều trả về `NO_ACTION` vì utilization 47.1% nằm trong khoảng [25%-70%].

**Nhận xét:** Cơ chế autoscaling cost-aware là một tính năng quan trọng: nó không chỉ dựa trên utilization mà còn cân nhắc chi phí GPU khi quyết định scale. Việc chọn T4 làm preferred_gpu_type là hợp lý cho cost optimization vì T4 có giá rẻ nhất ($0.35/hr).

#### 2.1.5 Part 5 — Cost Analysis & Optimization (Cells 12-15)

**5 Cost Snapshots:**
- Tất cả snapshots ghi nhận Total=$0.038056, Idle=$0.008833
- **Waste=23.2%** (nhất quán qua 5 snapshots)

**Waste Analysis Report:**
- Average Waste: 23.2%
- Total Idle Cost: $0.044165
- Total Cost: $0.190280
- **Potential Monthly Save: $2,289.51**
- Severity: LOW

**Optimization Recommendations:**
1. 🟡 [MEDIUM] USE_SPOT — Chuyển fault-tolerant workloads sang spot, ước tính tiết kiệm 65%
2. 🟢 [LOW] SCHEDULING — Lên lịch training jobs vào off-peak hours, ước tính tiết kiệm 20%

**Dashboard tổng hợp:**
- Cluster: 8 GPUs / 4 nodes, Utilization 47.1%
- Billing: $1.1949 / $100.00 budget, Alert OK
- Spot: Saved $0.0202 (70.0%)
- Waste: 23.2%, Severity LOW

**Nhận xét:** Waste 23.2% đến từ idle GPUs — 3/8 GPUs không được sử dụng. Nếu scale-down cluster về 3 nodes (6 GPUs) khi không có demand cao, có thể giảm đáng kể idle cost.

#### 2.1.6 Part 6 — Visualization (Cells 16-17)

Đã tạo 2 biểu đồ:
- **finops_cost_breakdown.png**: 3 subplots hiển thị cost breakdown theo workload type, GPU type, và category
- **finops_timeseries.png**: Stackplot cost theo thời gian và waste percentage trend

**Nhận xét:** Visualization giúp trực quan hóa dữ liệu cost — dễ dàng nhận diện pattern lãng phí và workload nào chiếm nhiều chi phí nhất.

#### 2.1.7 Part 7 — Complete FinOps Workflow (Cell 18)

Full 7-step optimization workflow:

| Step | Action | Result |
|------|--------|--------|
| 1 | Initial cluster state | 8 GPUs, Util 47.1%, Idle 3 |
| 2 | Submit heavy workloads | Util tăng lên 70.8%, Busy 8/8 |
| 3 | Autoscaler evaluation | Decision: scale_up (Util 70.8% > threshold 70%) |
| 4 | Cost analysis | Cost/interval $0.04, Waste 4.9% |
| 5 | Recommendations | USE_SPOT (65%), SCHEDULING (20%) |
| 6 | Apply optimization | Switch to spot, Spot savings $0.0393 (70%) |
| 7 | Final billing | Total spend $1.3640, Saved $1.4151, Budget 1.4% |

**Nhận xét:** Sau khi apply full optimization workflow, waste giảm từ 23.2% xuống 4.9%. Spot instances tiết kiệm 70%. Budget utilization vẫn an toàn ở mức 1.4%.

---

### 2.2 Part 8: Phân tích Real GPU Training

#### 2.2.1 GPU Detection & Diagnostics (Cells 19-20)

**Real GPU Detected:**
- Name: Tesla T4
- Memory: 15.6 GB
- CUDA: 12.8
- Pricing: $0.35/hr (on-demand)
- pynvml: available

GPU metrics diagnostic hoạt động tốt, sẵn sàng cho training.

#### 2.2.2 FP32 vs AMP Training Results (Cells 22-24)

| Metric               | FP32        | AMP         | Improvement        |
|----------------------|-------------|-------------|--------------------|
| Total Time           | 129.6s      | 72.0s       | **1.80x faster**   |
| Peak Memory (GB)     | 0.82        | 0.60        | 0.22 GB saved      |
| Cost (USD)           | $0.012599   | $0.006997   | **$0.005602 saved** |
| Cost Saving %        | —           | —           | **44.5%**          |
| Avg GPU Util %       | 91.9        | 72.7        | —                  |
| Avg Power (W)        | 67.5        | 64.8        | —                  |
| Avg Temperature (C)  | 65.9        | 76.8        | —                  |

**Extrapolated Savings at Scale:**
- 1 day training: FP32=$8.40 vs AMP=$4.66 → Save ~$3.74/ngày
- 1 month training: FP32=$252.00 vs AMP=$140.00 → Save ~$112.00/tháng

**Nhận xét:** AMP giảm 44.5% chi phí và tăng tốc 1.8x so với FP32, với accuracy tương đương (FP32: 63.5%, AMP: 66.1%). AMP sử dụng ít GPU utilization hơn (72.7% vs 91.9%) nhưng vẫn cho kết quả tốt hơn do thời gian training ngắn hơn. Đây là kỹ thuật đơn giản nhưng hiệu quả nhất để giảm chi phí GPU training.

#### 2.2.3 Real GPU Cost Reporting (Cell 25)

- FP32 workload: cost $0.012600 (on-demand rate)
- AMP workload: cost $0.002100 (spot rate), saved $0.004900
- Project: real-gpu-lab, Total Cost: $0.014700, Total Savings: $0.004900
- Final Dashboard: Platform Cost $1.3640, Savings $1.4151, Budget 1.4%

#### 2.2.4 GPU Telemetry Visualization (Cell 26)

Đã tạo các biểu đồ:
- **real_gpu_comparison.png**: So sánh FP32 vs AMP
- **cost_per_epoch.png**: Chi phí theo từng epoch
- **real_gpu_telemetry.png**: GPU telemetry trong quá trình training

---

### 2.3 Part 8.5: Advanced GPU Cost Optimization

#### 2.3.1 Multi-GPU Cost Analysis (Cell 27)

Phân tích scaling efficiency với T4 @ $0.35/hr, base time 2 giờ:

| GPUs | Speedup | Efficiency | Time (h) | Total Cost | $/Speedup |
|------|---------|------------|----------|------------|-----------|
| 1    | 1.00x   | 100.0%     | 2.00     | $0.7000    | $0.7000/x |
| 2    | 1.85x   | 92.5%      | 1.08     | $0.7568    | $0.4091/x |
| 4    | 3.30x   | 82.5%      | 0.61     | $0.8485    | $0.2571/x |
| 8    | 5.60x   | 70.0%      | 0.36     | $1.0000    | $0.1786/x |

**Nhận xét:** Khi số GPU tăng, scaling efficiency giảm dần (100% → 70%). Cost per speedup giảm từ $0.70/x xuống $0.18/x, nhưng tổng chi phí tăng. **Best cost** là 1 GPU ($0.70), **best efficiency** là 1 GPU (100%). Điều này cho thấy multi-GPU chỉ có lợi khi thời gian là yếu tố quan trọng (time-to-market), không phải để tiết kiệm chi phí.

#### 2.3.2 Project Cost Forecasting (Cell 28)

| Phase                  | GPU   | Count | Hours | Base Cost | Uncertainty  |
|------------------------|-------|-------|-------|-----------|--------------|
| Data Preparation       | T4    | 1     | 40    | $14.00    | ±$2.10 (15%) |
| Model Training         | A100  | 4     | 120   | $1,761.60 | ±$440.40 (25%)|
| Hyperparameter Tuning  | A100  | 8     | 60    | $1,761.60 | ±$528.48 (30%)|
| Model Evaluation       | T4    | 2     | 20    | $14.00    | ±$1.40 (10%) |

**Project Total:** ~$3,551.20

**Nhận xét:** Model Training và Hyperparameter Tuning chiếm phần lớn chi phí (~99% tổng). Uncertainty cao nhất ở Hyperparameter Tuning (30%) do số lượng thử nghiệm khó dự đoán. Data Preparation và Evaluation chi phí thấp, có thể dùng T4 thay vì A100 để tiết kiệm.

#### 2.3.3 Optimization Opportunity Analysis (Cell 29)

Baseline: $1,468.00 (4x A100, 100 giờ)

| Rank | Strategy                        | Savings | Effort | Risk  | Priority |
|------|---------------------------------|---------|--------|-------|----------|
| 1    | Switch to Mixed Precision (AMP) | 25.0%   | LOW    | LOW   | 🟢 25.0  |
| 2    | Optimize Batch Size             | 15.0%   | LOW    | LOW   | 🟢 15.0  |
| 3    | Use Spot Instances              | 60.0%   | MEDIUM | HIGH  | 🟡 10.0  |
| 4    | Implement Early Stopping        | 20.0%   | MEDIUM | LOW   | 🟡 10.0  |
| 5    | Switch to More Efficient GPU    | 40.0%   | HIGH   | MEDIUM| 🟡 6.7   |

**Nhận xét:** AMP và Batch Size optimization là "low-hanging fruits" — effort thấp, risk thấp, savings tốt. Spot instances tiết kiệm nhất (60%) nhưng risk cao do preemption. Priority score kết hợp savings, effort và risk để xếp hạng các chiến lược một cách khách quan.

#### 2.3.4 Integrated Dashboard (Cell 30)

Dashboard 2x3 tổng hợp tất cả analyses:
- Multi-GPU cost comparison
- Scaling efficiency
- Project cost forecast
- Phase breakdown
- Optimization matrix
- Cumulative savings

#### 2.3.5 Challenge Exercise — LLM Fine-tuning Strategy (Cell 31)

**Scenario:** Fine-tune LLM với 8x A100, 200 giờ, ngân sách $5,000, deadline 2 tuần.

- **Baseline cost:** $5,872.00 — **Vượt ngân sách $872.00!**
- Multi-GPU analysis cho thấy 8x GPU có efficiency chỉ 70%
- Đề xuất chiến lược tối ưu:
  - Giảm xuống 4x GPU (82.5% efficiency), tăng thời gian
  - Sử dụng AMP để giảm 25% thời gian training
  - Sử dụng Spot Instances cho 50% thời gian
  - Early stopping để tránh over-training

**Nhận xét:** Đây là bài toán thực tế mà các ML team thường gặp: ngân sách có hạn nhưng yêu cầu compute lớn. Trade-off giữa số lượng GPU, thời gian, và chi phí cần được cân nhắc kỹ lưỡng.

---

## 3. Kết luận và học hỏi

### 3.1 Những kỹ năng FinOps đã học

1. **Cluster Monitoring**: Sử dụng metrics gateway để theo dõi GPU utilization, memory, power, temperature theo thời gian thực
2. **Cost Tracking**: Gán chi phí cho từng workload cụ thể, phân biệt on-demand vs spot pricing
3. **Spot Instance Management**: Đánh giá rủi ro/lợi ích khi sử dụng spot instances, tính toán savings
4. **Autoscaling**: Cấu hình policy dựa trên utilization threshold, cooldown, và cost-awareness
5. **Waste Analysis**: Xác định idle resources, tính toán potential monthly savings
6. **Real GPU Profiling**: Đo lường utilization, memory, power, temperature trên GPU thực
7. **Training Optimization**: So sánh FP32 vs AMP, đánh giá trade-off speed/cost/accuracy
8. **Multi-GPU Analysis**: Phân tích scaling efficiency và cost per speedup
9. **Cost Forecasting**: Dự báo chi phí dự án với confidence intervals
10. **Strategy Design**: Ưu tiên hóa các chiến lược tối ưu dựa trên savings, effort, risk

### 3.2 Các chiến lược cost optimization hiệu quả

| Chiến lược | Mức tiết kiệm | Độ phức tạp | Phù hợp khi |
|------------|--------------|-------------|-------------|
| Mixed Precision (AMP) | 25-45% | Thấp | Tất cả GPU training |
| Spot Instances | 60-70% | Trung bình | Fault-tolerant workloads |
| Autoscaling | 20-50% | Trung bình | Workloads theo batch |
| Batch Size Optimization | 15-30% | Thấp | Có thể điều chỉnh batch size |
| Early Stopping | 15-30% | Trung bình | Training dài, risk overfitting |
| GPU Type Selection | 20-50% | Cao | Có flexibility về hardware |

### 3.3 Ứng dụng thực tế trong projects

- **Luôn bắt đầu với AMP**: Đây là cách đơn giản nhất để giảm ~40% chi phí training mà không cần thay đổi code nhiều
- **Sử dụng spot instances cho training jobs**: Hầu hết training jobs có thể chịu được interruption nếu có checkpointing
- **Theo dõi waste thường xuyên**: Idle GPUs là nguồn lãng phí lớn nhất — autoscaling giúp giải quyết vấn đề này
- **Lập kế hoạch ngân sách trước**: Dự báo chi phí với uncertainty buffer (20-30%) để tránh vượt ngân sách
- **Multi-GPU không phải lúc nào cũng rẻ hơn**: Scaling efficiency giảm khi tăng số GPU, cần cân nhắc kỹ trade-off
- **Chọn GPU type phù hợp với workload**: Không phải lúc nào A100 cũng cần thiết — T4/V100 có thể đủ cho nhiều tác vụ

---

## Phụ lục: Danh sách file đã tạo

### Screenshots (26 files)

| Part | File | Mô tả |
|------|------|-------|
| 1 | `part1_cluster_monitoring.png` | Cluster nodes view |
| 2 | `part2_workload_submission.png` | Workload submission results |
| 2 | `part2_billing_summary.png` | Billing summary |
| 3 | `part3_spot_pricing.png` | Spot pricing table |
| 3 | `part3_spot_request.png` | Spot instance requests |
| 3 | `part3_spot_preemption.png` | Preemption simulation & savings |
| 4 | `part4_autoscaler_policy.png` | Policy before/after update |
| 4 | `part4_autoscaler_evaluation.png` | 5 evaluation cycles |
| 5 | `part5_cost_snapshots.png` | 5 cost snapshots |
| 5 | `part5_waste_report.png` | Waste analysis report |
| 5 | `part5_recommendations.png` | Optimization recommendations |
| 5 | `part5_dashboard.png` | Full dashboard view |
| 6 | `part6_cost_breakdown_viz.png` | Cost breakdown charts |
| 6 | `part6_timeseries_viz.png` | Time-series tracking |
| 7 | `part7_full_workflow.png` | Full 7-step workflow |
| 8 | `part8_gpu_detection.png` | GPU detection info |
| 8 | `part8_gpu_metrics_diagnostic.png` | GPU diagnostic results |
| 8 | `part8_fp32_summary.png` | FP32 training summary |
| 8 | `part8_amp_summary.png` | AMP training summary |
| 8 | `part8_fp32_vs_amp_comparison.png` | FP32 vs AMP comparison |
| 8 | `part8_real_gpu_cost_report.png` | Real GPU cost report |
| 8.5 | `part85_multi_gpu_analysis.png` | Multi-GPU scaling analysis |
| 8.5 | `part85_project_forecast.png` | Project cost forecast |
| 8.5 | `part85_optimization_analysis.png` | Optimization opportunity analysis |
| 8.5 | `part85_integrated_dashboard.png` | Integrated dashboard |
| 8.5 | `part85_challenge_strategy.png` | Challenge exercise strategy |

### Generated Charts (9 files)

| File | Mô tả |
|------|-------|
| `finops_cost_breakdown.png` | Cost breakdown visualization (3 subplots) |
| `finops_timeseries.png` | Time-series cost tracking |
| `real_gpu_comparison.png` | FP32 vs AMP comparison charts |
| `real_gpu_telemetry.png` | Real GPU telemetry during training |
| `cost_per_epoch.png` | Cost per epoch visualization |
| `multi_gpu_analysis.png` | Multi-GPU scaling analysis chart |
| `project_forecast.png` | Project cost forecast visualization |
| `optimization_analysis.png` | Optimization opportunity chart |
| `advanced_finops_dashboard.png` | Integrated 2x3 dashboard |
