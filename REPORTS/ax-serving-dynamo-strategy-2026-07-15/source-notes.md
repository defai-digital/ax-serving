# AX Serving 與 Dynamo 策略：來源與方法

## 決策框架

- 問題：如何用 upstream Dynamo 與既有 AX Serving 建立新版 AX Serving，管理 NVIDIA Thor、CUDA GPU PC 與 Apple Silicon Mac？
- 主要讀者：AX Serving 產品與技術決策者。
- 時間點：2026-07-15。
- 比較基準：直接使用 vLLM／TensorRT-LLM、vLLM Production Stack，以及 NVIDIA Dynamo。
- 決策結果：AX Serving 管理 execution domains；NVIDIA Thor 與 CUDA GPU PC 使用 Dynamo，但分成明確 hardware pools／deployment classes；Mac 使用 AX Engine。採 upstream service／adapter composition，不追求 Dynamo 的 CUDA 內部功能 parity，也不建立長期 private fork。

## 權威來源與可建立的事實

1. `.internal/prd/PRD-AX-SERVING.md`
   - AX Serving 的 canonical 邊界是 runtime-neutral control plane。
   - Runtime 負責 tokenization、batching、KV、distributed execution 與 kernels。
   - AX Serving 負責 fleet state、admission、endpoint selection、safe failover、安全邊界與 operations。
   - Production goodput loss gate 是低於 3%，並要求 live mixed-fleet evidence。

2. `.internal/IMPLEMENTATION-STATUS.md`
   - Runtime-neutral architecture、request profiling、hard eligibility、inference-aware scoring、async deployment desired state 與 rollback 已在 source 實作或由本地測試覆蓋。
   - Live AX Engine＋CUDA certification、production benchmark、兩 gateway HA、Redis fault test 與 60-minute soak 仍未完成。
   - 因此目前能證明工程基礎，不能證明 production value 或市場需求。

3. `docs/market-positioning.md`、`docs/icp-and-demand.md`、`docs/competitive-landscape.md`
   - Repo 自己已排除單 endpoint、單 runtime 與 CUDA token scheduling 作為 AX Serving 的主要適配場景。
   - Demand 必須由實際 evaluation 與 retained evidence 驗證，不能從硬體品牌或模型大小推論。

4. `.internal/prd/PRD-AGENT-AWARE-INFERENCE-FABRIC.md`
   - Agent session affinity、capability negotiation 與 private fleet placement 是已定義的鄰接方向。
   - Agent planning、tool execution、memory 與 KV ownership 明確不屬於 AX Serving。
   - 其 release gate 包含 sequential-turn p95 time-to-next-action 至少改善 25%，可作為 adaptive control plane 的 outcome gate。

5. vLLM 官方文件
   - vLLM 提供 plugin system，並透過 Production Stack 擴展到 Kubernetes deployment、routing、observability、autoscaling 與 KV cache offloading。
   - 這表示通用 serving feature 應假設會快速商品化；AX 不應把單一 router feature 當作長期護城河。

6. NVIDIA Dynamo 官方文件（v1.2.1）
   - Dynamo 已定位為 vLLM、SGLang、TensorRT-LLM 上方的 engine-agnostic distributed inference layer。
   - 它涵蓋 KV-aware routing、disaggregated serving、KVBM、NIXL、Planner、Kubernetes operator、Grove、ModelExpress、observability 與 lifecycle。
   - Kubernetes production path 使用 Kubernetes-native discovery；非 Kubernetes distributed mode 可選 etcd／NATS。進階部署仍可能涉及 Prometheus、shared storage、RDMA、Grove 與 KAI Scheduler。
   - Custom unified backend contract 仍為 beta，部分 backend 功能有已知缺口。

7. `https://github.com/ai-dynamo/dynamo`
   - Upstream repo 明確把 Dynamo 定位為 inference engines 上方的 datacenter-scale orchestration layer。
   - 主要 codebase 採 Apache License 2.0；依賴或修改時需保留適用的 license、notice 與 modified-file obligations。這不是法律意見。
   - Repo 更新速度高，適合 pinned release／container 與 adapter integration，不適合建立需要長期同步的大型 private fork。

8. NVIDIA Dynamo support matrix 與 Jetson AGX Thor 官方文件
   - Dynamo 官方矩陣支援 ARM64、Ubuntu 24.04 與 Blackwell architecture，並提供 multi-arch images。
   - Jetson AGX Thor 可執行 ARM64 CUDA 13 container；但本次未找到 Dynamo 官方明確列出 Jetson AGX Thor 的 certified recipe。
   - 因此 Thor compatibility 是合理 hypothesis，不是已證明支援；必須以 JetPack／driver／container／backend 的 pinned live certification 決定。

## 推論與不確定性

- 「AX 可以在易用性、非 Kubernetes private deployment、Mac／edge、semantic safety 與 enterprise policy 上勝出」是策略推論，尚無客戶或 benchmark 證明。
- 「Dynamo 將持續擴大跨硬體支援」由 NVIDIA 已公開的 Intel／AMD 擴展方向支持，但不能視為 Apple Silicon roadmap 承諾。
- Thor 與 GPU PC 不應預設屬於同一 homogeneous model pool，也不應跨兩者做 tensor parallel；CPU architecture、driver、memory topology、engine artifact 與性能 envelope 應分別認證。
- 本次未取得 customer interviews、usage telemetry、win/loss、pricing、support burden 或 willingness-to-pay 資料。所有市場結論均為 provisional。
- 報告使用一張 ownership decision map。其 0–3 數值是明確定義的策略分類（委派、整合、學習、核心自有），不是市場、效能或成熟度量測；市場規模、需求強度與經濟效益仍不繪圖，避免製造虛假精確度。

## Chart map

- Section：`可以學架構，但不應複製整個產品範圍`
- Analytical question：哪些能力應由 AX 自有，哪些應交給 execution runtime 或 Dynamo？
- Takeaway：AX 應自有跨 runtime identity／policy／audit，委派 token execution 與 CUDA cluster internals。
- Family／type：category comparison／horizontal bar。
- Fields：`chart_label`、`layer`、`ownership_level`、`ownership_label`、`reason`；圖上使用短標籤，完整能力名稱保留在表格與 tooltip。
- Rubric：0=delegate、1=integrate、2=learn and standardize、3=core AX ownership。
- Palette：single-root；不使用第二 categorical color encoding。
- Delivery：canonical report artifact chart，HTML portable builder。

## Executive report 結構對應

- Title：`AX Serving 的差異化控制面策略`
- Executive Summary：直接回答是否應做「better Dynamo」。
- Key findings：學習／委派／自有 ownership map、產品 wedge 與目標架構。
- Recommended next steps：四階段 evidence-gated roadmap。
- Further questions：ICP、budget owner、替代方案與經濟效益。
- Caveats and assumptions：缺少市場與 production 證據、競品快速變動。

## Delivery blocker

- Selected delivery mode：portable HTML；audience：product stakeholders。
- Canonical `artifact.json` 通過 JSON 與 package validation，但 shared portable verifier 在 1440px 回報 `horizontal_overflow`，因此沒有發布未通過驗證的 HTML。
- 同一 builder 對既有 `REPORTS/ax-serving-high-value-strategy-2026-07-15/artifact.json` 控制組產生相同的約 scrollbar-width overflow，表示這是目前 shared reader／Chromium verification regression，而不是本報告的表格或 chart 特有問題。
- 已完成的 canonical report input 保留在同目錄 `artifact.json`；待 shared builder 修復後應重新執行 `report:deliver`，而不是手寫另一套 HTML renderer。
