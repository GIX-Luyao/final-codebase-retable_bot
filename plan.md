# LLM Planner 功能开发计划

> **状态**：设计已确认，准备实现  
> **分支**：`llm-planner`（基于 `integreation_two_models`）

## 1. 目标概述

在现有的 Pipeline 流程中引入 **LLM 视觉规划器（Vision Planner）**，让系统能够：
1. 拍摄 front 摄像头当前画面，发送给 Gemini 模型
2. 模型判断桌面上 **Lemon**、**Tissue**、**Cup**、**Cloth** 四个物体的整理完成状态
3. 根据判断结果，**只执行未完成的 stage**，跳过已完成的
4. 前端实时显示每个物体的完成/未完成状态
5. 每次按 **Start** 或 **Restart** 时，先调用 LLM 规划，再执行

---

## 2. 已确认的设计决策

| # | 问题 | 决策 |
|---|------|------|
| 1 | Cloth 是否参与 LLM 规划 | ✅ 参与。规则：抹布在图片右下角 → todo，否则 → done |
| 2 | LLM 触发时机 | ✅ **Start + Restart** 都触发 |
| 3 | Stage 过滤方案 | ✅ **方案 A**：通过控制文件 IPC 传递 `PLAN:stage1,stage2` |
| 4 | LLM 调用失败 | ✅ 直接报错，不 fallback |
| 5 | 完成后验证 | ✅ 后续可加，本次不做 |
| 6 | 执行顺序 | ✅ 未完成的按原始 PIPELINE_STAGES 顺序执行 |

---

## 3. 系统架构

```
                          ┌─────────────────┐
                          │   OpenRouter     │
                          │  Gemini 3.1      │
                          │  Flash Lite      │
                          └────────▲─────────┘
                                   │ HTTP API (vision)
                          ┌────────┴─────────┐
                          │  llm_planner.py  │
                          │  (Python module) │
                          └────────▲─────────┘
                                   │
        ┌──────────────────────────┼──────────────────────┐
        │                          │                       │
  ┌─────┴──────┐          ┌───────┴────────┐     ┌───────┴────────┐
  │ main_robot │          │ eval_pipeline  │     │   Frontend     │
  │  (FastAPI) │──IPC────►│   (.py)        │     │   (React)      │
  │            │◄──WS────►│                │     │                │
  └────────────┘          └────────────────┘     └────────────────┘
       │                                                │
       └────────────── WebSocket ───────────────────────┘
```

### 核心流程（Start / Restart）

```
用户按 Start 或 Restart
    │
    ▼
[1] Restart 时：发送 HOME，等待到达 Home
    │
    ▼
[2] 后端读取 /tmp/lerobot_frames/front.jpg
    │
    ▼
[3] 调用 OpenRouter Gemini API（llm_planner.py）
    │
    ▼
[4] 解析返回 JSON：{ "Lemon": "done", "Tissue": "todo", "Cup": "todo", "Cloth": "done" }
    │
    ▼
[5] 广播规划结果到前端（WebSocket）
    │
    ▼
[6] 过滤出 todo stages → ["Tissue", "Cup"]
    │
    ▼
[7] 写入控制文件：PLAN:Tissue,Cup  然后写入 START
    │
    ▼
[8] eval_pipeline 读取 PLAN 命令，只执行 Tissue 和 Cup
    │
    ▼
[9] Pipeline 完成
```

---

## 4. 技术方案

### 4.1 新增文件：`backend/llm_planner.py`

```python
# 核心接口
async def plan_from_camera(frame_path: str) -> PlanResult:
    """
    读取 front 摄像头帧，调用 LLM 判断物体状态。
    
    Returns:
        PlanResult {
            objects: {
                "Lemon":  {"status": "done"|"todo", "reason": "..."},
                "Tissue": {"status": "done"|"todo", "reason": "..."},
                "Cup":    {"status": "done"|"todo", "reason": "..."},
                "Cloth":  {"status": "done"|"todo", "reason": "..."},
            },
            stages_to_run: ["Tissue", "Cup"],  # 只包含 todo 的，按原始顺序
            raw_response: "...",
        }
    Raises:
        LLMPlannerError: API 调用失败或返回格式错误时抛出（不 fallback）
    """
```

**OpenRouter API 配置：**
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Model: `google/gemini-3.1-flash-lite-preview`
- API Key: `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`

**Prompt 设计：**
```
你是一个桌面整理机器人的视觉规划器。请观察这张桌面照片，判断以下物体的整理状态：

1. Lemon（柠檬）— 是否已被放入指定盒子中。如果柠檬仍在桌面上，状态为 todo；如果已不在桌面上或已在盒子中，状态为 done。
2. Tissue（纸巾盒）— 是否已被放到指定位置。如果纸巾盒仍在桌面原位，状态为 todo；如果已被移走，状态为 done。
3. Cup（水杯）— 是否已被放入指定盒子中。如果水杯仍在桌面上，状态为 todo；如果已不在桌面上或已在盒子中，状态为 done。
4. Cloth（抹布）— 观察图片右下角区域。如果抹布在图片右下角，状态为 todo（需要擦桌）；如果抹布不在右下角，状态为 done（不需要擦桌）。

请严格按照以下 JSON 格式回复，不要包含其他文字：
{
  "Lemon": {"status": "done" 或 "todo", "reason": "简短原因"},
  "Tissue": {"status": "done" 或 "todo", "reason": "简短原因"},
  "Cup": {"status": "done" 或 "todo", "reason": "简短原因"},
  "Cloth": {"status": "done" 或 "todo", "reason": "简短原因"}
}
```

### 4.2 修改：`backend/config.py`

```python
# ── LLM Planner ──
LLM_PLANNER_ENABLED = True
LLM_API_BASE = "https://openrouter.ai/api/v1"
LLM_API_KEY = "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"
LLM_MODEL = "google/gemini-3.1-flash-lite-preview"
LLM_PLANNABLE_OBJECTS = ["Lemon", "Tissue", "Cup", "Cloth"]
```

### 4.3 修改：`backend/main_robot.py`

- `RobotState` 新增 `llm_plan`, `llm_planning`, `llm_plan_error` 字段
- `to_dict()` 广播 LLM 状态到前端
- `/api/start` → 先 LLM 规划 → 写 `PLAN:stages` → 写 `START`
- `/api/restart` → 写 `HOME` → 等 Home 到达 → LLM 规划 → 写 `PLAN:stages` → 写 `RESTART`（或 START）
- 新增 `/api/plan` GET 端点（获取当前规划结果）
- 新增 `/api/replan` POST 端点（手动触发重新规划）

### 4.4 修改：`eval_pipeline.py`

在 `check_control_file()` / `read_control_command()` 中支持新命令：

```
PLAN:Lemon,Tissue,Cup    → 设置当前 pipeline 只执行这些 stage
```

在主循环的 stage 遍历中，检查当前 stage 是否在 plan 列表中，不在则跳过。

### 4.5 修改：前端 `App.tsx`

新增 **LLM Plan 状态面板**，显示在 Pipeline 进度条上方：

```
┌─────────────────────────────────────────────┐
│  🧠 VISION PLANNER                         │
│                                              │
│  🍋 Lemon    ✅ Done   "已放入盒中"         │
│  🧻 Tissue   ⏳ Todo   "仍在桌面上"         │
│  🥤 Cup      ⏳ Todo   "仍在桌面上"         │
│  🧹 Cloth    ✅ Done   "抹布不在右下角"     │
│                                              │
│  [规划中... ⏳] (调用 LLM 时显示)           │
└─────────────────────────────────────────────┘
```

WebSocket 新增字段：
```typescript
llm_plan?: Record<string, { status: string; reason?: string }>
llm_planning?: boolean
llm_plan_error?: string
```

---

## 5. 实现步骤

### Phase 1：LLM Planner 核心模块
- [ ] 创建 `backend/llm_planner.py`
- [ ] 在 `config.py` 中添加 LLM 配置

### Phase 2：后端集成
- [ ] `main_robot.py` — RobotState 新增 LLM 字段 + WebSocket 广播
- [ ] `main_robot.py` — 新增 `/api/plan`, `/api/replan` 端点
- [ ] `main_robot.py` — 修改 `/api/start` 和 `/api/restart` 流程

### Phase 3：eval_pipeline IPC
- [ ] `eval_pipeline.py` — 支持 `PLAN:stage1,stage2` 控制命令
- [ ] 确保 stage 过滤后按原始顺序执行

### Phase 4：前端展示
- [ ] `App.tsx` — LLM Plan 状态面板组件
- [ ] `App.tsx` — WebSocket 处理 `llm_plan` 字段
- [ ] 构建前端 dist

---

## 6. 风险和注意事项

1. **API 延迟**：OpenRouter 调用约 2-5 秒，前端需显示 loading
2. **判断准确率**：需实际测试 Gemini Flash Lite 对桌面场景的判断效果
3. **API Key 安全**：当前硬编码在 config.py，后续可改为环境变量
4. **IPC 时序**：PLAN 命令必须在 START 之前写入控制文件
5. **Cloth 判断**：右下角区域的判断可能需要 prompt 调优
