# configs/

训练和环境配置文件。所有 YAML 参数均带行内注释说明用途和约束。

## 命名规则

```text
<stage>_<purpose>[_<model>].yaml
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `sft_cold_start_4b.yaml` | Qwen3-4B SFT cold-start（格式对齐） |
| `grpo_smoke.yaml` | Replay GRPO smoke test（2 步验证链路） |
| `grpo_train.yaml` | Replay GRPO 正式训练 |
| `ds_zero2.json` | DeepSpeed ZeRO-2（JSON 格式，DS 直接消费） |
| `live_mcp/suite_mvp.yaml` | Live MCP MVP 套件总配置 |
| `live_mcp/backend_live.yaml` | 覆盖层：启用 live 执行 |
| `live_mcp/backend_replay.yaml` | 覆盖层：强制 replay |
| `live_mcp/calendar.yaml` | Calendar server 定义 |
| `live_mcp/shopping.yaml` | Shopping server 定义 |

## 注意

- `ds_zero2.json` 保持 JSON 格式（DeepSpeed 不支持 YAML）
- Live MCP 配置是可选并行分支，不影响默认 replay 训练路径
- 所有路径使用项目根目录相对路径，禁止写死机器绝对路径
