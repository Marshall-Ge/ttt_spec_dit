---
name: input_tokens 错误后自动续做
description: 遇到 undefined input_tokens 客户端错误时自动恢复任务，不以空响应或中间状态结束
type: feedback
originSessionId: ad0bd7f3-c296-4d8b-859b-4fb1ceaef042
---
遇到 `undefined is not an object (evaluating '_.input_tokens')`、工具结果截断或上下文恢复时，自动从最近的未完成任务继续，不输出空 final，不要求用户重复发送“继续”。优先拆分大文件读取、控制并行工具返回量，并将重要结果及时写入目标文件；只在确实无法安全继续时报告具体阻塞。

**Why:** 用户多次被该客户端 token 统计错误打断，当前文档持久化任务因此没有完成。

**How to apply:** 所有较长的代码、分析和文档任务都要持续到产物写入并验证；发生调用异常后先检查任务与文件现状，再直接续做。
