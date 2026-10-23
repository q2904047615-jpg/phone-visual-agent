# 测试结构与离线入口

以下命令均在 `poc` 目录运行，不调用真实手机或付费模型。

日常 Python 回归（含根目录核心/维护/组合测试及 `contract_tests`）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s . -p 'test_*.py'
```

历史实验保护测试独立运行，不计入日常通过数：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s experiments/historical_tests -p 'test_*.py'
```

完整离线回归须分别运行上面两条及 `npm test`，三组均通过才可汇报完整回归。

- `test_support/`：共同夹具、假设备、假模型及没有测试方法的基类。消费者直接导入正式类；夹具不得拦截执行入口来偷偷补当前请求字段、截图或授权。
- `contract_tests/input/`：输入、焦点、正文合同。
- `contract_tests/observation/`：当前图、scope、点按投影及画面变化合同。
- `contract_tests/protocol/`：动作解析、提示词、导航和结果权威合同。
- `contract_tests/retirement/`：已退役机制不得恢复。
- 根目录按控制器、观察器、适配器、DDD依赖、Web、硬件维护等实际职责分组。跨模块组合测试仍保留，不能由身份/文件名断言替代。

历史实验保留固定配额、停止状态和防重放检查；这次隔离不授权重新采样。部分模型响应构造夹具仍会规范化合成数据，不能将这些单测等同于原始响应端到端验证；原始响应组合测试单独保留。
