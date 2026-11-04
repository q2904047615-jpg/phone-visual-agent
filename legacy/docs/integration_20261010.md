# 2026-10-10 手机自动点击器整合验收记录

## 整合范围

- 整合分支：`codex/integration-qishui-console-20261010`。
- 独立整合目录：`C:\Users\Administrator\AppData\Local\Temp\phone-integration-20261010`。
- 基线：本地可用的 6e64 工作树提交 `6056c944`。6e64 工作树的当前 `poc/static/app.js`、`index.html`、`styles.css`、`poc/web_app.py` 作为网页权威来源；原工作树和 6e64 工作树均未修改。
- 汽水来源：本地 dd57 工作树中的 `qishui_ad_test` 实现和契约测试。用户指定的 `b198694...` 对象不在本地，网络无法抓取。
- 用户指定的 main `57e15de...` 和任务控制台 UI `c640896...` 对象同样不在本地；本轮使用本地 6e64 当前网页和可见主线支撑，并保留缺口记录。

## 已接入能力

- 6e64 当前网页：常驻实时预览（`refreshPreview` 700ms）、Qwen 原生回答/对话区、设备选择、响应式弹层、短屏输入区和当前设备预览。
- 设备选择器读取当前注册表的 10 个设备：`device-local-01` 至 `device-local-10`。
- 汽水测试网关和生命周期接入统一会话服务，提供：
  - `GET /api/features/qishui-ad-test`
  - `POST /api/features/qishui-ad-test/start`
  - `GET /api/features/qishui-ad-test/{test_id}`
  - `POST /api/features/qishui-ad-test/{test_id}/observe`
  - `POST /api/features/qishui-ad-test/{test_id}/confirm`
  - `POST /api/features/qishui-ad-test/{test_id}/cancel`
- 汽水功能只观察用户手动打开的一次广告并在奖励/结束页等待人工确认，不自动点击广告、不自动领取金币；该专项能力不能替代通用跨 App 任务验收。

## 离线验证结果

测试使用固定临时目录适配当前受管 Windows 沙箱的 `TemporaryDirectory` 子目录权限限制；没有修改生产代码或测试断言来绕过该环境问题。

| 验证项 | 结果 |
| --- | --- |
| 顶层 Python 测试（当前整合范围） | 724 通过，3 跳过，0 失败，0 错误 |
| Python 契约测试 | 133 通过，0 失败，0 错误 |
| Qishui 核心测试 | 5/5 通过 |
| Qishui Web 测试 | 3/3 通过 |
| 前端单元测试 | 19/19 通过 |
| 浏览器契约 `test_frontend_browser_contract.js` | 21/21 通过 |
| 浏览器契约 `test_phase2_console_browser_contract.js` | 4/4 通过 |
| `python -m compileall -q poc` | 通过 |
| `python -m pip check` | `No broken requirements found` |
| `git diff --check` | 通过（仅报告 LF/CRLF 转换提示） |

静态和浏览器尺寸核对覆盖 390x530 与 390x844：输入框在短屏仍可见且可滚动，弹层保持 `overflow:auto`；同时核对了实时预览、Qwen 区域和十个设备注册项。该核对是网页/离线证据，不是真机证据。

## 尚未完成的证据和风险

- 未完成跨 App、多设备真实设备动作验收，也没有完成“自然语言任务 → 截图观察 → 动作 → 新截图确认效果”的同会话真机证据。当前离线 fixture、浏览器和 Mock 结果不能替代该证据。
- 未重载项目 API，没有操作卖家 `main.exe`，也没有推送或合并 main；真实设备、控制器、相机和 Qwen 在线状态应由后续主任务在具备环境后只读核对。
- GitHub 网络请求失败：`Failed to connect to github.com port 443 via 127.0.0.1`，因此无法核验用户指定的三个远程提交对象。该整合以本地 6e64 当前网页、可见本地主线和 dd57 汽水源码为证据，不能声称与远程对象逐字一致。
- 本记录是整合范围的证据清单，不是全项目审查结论；依赖、生成物和历史工作树未作为完整源码审查范围。
