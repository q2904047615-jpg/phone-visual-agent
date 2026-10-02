# Phone Visual Agent — A 版 · 上游识别 / ADB

这是完整通用手机视觉 Agent；福袋只是其中一个功能。普通任务仍由 Qwen 决策并交给原执行器。

## 本机试用

双击 `启动试用版.cmd`，访问 http://127.0.0.1:8766 。先手动打开直播间，再点“福袋模块”。
“暂停 / 继续任务 / 停止”控制福袋。刷新会暂停，不自动续跑。两版不能同时控制同一设备。
启动脚本只启动本版本 API，不操作卖家 main.exe，也不自动发起手机任务。

## 新机器安装

Python 3.12：`python setup_trial.py`，安装后 `python start_trial.py`。
使用原项目设备注册、ADB 配对、机械标定和 Qwen 环境变量。凭据不上传 GitHub。
Gmail 沿用 `GMAIL_SENDER` + 已授权 API token/refresh 配置，或应用专用密码。网页福袋功能配置返回真实通知状态；未配置时只保存本地通知，不代表邮件已发送。

## 福袋规则

无福袋每 60 秒新取图；参与后每 300 秒观察，接近开奖按已读取剩余时间提前。
只发 App 预填评论一次，不输入、不重发。明确没抽中点“知道了”并继续。
开奖时未见明确没抽中立即锁定停手，保存截图/时间，通知标题正文为“疑似中奖”，默认收件人 q2904047615@gmail.com。
不自动换房、关注、付费或做其他参与条件；当前版本的参与操作覆盖用户提供的评论类流程。
邮件失败仍停手，并显示原因。服务重启后须手动继续；已尝试发送不重放。

## 两版差异

A: `pokemonzlj/douyin_guaji` 的 `check_have_fudai` 固定提交 `894cd38476374b75077d25f9f54884e090b694be`，原方法保持不变；不运行其自动换房、验证码或夜间关 App 主循环。参与条件用本地 OCR，截图/点击用可信设备 ADB。
B: 自有多尺度模板检测 + RapidOCR，使用当前相机坐标交给既有机械执行器。初始模板来自用户的 iPhone 参考图，只是种子，不能证明适配 Android 实机。

B 更换/追加 Android 图标模板：`python calibrate_template.py --image 相机截图.png --box 左 上 右 下 --name android_bag`。
裁剪只保留图标，不含倒计时、弹窗或其他按钮。点击点由当前画面匹配结果生成，不使用参考图固定坐标。

## 证据边界

软件、离线流程/接口/识别测试与真实手机验收分开记录，见 `试用验收.md`。
A 的上游颜色/位置假设可能不识别当前红色钻石福袋；B 的透视、反光、缩放同样需实机样本检验。
依赖未配置、ADB 断开、相机/机械臂离线会明确显示技术原因，不宣称已参与。

## 来源

- 上游识别：https://github.com/pokemonzlj/douyin_guaji ，无可见许可证，安装时下载并核验，不随本库再分发源码。
- 模板识别方案参考 Auto.js `images.matchTemplate` 的机制，当前算法与编排为本项目独立实现。
- RapidOCR https://github.com/RapidAI/RapidOCR ，OpenCV https://github.com/opencv/opencv。
