# Claude / cross-computer handoff — 2026-10-24

## Start from the correct branch

Repository: https://github.com/q2904047615-jpg/phone-visual-agent

Authoritative implementation branch: `codex/platform-policy-verifier-20261013`.

Read this branch explicitly. At the handoff audit, default `main` was `d3f2395` and still described the obsolete Qwen-owned action/finish loop. `lean-rebuild` at `8f283b1` was created directly from that old main commit. Its existence is not approval to replace the platform: the user explicitly approved independent verification only.

The implementation branch had been pushed only through `e753b34`. Nine additional local commits through `4d6f41a` were absent from all remote branches. The current synchronization batch publishes that history plus this documentation correction. Verify the remote branch head instead of relying on an earlier handoff message. Do not use the repository's default landing page as proof of the latest architecture.

The complete implementation plan, including website design stages, was also missing from the implementation branch. It has been restored byte-for-byte from the user's designated plan, whose Git blob is identical to historical commit `47f6a799a7bc9da4485dd6d885afc904a4638a91` (`docs: align website plan with agent platform patterns`). Its historical environment and authorization notes are provenance, not instructions to revert newer user decisions. Use current user decisions and handoff records for subsequent implementation and acceptance status.

## Required reading on this branch

- [AGENTS.md](../AGENTS.md)
- [Product authority](../项目最终目标.md), especially the 2026-10-12 section overriding historical sections.
- [User decisions and protocol boundaries](../用户决策与协议边界.md)
- [Implementation plan](项目首席架构师与验收执行提示词.md)
- [Latest handoff](../项目交接文档.md)
- [Remote-tool acceptance](remote_tools_acceptance_20261023.md)
- [Production validation](production_validation_20261024.md)

Older documents may preserve historical designs. Do not treat their Qwen-only descriptions as authority over the explicit current platform decision. The README's obsolete opening was corrected in this synchronization batch; this does not claim every historical document has been rewritten.

## Product and architecture

This is a general Agent platform with phones as the first execution connector:

`Task/Run → Planner → Policy/Approval → Executor → Observation → Verifier → Recovery/Replan/Completion`.

Runtime owns durable lifecycle transitions. Qwen is one Model Gateway provider and may supply observations or proposals. It must not directly own canonical actions, policy approval, or business completion. Preserve the per-phone physical loop: fresh observation, one authorized action, fresh observation.

The deployed architecture has multiple API ingress workers, a single exclusive Runtime executor, and an authenticated Windows device connector. Do not restore competing Runtime instances or a parallel legacy action authority.

Inspect implementation under `poc/agent/`, the composition root `poc/agent/bootstrap/runtime.py`, `poc/multi_worker_api.py`, `poc/single_executor.py`, and `poc/device_connector.py`. Confirm actual behavior and tests; directory names alone do not prove architectural completion.

## Implementation is not deployment

- `adcab73` implements station-bound remote trusted phone tools using connector protocol v3. It requires coordinated cloud and Windows connector deployment. It was not deployed at handoff.
- `4d6f41a` contains production backup repair, bounded read-probe tooling, and evidence. The backup maintenance scripts were loaded independently of the API release.
- The last verified public API release was `21696e1`. A GitHub push does not update that service.
- The production record establishes 600 successful bounded read rounds, 60 anonymous denials, three-store backup/isolated restore, idle process recovery, and 20 post-recovery read rounds. Executor recovery caused temporary 503 responses. These are not zero-downtime, capacity, or in-flight physical-task guarantees.
- A Gmail test was sent through the assistant's connector. Server-side automatic alerting is still missing sending configuration and operational triggers.
- A second real GitHub account is unavailable. Real two-account isolation remains unverified.
- Phone connection, real tool/business acceptance, connector persistence/startup, and broader physical multi-device evidence remain separate work items.

## Reassess the rebuild without discarding either history

1. Fetch and inspect the authoritative implementation branch; report its exact SHA.
2. Compare its plan, code, tests, control plane, persistence, permissions, model/tool boundaries, deployment, and recovery with `lean-rebuild`.
3. Report retained capabilities, lost capabilities, architecture conflicts, useful simplifications, and unverified claims with file references.
4. Recommend how to reuse useful rebuild work while preserving the user's platform requirements. Do not assume a full rewrite is necessary.
5. Do not merge the rebuild, replace the platform, or silently alter product authority. Obtain a decision if the proposed direction changes the architecture or removes capabilities.

From another computer, work only with the repository and isolated tests available there. Original-machine paths, credentials, browser sessions, and physical devices are not accessible merely because they appear in a report. Never copy secrets into commits. Distinguish implemented, tested, deployed, and physically verified status. Communicate with the user in Chinese.
