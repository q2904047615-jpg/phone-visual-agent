# Visual Agent Companion IME

This is the project-owned Android text transport for the canonical
`input_verified_text` and `clear_verified_text` actions. It is deliberately not
a visual agent, UI locator, task planner, clipboard helper, Accessibility
service, ADB Keyboard, or shell bridge. It can only operate the Android
`InputConnection` that is currently owned by this IME.

## User setup

1. Build and install the APK once.
2. Open **Visual Agent Companion IME**.
3. Enter the controller host, TLS port, the controller certificate's SHA-256
   fingerprint, and a one-time pairing token. The token is never persisted.
4. Use the two buttons to open Android's input-method settings and select this
   IME.
5. Keep the phone and controller on a network on which the configured TLS port
   is reachable.

The phone initiates every TLS connection. A manually supplied certificate pin
is the trust root, so a private/self-signed controller certificate is allowed
without disabling TLS verification globally; TLS 1.0/1.1 are not enabled. The 32-byte pairing key returned
inside that pinned TLS channel is encrypted with an AES-GCM key held by Android
Keystore before it is written to `SharedPreferences`.

## Wire protocol (`2026-08-30-companion-ime-v1`)

Every frame is exactly `uint32_be length || UTF-8 JSON`, with a 1 MiB maximum.
Every message rejects missing and unknown keys. Authenticated messages use
lowercase 64-character hex HMAC-SHA256 over the inner canonical JSON object.
Canonical JSON is UTF-8, recursively sorts object keys, uses no insignificant
whitespace, and leaves Unicode unescaped. This matches Python
`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
Text is authenticated but is never logged or persisted by the Android app.

Initial pairing request keys:

```text
client_nonce, installation_id, one_time_token, protocol_version, type
```

`type` is `pair_request`. The exact response keys are:

```text
client_nonce, device_id, installation_id, pairing_id, protocol_version,
shared_key, type
```

`type` is `pair_response`; `shared_key` is standard base64 and must decode to
exactly 32 bytes. The controller must consume the one-time token atomically
before returning this response; the Android client never stores the token.
After encrypting the new key into a separate Android Keystore-backed pending
slot, while leaving any active key intact, the phone sends exact
`{confirm, signature}` with `type=pair_confirm` and the echoed
installation, client nonce, pairing, and device identities. The controller
verifies that HMAC without changing the active key, then returns exact
`{confirm_ack, signature}` with `type=pair_confirm_ack` and
`status=accepted`. Only then does Android promote pending to active and send a
signed `{commit, signature}` with `type=pair_commit`. The controller persists
and activates the key only after this post-promotion proof, returns signed
`pair_commit_ack`, and only then exposes setup completion. A dropped repair
connection before the confirm ACK therefore leaves the prior active pairing
unchanged. The setup CLI cannot report success from `pair_response` or
`pair_confirm` alone.
If a controller pairing already exists, this setup flow synchronizes that
same active key only to the same Android `installation_id`; it never rotates
or transfers a working key implicitly. A reinstalled app or replacement phone
requires explicit revocation on both sides followed by a fresh pairing.
Pairing and authenticated bridge traffic share the configured pinned-TLS
listener: its first length-prefixed JSON frame is either this direct
`pair_request` or the authenticated `hello` envelope below. A failed,
expired, or repeated token closes that setup connection without a response.

After pairing, the phone opens the TLS socket and first sends the exact envelope
`{hello, signature}`. `hello` contains:

```text
device_id, expires_at_epoch, issued_at_epoch, nonce, pairing_id,
protocol_version, type=bridge_hello
```

The server returns exact `{hello_ack, signature}`; `hello_ack` contains
`device_id, nonce, pairing_id, protocol_version, status=accepted,
type=bridge_hello_ack` and must echo the hello nonce.

For each active Android editor connection the IME generates a new opaque
`editor_session_id`, then sends exact `{ready, signature}`. `ready` contains:

```text
device_id, editor_session_id, expires_at_epoch, issued_at_epoch, nonce,
pairing_id, protocol_version, type=editor_ready
```

The server returns exact `{ready_ack, signature}`. It echoes the device,
pairing, editor-session and nonce and uses `type=editor_ready_ack` and
`status=accepted`. A new Android `InputConnection` always requires a fresh
ready exchange.

The controller then sends exact `{command, signature}`. `command` contains only
`operation, protocol_version, scope, text`; `scope` contains exactly:

```text
action_id, device_id, editor_session_id, expected_text_digest,
expires_at_epoch, fragment_text_digest, input_field_id, issued_at_epoch,
nonce, observation_fingerprint, prior_text_digest, protocol_version, revision,
session_id, task_id
```

`commit_text` carries a non-empty Unicode `text`; its UTF-8 SHA-256 must equal
`fragment_text_digest`, and its effect is only `finishComposingText()` followed
by one `commitText()`. If composition cannot be finished, the command remains
unknown and no commit/clear call follows. `clear_text` carries JSON `null`, never replacement text;
both its fragment and expected digests must be SHA-256 of the empty UTF-8
string. Clear is a separate command and can never silently replace text.
The complete authenticated command envelope—not only `text`—must fit within
the shared 1 MiB frame limit. Oversized text is rejected on the PC before any
send or physical attempt; an upper layer must create explicit canonical
segments instead of the transport retrying or splitting a command implicitly.

The command's `device_id` and `editor_session_id` must match the current ready
message. It must be unexpired and its HMAC must verify. The device, session,
action and nonce identity is reserved durably together with the canonical
command digest before any `InputConnection` call. A duplicate
completed command returns the prior accepted outcome in a newly authenticated
ACK without repeating the editor call. A duplicate whose final
result was not durably known—or the same action identity with a different
command digest—is not executed and the socket closes without an ACK, which the
PC records as `unknown` without retrying.

After `ready`, the authenticated TLS connection remains open for serial
commands on that same `editor_session_id`. Every command still receives at most
one ACK. An editor change, pairing change, unknown result, protocol failure, or
network failure closes it; a fresh editor requires a new hello/ready exchange.

An accepted action returns exact `{ack, signature}`. `ack` contains:

```text
acknowledged_at_epoch, action_id, command_digest, device_id, nonce, operation,
protocol_version, reason_code, status
```

`status=accepted` has JSON-null `reason_code`. The nonce, action ID and digest
bind the ACK to the signed command; that digest includes its
`editor_session_id`. This acknowledgement is transport evidence only. The PC
controller must still use a fresh camera observation and the existing exact
visual verifier before declaring the canonical action matched.

## Build and test

Required toolchain: JDK 17, Android SDK Platform 35, Android Build Tools 35, and
Gradle 8.9. Configure `ANDROID_HOME` (or an SDK path in `local.properties`). The
repository includes a Gradle 8.9 wrapper with its distribution SHA-256 pinned.
From this directory run:

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

On Windows, AGP's unit-test worker cannot reliably load classes when the
checkout path contains non-ASCII characters. Do not suppress that limitation
with `android.overridePathCheck`; use a temporary unused ASCII drive alias for
the build and remove it afterwards:

```powershell
$companionRepo = (Resolve-Path '..\..').Path
subst R: $companionRepo
try {
    Push-Location 'R:\android\companion-ime'
    try {
        .\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
    } finally {
        Pop-Location
    }
} finally {
    subst R: /D
}
```

Use another unused ASCII drive letter if `R:` already exists.

The expected APK is
`app/build/outputs/apk/debug/app-debug.apk`.

On 2026-08-30 this module was built on Windows with JDK 17.0.20.1, Gradle 8.9,
Platform 35 and Build Tools 35.0.0. All 38 JVM tests passed, `lintDebug` completed
with zero errors (13 warnings), and `assembleDebug` produced the expected APK.
This build evidence does not replace installation, input-method selection,
pairing, or real-device input acceptance.
