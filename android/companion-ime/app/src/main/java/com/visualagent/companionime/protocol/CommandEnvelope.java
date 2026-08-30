package com.visualagent.companionime.protocol;

import org.json.JSONException;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public final class CommandEnvelope {
    public enum Operation {
        COMMIT_TEXT,
        CLEAR_TEXT
    }

    private static final Set<String> ENVELOPE_KEYS = exact("command", "signature");
    private static final Set<String> COMMAND_KEYS = exact(
            "protocol_version", "operation", "scope", "text");
    private static final Set<String> SCOPE_KEYS = exact(
            "protocol_version", "device_id", "session_id", "task_id", "revision",
            "action_id", "input_field_id", "editor_session_id", "observation_fingerprint",
            "prior_text_digest", "fragment_text_digest", "expected_text_digest",
            "issued_at_epoch", "expires_at_epoch", "nonce");
    private static final String EMPTY_SHA256 = Hashing.sha256Hex("");

    private final String deviceId;
    private final String sessionId;
    private final String editorSessionId;
    private final String actionId;
    private final String nonce;
    private final String commandDigest;
    private final Operation operation;
    private final String text;

    private CommandEnvelope(
            String deviceId,
            String sessionId,
            String editorSessionId,
            String actionId,
            String nonce,
            String commandDigest,
            Operation operation,
            String text) {
        this.deviceId = deviceId;
        this.sessionId = sessionId;
        this.editorSessionId = editorSessionId;
        this.actionId = actionId;
        this.nonce = nonce;
        this.commandDigest = commandDigest;
        this.operation = operation;
        this.text = text;
    }

    public static CommandEnvelope parseAndAuthenticate(
            JSONObject envelope,
            byte[] sharedKey,
            String expectedDeviceId,
            String expectedEditorSessionId,
            double nowEpoch) throws ProtocolException {
        StrictJson.requireExactKeys(envelope, ENVELOPE_KEYS);
        JSONObject command = requireObject(envelope, "command");
        String signature = StrictJson.requireString(envelope, "signature", 64);
        StrictJson.requireExactKeys(command, COMMAND_KEYS);
        StrictJson.requireProtocolVersion(command);
        HmacAuthenticator.verify(command, signature, sharedKey);

        String operationName = StrictJson.requireString(command, "operation", 32);
        Operation operation;
        if ("commit_text".equals(operationName)) {
            operation = Operation.COMMIT_TEXT;
        } else if ("clear_text".equals(operationName)) {
            operation = Operation.CLEAR_TEXT;
        } else {
            throw new ProtocolException("unsupported command operation");
        }

        JSONObject scope = requireObject(command, "scope");
        StrictJson.requireExactKeys(scope, SCOPE_KEYS);
        StrictJson.requireProtocolVersion(scope);
        String deviceId = StrictJson.requireIdentifier(scope, "device_id");
        String editorSessionId = StrictJson.requireIdentifier(scope, "editor_session_id");
        if (!expectedDeviceId.equals(deviceId)
                || !expectedEditorSessionId.equals(editorSessionId)) {
            throw new ProtocolException("command is not bound to the active device editor");
        }

        String sessionId = StrictJson.requireIdentifier(scope, "session_id");
        StrictJson.requireIdentifier(scope, "task_id");
        StrictJson.requireLong(scope, "revision", 0);
        String actionId = StrictJson.requireIdentifier(scope, "action_id");
        StrictJson.requireIdentifier(scope, "input_field_id");
        StrictJson.requireIdentifier(scope, "observation_fingerprint");
        StrictJson.requireSha256(scope, "prior_text_digest");
        String fragmentDigest = StrictJson.requireSha256(scope, "fragment_text_digest");
        String expectedDigest = StrictJson.requireSha256(scope, "expected_text_digest");
        double issuedAt = StrictJson.requireFiniteDouble(scope, "issued_at_epoch");
        double expiresAt = StrictJson.requireFiniteDouble(scope, "expires_at_epoch");
        String nonce = StrictJson.requireNonce(scope, "nonce");
        if (nowEpoch < issuedAt || nowEpoch > expiresAt || expiresAt <= issuedAt) {
            throw new ProtocolException("command is outside its validity window");
        }

        String text = null;
        Object rawText = get(command, "text");
        if (operation == Operation.COMMIT_TEXT) {
            if (!(rawText instanceof String) || ((String) rawText).isEmpty()) {
                throw new ProtocolException("commit_text requires non-empty text");
            }
            text = (String) rawText;
            if (text.indexOf('\r') >= 0
                    || text.getBytes(StandardCharsets.UTF_8).length
                    > ProtocolConstants.MAX_TEXT_BYTES) {
                throw new ProtocolException("commit_text contains unsupported text");
            }
            String actual = Hashing.sha256Hex(text);
            if (!MessageDigest.isEqual(
                    actual.getBytes(StandardCharsets.US_ASCII),
                    fragmentDigest.getBytes(StandardCharsets.US_ASCII))) {
                throw new ProtocolException("text does not match fragment_text_digest");
            }
        } else if (rawText != JSONObject.NULL) {
            throw new ProtocolException("clear_text must carry JSON null text");
        } else if (!EMPTY_SHA256.equals(fragmentDigest) || !EMPTY_SHA256.equals(expectedDigest)) {
            throw new ProtocolException("clear_text must bind empty fragment and expected digests");
        }

        String commandDigest = Hashing.sha256Hex(
                CanonicalJson.serialize(command).getBytes(StandardCharsets.UTF_8));
        return new CommandEnvelope(
                deviceId, sessionId, editorSessionId, actionId, nonce, commandDigest,
                operation, text);
    }

    public String deviceId() {
        return deviceId;
    }

    public String editorSessionId() {
        return editorSessionId;
    }

    public String replayIdentity() {
        return component(deviceId) + component(sessionId) + component(actionId) + component(nonce);
    }

    public String actionId() {
        return actionId;
    }

    public String nonce() {
        return nonce;
    }

    public String commandDigest() {
        return commandDigest;
    }

    public Operation operation() {
        return operation;
    }

    public String wireOperation() {
        return operation == Operation.COMMIT_TEXT ? "commit_text" : "clear_text";
    }

    public String text() {
        return text;
    }

    private static JSONObject requireObject(JSONObject parent, String key) throws ProtocolException {
        Object value = get(parent, key);
        if (!(value instanceof JSONObject)) {
            throw new ProtocolException(key + " must be a JSON object");
        }
        return (JSONObject) value;
    }

    private static Object get(JSONObject parent, String key) throws ProtocolException {
        try {
            return parent.get(key);
        } catch (JSONException error) {
            throw new ProtocolException("missing key: " + key, error);
        }
    }

    private static Set<String> exact(String... keys) {
        return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(keys)));
    }

    private static String component(String value) {
        return value.length() + ":" + value;
    }
}
