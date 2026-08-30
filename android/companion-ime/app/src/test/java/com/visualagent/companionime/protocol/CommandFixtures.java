package com.visualagent.companionime.protocol;

import org.json.JSONObject;

public final class CommandFixtures {
    public static final byte[] KEY = new byte[32];
    public static final double NOW = 1_800_000_000.5;
    public static final String DEVICE = "device-local-01";
    public static final String EDITOR = "editor-1";

    private CommandFixtures() {
    }

    public static JSONObject envelope(String operation, String text) throws Exception {
        JSONObject scope = new JSONObject();
        scope.put("protocol_version", ProtocolConstants.VERSION);
        scope.put("device_id", DEVICE);
        scope.put("session_id", "session-1");
        scope.put("task_id", "task-1");
        scope.put("revision", 7);
        scope.put("action_id", "action-1");
        scope.put("input_field_id", "field-1");
        scope.put("editor_session_id", EDITOR);
        scope.put("observation_fingerprint", "observation-1");
        scope.put("prior_text_digest", Hashing.sha256Hex("prior"));
        scope.put("fragment_text_digest", Hashing.sha256Hex(
                "clear_text".equals(operation) ? "" : text));
        scope.put("expected_text_digest", Hashing.sha256Hex(
                "clear_text".equals(operation) ? "" : "prior" + text));
        scope.put("issued_at_epoch", NOW - 1.0);
        scope.put("expires_at_epoch", NOW + 10.0);
        scope.put("nonce", "nonce-1234567890123456");

        JSONObject command = new JSONObject();
        command.put("protocol_version", ProtocolConstants.VERSION);
        command.put("operation", operation);
        command.put("scope", scope);
        command.put("text", text == null ? JSONObject.NULL : text);

        JSONObject envelope = new JSONObject();
        envelope.put("command", command);
        envelope.put("signature", HmacAuthenticator.sign(command, KEY));
        return envelope;
    }

    public static CommandEnvelope parse(JSONObject value) throws Exception {
        return CommandEnvelope.parseAndAuthenticate(
                value, KEY, DEVICE, EDITOR, NOW);
    }

    public static JSONObject command(JSONObject envelope) throws Exception {
        return envelope.getJSONObject("command");
    }

    public static JSONObject scope(JSONObject envelope) throws Exception {
        return command(envelope).getJSONObject("scope");
    }

    public static void resign(JSONObject envelope) throws Exception {
        envelope.put("signature", HmacAuthenticator.sign(command(envelope), KEY));
    }
}
