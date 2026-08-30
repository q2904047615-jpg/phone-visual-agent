package com.visualagent.companionime.protocol;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public final class AuthenticatedMessages {
    private static final double SESSION_AUTH_SECONDS = 30.0;
    private static final Set<String> ENVELOPE_KEYS = exact("hello_ack", "signature");
    private static final Set<String> HELLO_ACK_KEYS = exact(
            "protocol_version", "type", "device_id", "pairing_id", "nonce", "status");
    private static final Set<String> READY_ENVELOPE_KEYS = exact("ready_ack", "signature");
    private static final Set<String> READY_ACK_KEYS = exact(
            "protocol_version", "type", "device_id", "pairing_id", "editor_session_id",
            "nonce", "status");

    private AuthenticatedMessages() {
    }

    public static JSONObject bridgeHello(
            String pairingId,
            String deviceId,
            double nowEpoch,
            byte[] sharedKey) throws ProtocolException {
        JSONObject hello = new JSONObject();
        try {
            hello.put("protocol_version", ProtocolConstants.VERSION);
            hello.put("type", "bridge_hello");
            hello.put("device_id", deviceId);
            hello.put("pairing_id", pairingId);
            hello.put("issued_at_epoch", nowEpoch);
            hello.put("expires_at_epoch", nowEpoch + SESSION_AUTH_SECONDS);
            hello.put("nonce", Nonce.create());
            return envelope("hello", hello, sharedKey);
        } catch (JSONException error) {
            throw new ProtocolException("unable to create bridge hello", error);
        }
    }

    public static void verifyBridgeHelloAck(
            JSONObject envelope,
            String pairingId,
            String deviceId,
            String helloNonce,
            byte[] sharedKey) throws ProtocolException {
        StrictJson.requireExactKeys(envelope, ENVELOPE_KEYS);
        JSONObject ack = requireObject(envelope, "hello_ack");
        StrictJson.requireExactKeys(ack, HELLO_ACK_KEYS);
        HmacAuthenticator.verify(
                ack, StrictJson.requireString(envelope, "signature", 64), sharedKey);
        StrictJson.requireProtocolVersion(ack);
        if (!"bridge_hello_ack".equals(StrictJson.requireString(ack, "type", 64))
                || !deviceId.equals(StrictJson.requireIdentifier(ack, "device_id"))
                || !pairingId.equals(StrictJson.requireIdentifier(ack, "pairing_id"))
                || !helloNonce.equals(StrictJson.requireNonce(ack, "nonce"))
                || !"accepted".equals(StrictJson.requireString(ack, "status", 32))) {
            throw new ProtocolException("bridge hello acknowledgement is not bound to this client");
        }
    }

    public static JSONObject editorReady(
            String pairingId,
            String deviceId,
            String editorSessionId,
            double nowEpoch,
            byte[] sharedKey) throws ProtocolException {
        JSONObject ready = new JSONObject();
        try {
            ready.put("protocol_version", ProtocolConstants.VERSION);
            ready.put("type", "editor_ready");
            ready.put("device_id", deviceId);
            ready.put("pairing_id", pairingId);
            ready.put("editor_session_id", editorSessionId);
            ready.put("issued_at_epoch", nowEpoch);
            ready.put("expires_at_epoch", nowEpoch + SESSION_AUTH_SECONDS);
            ready.put("nonce", Nonce.create());
            return envelope("ready", ready, sharedKey);
        } catch (JSONException error) {
            throw new ProtocolException("unable to create editor ready", error);
        }
    }

    public static void verifyEditorReadyAck(
            JSONObject envelope,
            String pairingId,
            String deviceId,
            String editorSessionId,
            String readyNonce,
            byte[] sharedKey) throws ProtocolException {
        StrictJson.requireExactKeys(envelope, READY_ENVELOPE_KEYS);
        JSONObject ack = requireObject(envelope, "ready_ack");
        StrictJson.requireExactKeys(ack, READY_ACK_KEYS);
        HmacAuthenticator.verify(
                ack, StrictJson.requireString(envelope, "signature", 64), sharedKey);
        StrictJson.requireProtocolVersion(ack);
        if (!"editor_ready_ack".equals(StrictJson.requireString(ack, "type", 64))
                || !deviceId.equals(StrictJson.requireIdentifier(ack, "device_id"))
                || !pairingId.equals(StrictJson.requireIdentifier(ack, "pairing_id"))
                || !editorSessionId.equals(
                        StrictJson.requireIdentifier(ack, "editor_session_id"))
                || !readyNonce.equals(StrictJson.requireNonce(ack, "nonce"))
                || !"accepted".equals(StrictJson.requireString(ack, "status", 32))) {
            throw new ProtocolException("editor ready acknowledgement is not bound to this editor");
        }
    }

    public static JSONObject actionAck(
            CommandEnvelope command,
            String status,
            String reasonCode,
            double nowEpoch,
            byte[] sharedKey) throws ProtocolException {
        if (!"accepted".equals(status) && !"rejected".equals(status)) {
            throw new ProtocolException("action acknowledgement status is invalid");
        }
        if ("accepted".equals(status) && reasonCode != null) {
            throw new ProtocolException("accepted acknowledgement cannot contain a reason");
        }
        if ("rejected".equals(status)
                && (reasonCode == null || !reasonCode.matches("[a-z][a-z0-9_.-]{0,63}"))) {
            throw new ProtocolException("rejected acknowledgement requires a reason code");
        }
        JSONObject ack = new JSONObject();
        try {
            ack.put("protocol_version", ProtocolConstants.VERSION);
            ack.put("device_id", command.deviceId());
            ack.put("action_id", command.actionId());
            ack.put("nonce", command.nonce());
            ack.put("operation", command.wireOperation());
            ack.put("status", status);
            ack.put("reason_code", reasonCode == null ? JSONObject.NULL : reasonCode);
            ack.put("command_digest", command.commandDigest());
            ack.put("acknowledged_at_epoch", nowEpoch);
            return envelope("ack", ack, sharedKey);
        } catch (JSONException error) {
            throw new ProtocolException("unable to create action acknowledgement", error);
        }
    }

    public static String innerNonce(JSONObject envelope, String name) throws ProtocolException {
        return StrictJson.requireNonce(requireObject(envelope, name), "nonce");
    }

    private static JSONObject envelope(String name, JSONObject value, byte[] sharedKey)
            throws ProtocolException {
        JSONObject envelope = new JSONObject();
        try {
            envelope.put(name, value);
            envelope.put("signature", HmacAuthenticator.sign(value, sharedKey));
            return envelope;
        } catch (JSONException error) {
            throw new ProtocolException("unable to create authenticated envelope", error);
        }
    }

    private static JSONObject requireObject(JSONObject parent, String key) throws ProtocolException {
        try {
            Object value = parent.get(key);
            if (!(value instanceof JSONObject)) {
                throw new ProtocolException(key + " must be a JSON object");
            }
            return (JSONObject) value;
        } catch (JSONException error) {
            throw new ProtocolException("missing key: " + key, error);
        }
    }

    private static Set<String> exact(String... keys) {
        return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(keys)));
    }
}
