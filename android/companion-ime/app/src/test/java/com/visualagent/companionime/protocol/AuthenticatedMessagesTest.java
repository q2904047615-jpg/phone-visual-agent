package com.visualagent.companionime.protocol;

import com.visualagent.companionime.foreground.ForegroundAppIdentity;

import org.json.JSONObject;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public final class AuthenticatedMessagesTest {
    @Test
    public void generatedNoncesAlwaysMatchPythonContract() {
        for (int index = 0; index < 1_000; index++) {
            String nonce = Nonce.create();
            assertTrue(nonce.matches("[A-Za-z0-9][A-Za-z0-9._:-]{15,127}"));
        }
    }

    @Test
    public void readyCarriesCurrentEditorAndValidHexSignature() throws Exception {
        JSONObject envelope = AuthenticatedMessages.editorReady(
                "pairing-1", "device-1", "editor-1", 1000.25, CommandFixtures.KEY);
        JSONObject ready = envelope.getJSONObject("ready");

        assertEquals("editor_ready", ready.getString("type"));
        assertEquals("editor-1", ready.getString("editor_session_id"));
        HmacAuthenticator.verify(
                ready, envelope.getString("signature"), CommandFixtures.KEY);
        assertEquals(new HashSet<>(Arrays.asList("ready", "signature")), keys(envelope));
        assertEquals(new HashSet<>(Arrays.asList(
                        "protocol_version", "type", "device_id", "pairing_id",
                        "editor_session_id", "issued_at_epoch", "expires_at_epoch", "nonce")),
                keys(ready));
    }

    @Test
    public void helloMatchesExactPythonEnvelopeAndInnerKeys() throws Exception {
        JSONObject envelope = AuthenticatedMessages.bridgeHello(
                "pairing-1", "device-1", 1000.25, CommandFixtures.KEY);
        JSONObject hello = envelope.getJSONObject("hello");

        assertEquals(new HashSet<>(Arrays.asList("hello", "signature")), keys(envelope));
        assertEquals(new HashSet<>(Arrays.asList(
                        "protocol_version", "type", "device_id", "pairing_id",
                        "issued_at_epoch", "expires_at_epoch", "nonce")),
                keys(hello));
        HmacAuthenticator.verify(
                hello, envelope.getString("signature"), CommandFixtures.KEY);
    }

    @Test
    public void foregroundStateUsesSeparateProtocolAndExactSignedShape() throws Exception {
        ForegroundAppIdentity identity = ForegroundAppIdentity.usageStats(
                "com.android.settings", 990.0, 1000.0);
        JSONObject envelope = AuthenticatedMessages.foregroundState(
                "pairing-1", "device-1", identity, 1000.25, CommandFixtures.KEY);
        JSONObject state = envelope.getJSONObject("foreground_state");

        assertEquals(ProtocolConstants.FOREGROUND_IDENTITY_VERSION,
                state.getString("protocol_version"));
        assertEquals("com.android.settings", state.getString("package_name"));
        assertEquals("usage_stats", state.getString("source"));
        assertTrue(state.isNull("reason_code"));
        assertEquals(new HashSet<>(Arrays.asList("foreground_state", "signature")),
                keys(envelope));
        assertEquals(new HashSet<>(Arrays.asList(
                        "protocol_version", "type", "device_id", "pairing_id",
                        "package_name", "source", "event_at_epoch", "observed_at_epoch",
                        "reason_code", "issued_at_epoch", "expires_at_epoch", "nonce")),
                keys(state));
        HmacAuthenticator.verify(
                state, envelope.getString("signature"), CommandFixtures.KEY);
    }

    @Test
    public void actionAckBindsScopeNonceAndCommandDigestWithoutText() throws Exception {
        CommandEnvelope command = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", "你好"));
        JSONObject envelope = AuthenticatedMessages.actionAck(
                command, "accepted", null, CommandFixtures.NOW, CommandFixtures.KEY);
        JSONObject ack = envelope.getJSONObject("ack");

        assertEquals(command.nonce(), ack.getString("nonce"));
        assertEquals(command.actionId(), ack.getString("action_id"));
        assertEquals(command.commandDigest(), ack.getString("command_digest"));
        assertTrue(ack.isNull("reason_code"));
        assertEquals(new HashSet<>(Arrays.asList("ack", "signature")), keys(envelope));
        assertEquals(new HashSet<>(Arrays.asList(
                        "protocol_version", "device_id", "action_id", "nonce", "operation",
                        "status", "reason_code", "command_digest", "acknowledged_at_epoch")),
                keys(ack));
        assertTrue(!ack.has("text"));
        assertTrue(!ack.has("editor_session_id"));
        HmacAuthenticator.verify(ack, envelope.getString("signature"), CommandFixtures.KEY);
    }

    private static Set<String> keys(JSONObject object) {
        Set<String> keys = new HashSet<>();
        Iterator<String> iterator = object.keys();
        while (iterator.hasNext()) {
            keys.add(iterator.next());
        }
        return keys;
    }
}
