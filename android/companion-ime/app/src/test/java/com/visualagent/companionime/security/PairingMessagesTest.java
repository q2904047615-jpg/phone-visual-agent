package com.visualagent.companionime.security;

import com.visualagent.companionime.protocol.ProtocolConstants;
import com.visualagent.companionime.protocol.ProtocolException;
import com.visualagent.companionime.protocol.HmacAuthenticator;

import org.json.JSONObject;
import org.junit.Test;

import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class PairingMessagesTest {
    private static final String NONCE = "nonce-1234567890123456";
    private static final String INSTALLATION = "installation-1";

    @Test
    public void requestMatchesExactPcPairingContractWithoutEnvelope() throws Exception {
        JSONObject request = PairingMessages.pairRequest(INSTALLATION, NONCE, "one-time-token");

        assertEquals(new HashSet<>(Arrays.asList(
                        "protocol_version", "type", "installation_id", "client_nonce",
                        "one_time_token")),
                keys(request));
        assertEquals(ProtocolConstants.VERSION, request.getString("protocol_version"));
        assertEquals("pair_request", request.getString("type"));
        assertEquals("one-time-token", request.getString("one_time_token"));
    }

    @Test
    public void responseRequiresExactEchoAndThirtyTwoByteStandardBase64Key() throws Exception {
        byte[] key = new byte[32];
        key[0] = 7;
        JSONObject response = response(key);

        PairingRecord record = PairingMessages.pairResponse(
                response,
                "controller.local",
                9443,
                repeat("aa", 32),
                INSTALLATION,
                NONCE);
        try {
            assertEquals("pairing-1", record.pairingId());
            assertEquals("device-local-01", record.deviceId());
            assertArrayEquals(key, record.sharedKey());
        } finally {
            record.destroy();
        }
    }

    @Test
    public void responseRejectsExtraKeyWrongEchoAndWrongKeyLength() throws Exception {
        JSONObject extra = response(new byte[32]);
        extra.put("capabilities", "not-negotiated-here");
        assertThrows(ProtocolException.class, () -> PairingMessages.pairResponse(
                extra, "host", 443, repeat("aa", 32), INSTALLATION, NONCE));

        JSONObject wrongEcho = response(new byte[32]);
        wrongEcho.put("client_nonce", "different-1234567890123456");
        assertThrows(ProtocolException.class, () -> PairingMessages.pairResponse(
                wrongEcho, "host", 443, repeat("aa", 32), INSTALLATION, NONCE));

        JSONObject shortKey = response(new byte[31]);
        assertThrows(ProtocolException.class, () -> PairingMessages.pairResponse(
                shortKey, "host", 443, repeat("aa", 32), INSTALLATION, NONCE));
    }

    @Test
    public void requestRejectsInvalidInstallationAndTokenBeforeNetworkUse() {
        assertThrows(ProtocolException.class,
                () -> PairingMessages.pairRequest("invalid installation", NONCE, "token"));
        assertThrows(ProtocolException.class,
                () -> PairingMessages.pairRequest(INSTALLATION, NONCE, ""));
    }

    @Test
    public void signedConfirmationBindsTheDurablyStoredPhoneKeyBeforePcSuccess() throws Exception {
        byte[] key = new byte[32];
        key[0] = 7;
        PairingRecord record = PairingMessages.pairResponse(
                response(key), "controller.local", 9443, repeat("aa", 32), INSTALLATION, NONCE);
        try {
            JSONObject envelope = PairingMessages.pairConfirm(record, NONCE);
            assertEquals(new HashSet<>(Arrays.asList("confirm", "signature")), keys(envelope));
            JSONObject confirm = envelope.getJSONObject("confirm");
            assertEquals("pair_confirm", confirm.getString("type"));
            HmacAuthenticator.verify(confirm, envelope.getString("signature"), key);

            JSONObject ack = new JSONObject(confirm.toString());
            ack.put("type", "pair_confirm_ack");
            ack.put("status", "accepted");
            JSONObject ackEnvelope = new JSONObject();
            ackEnvelope.put("confirm_ack", ack);
            ackEnvelope.put("signature", HmacAuthenticator.sign(ack, key));
            PairingMessages.verifyPairConfirmAck(ackEnvelope, record, NONCE);

            JSONObject commitEnvelope = PairingMessages.pairCommit(record, NONCE);
            JSONObject commit = commitEnvelope.getJSONObject("commit");
            assertEquals("pair_commit", commit.getString("type"));
            HmacAuthenticator.verify(commit, commitEnvelope.getString("signature"), key);
            JSONObject commitAck = new JSONObject(commit.toString());
            commitAck.put("type", "pair_commit_ack");
            commitAck.put("status", "accepted");
            JSONObject commitAckEnvelope = new JSONObject();
            commitAckEnvelope.put("commit_ack", commitAck);
            commitAckEnvelope.put("signature", HmacAuthenticator.sign(commitAck, key));
            PairingMessages.verifyPairCommitAck(commitAckEnvelope, record, NONCE);

            ack.put("client_nonce", "wrong-nonce-123456789");
            ackEnvelope.put("signature", HmacAuthenticator.sign(ack, key));
            assertThrows(ProtocolException.class,
                    () -> PairingMessages.verifyPairConfirmAck(ackEnvelope, record, NONCE));
        } finally {
            record.destroy();
        }
    }

    private static JSONObject response(byte[] key) throws Exception {
        JSONObject response = new JSONObject();
        response.put("protocol_version", ProtocolConstants.VERSION);
        response.put("type", "pair_response");
        response.put("installation_id", INSTALLATION);
        response.put("client_nonce", NONCE);
        response.put("pairing_id", "pairing-1");
        response.put("device_id", "device-local-01");
        response.put("shared_key", Base64.getEncoder().encodeToString(key));
        return response;
    }

    private static String repeat(String value, int count) {
        StringBuilder output = new StringBuilder();
        for (int index = 0; index < count; index++) {
            output.append(value);
        }
        return output.toString();
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
