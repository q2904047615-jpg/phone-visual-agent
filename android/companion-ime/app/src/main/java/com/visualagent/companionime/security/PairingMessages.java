package com.visualagent.companionime.security;

import com.visualagent.companionime.protocol.ProtocolConstants;
import com.visualagent.companionime.protocol.ProtocolException;
import com.visualagent.companionime.protocol.HmacAuthenticator;
import com.visualagent.companionime.protocol.StrictJson;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

final class PairingMessages {
    private static final Pattern IDENTIFIER =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,255}");
    private static final Set<String> REQUEST_KEYS = exact(
            "protocol_version", "type", "installation_id", "client_nonce",
            "one_time_token");
    private static final Set<String> RESPONSE_KEYS = exact(
            "protocol_version", "type", "installation_id", "client_nonce", "pairing_id",
            "device_id", "shared_key");
    private static final Set<String> CONFIRM_KEYS = exact(
            "protocol_version", "type", "installation_id", "client_nonce", "pairing_id",
            "device_id");
    private static final Set<String> CONFIRM_ACK_KEYS = exact(
            "protocol_version", "type", "installation_id", "client_nonce", "pairing_id",
            "device_id", "status");
    private static final Set<String> CONFIRM_ENVELOPE_KEYS = exact("confirm", "signature");
    private static final Set<String> CONFIRM_ACK_ENVELOPE_KEYS = exact("confirm_ack", "signature");
    private static final Set<String> COMMIT_ENVELOPE_KEYS = exact("commit", "signature");
    private static final Set<String> COMMIT_ACK_ENVELOPE_KEYS = exact("commit_ack", "signature");

    private PairingMessages() {
    }

    static JSONObject pairRequest(
            String installationId,
            String clientNonce,
            String oneTimeToken) throws ProtocolException {
        if (installationId == null || !IDENTIFIER.matcher(installationId).matches()) {
            throw new ProtocolException("installation identity is invalid");
        }
        if (clientNonce == null
                || !clientNonce.matches("[A-Za-z0-9][A-Za-z0-9._:-]{15,127}")) {
            throw new ProtocolException("pairing nonce is invalid");
        }
        if (oneTimeToken == null || oneTimeToken.isEmpty() || oneTimeToken.length() > 512) {
            throw new ProtocolException("one-time token is invalid");
        }
        JSONObject request = new JSONObject();
        try {
            request.put("protocol_version", ProtocolConstants.VERSION);
            request.put("type", "pair_request");
            request.put("installation_id", installationId);
            request.put("client_nonce", clientNonce);
            request.put("one_time_token", oneTimeToken);
        } catch (JSONException impossible) {
            throw new ProtocolException("unable to create pairing request", impossible);
        }
        StrictJson.requireExactKeys(request, REQUEST_KEYS);
        return request;
    }

    static PairingRecord pairResponse(
            JSONObject response,
            String host,
            int port,
            String certificateSha256,
            String expectedInstallationId,
            String expectedClientNonce) throws ProtocolException {
        StrictJson.requireExactKeys(response, RESPONSE_KEYS);
        StrictJson.requireProtocolVersion(response);
        StrictJson.requireType(response, "pair_response");
        String returnedInstallation = StrictJson.requireIdentifier(
                response, "installation_id");
        String returnedNonce = StrictJson.requireNonce(response, "client_nonce");
        if (!expectedInstallationId.equals(returnedInstallation)
                || !expectedClientNonce.equals(returnedNonce)) {
            throw new ProtocolException("pairing response is not bound to this request");
        }
        String pairingId = StrictJson.requireIdentifier(response, "pairing_id");
        String deviceId = StrictJson.requireIdentifier(response, "device_id");
        String encodedKey = StrictJson.requireString(response, "shared_key", 128);
        byte[] sharedKey;
        try {
            sharedKey = Base64.getDecoder().decode(encodedKey);
        } catch (IllegalArgumentException error) {
            throw new ProtocolException("shared key is not valid base64", error);
        }
        try {
            if (sharedKey.length != 32) {
                throw new ProtocolException("shared key must contain 32 bytes");
            }
            return new PairingRecord(
                    host, port, certificateSha256, expectedInstallationId, pairingId, deviceId,
                    sharedKey);
        } finally {
            Arrays.fill(sharedKey, (byte) 0);
        }
    }

    static JSONObject pairConfirm(PairingRecord record, String clientNonce)
            throws ProtocolException {
        JSONObject confirm = new JSONObject();
        byte[] sharedKey = record.sharedKey();
        try {
            confirm.put("protocol_version", ProtocolConstants.VERSION);
            confirm.put("type", "pair_confirm");
            confirm.put("installation_id", record.installationId());
            confirm.put("client_nonce", clientNonce);
            confirm.put("pairing_id", record.pairingId());
            confirm.put("device_id", record.deviceId());
            StrictJson.requireExactKeys(confirm, CONFIRM_KEYS);
            JSONObject envelope = new JSONObject();
            envelope.put("confirm", confirm);
            envelope.put("signature", HmacAuthenticator.sign(confirm, sharedKey));
            StrictJson.requireExactKeys(envelope, CONFIRM_ENVELOPE_KEYS);
            return envelope;
        } catch (JSONException error) {
            throw new ProtocolException("unable to create pairing confirmation", error);
        } finally {
            Arrays.fill(sharedKey, (byte) 0);
        }
    }

    static void verifyPairConfirmAck(
            JSONObject envelope,
            PairingRecord record,
            String expectedClientNonce) throws ProtocolException {
        StrictJson.requireExactKeys(envelope, CONFIRM_ACK_ENVELOPE_KEYS);
        JSONObject ack;
        try {
            Object raw = envelope.get("confirm_ack");
            if (!(raw instanceof JSONObject)) {
                throw new ProtocolException("pairing confirmation acknowledgement is invalid");
            }
            ack = (JSONObject) raw;
        } catch (JSONException error) {
            throw new ProtocolException("pairing confirmation acknowledgement is missing", error);
        }
        StrictJson.requireExactKeys(ack, CONFIRM_ACK_KEYS);
        byte[] sharedKey = record.sharedKey();
        try {
            HmacAuthenticator.verify(
                    ack, StrictJson.requireString(envelope, "signature", 64), sharedKey);
        } finally {
            Arrays.fill(sharedKey, (byte) 0);
        }
        StrictJson.requireProtocolVersion(ack);
        StrictJson.requireType(ack, "pair_confirm_ack");
        String status = StrictJson.requireString(ack, "status", 32);
        if (!"accepted".equals(status)
                || !record.installationId().equals(StrictJson.requireIdentifier(ack, "installation_id"))
                || !expectedClientNonce.equals(StrictJson.requireNonce(ack, "client_nonce"))
                || !record.pairingId().equals(StrictJson.requireIdentifier(ack, "pairing_id"))
                || !record.deviceId().equals(StrictJson.requireIdentifier(ack, "device_id"))) {
            throw new ProtocolException("pairing confirmation acknowledgement is not bound to this request");
        }
    }

    static JSONObject pairCommit(PairingRecord record, String clientNonce)
            throws ProtocolException {
        JSONObject commit = new JSONObject();
        byte[] sharedKey = record.sharedKey();
        try {
            commit.put("protocol_version", ProtocolConstants.VERSION);
            commit.put("type", "pair_commit");
            commit.put("installation_id", record.installationId());
            commit.put("client_nonce", clientNonce);
            commit.put("pairing_id", record.pairingId());
            commit.put("device_id", record.deviceId());
            StrictJson.requireExactKeys(commit, CONFIRM_KEYS);
            JSONObject envelope = new JSONObject();
            envelope.put("commit", commit);
            envelope.put("signature", HmacAuthenticator.sign(commit, sharedKey));
            StrictJson.requireExactKeys(envelope, COMMIT_ENVELOPE_KEYS);
            return envelope;
        } catch (JSONException error) {
            throw new ProtocolException("unable to create pairing commit", error);
        } finally {
            Arrays.fill(sharedKey, (byte) 0);
        }
    }

    static void verifyPairCommitAck(
            JSONObject envelope,
            PairingRecord record,
            String expectedClientNonce) throws ProtocolException {
        StrictJson.requireExactKeys(envelope, COMMIT_ACK_ENVELOPE_KEYS);
        JSONObject ack;
        try {
            Object raw = envelope.get("commit_ack");
            if (!(raw instanceof JSONObject)) {
                throw new ProtocolException("pairing commit acknowledgement is invalid");
            }
            ack = (JSONObject) raw;
        } catch (JSONException error) {
            throw new ProtocolException("pairing commit acknowledgement is missing", error);
        }
        StrictJson.requireExactKeys(ack, CONFIRM_ACK_KEYS);
        byte[] sharedKey = record.sharedKey();
        try {
            HmacAuthenticator.verify(
                    ack, StrictJson.requireString(envelope, "signature", 64), sharedKey);
        } finally {
            Arrays.fill(sharedKey, (byte) 0);
        }
        StrictJson.requireProtocolVersion(ack);
        StrictJson.requireType(ack, "pair_commit_ack");
        String status = StrictJson.requireString(ack, "status", 32);
        if (!"accepted".equals(status)
                || !record.installationId().equals(StrictJson.requireIdentifier(ack, "installation_id"))
                || !expectedClientNonce.equals(StrictJson.requireNonce(ack, "client_nonce"))
                || !record.pairingId().equals(StrictJson.requireIdentifier(ack, "pairing_id"))
                || !record.deviceId().equals(StrictJson.requireIdentifier(ack, "device_id"))) {
            throw new ProtocolException("pairing commit acknowledgement is not bound to this request");
        }
    }

    private static Set<String> exact(String... keys) {
        return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(keys)));
    }
}
