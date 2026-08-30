package com.visualagent.companionime.security;

import com.visualagent.companionime.protocol.LengthPrefixedJsonCodec;
import com.visualagent.companionime.protocol.Nonce;
import com.visualagent.companionime.protocol.ProtocolException;

import org.json.JSONObject;

import java.io.IOException;

import javax.net.ssl.SSLSocket;

public final class PairingClient {
    public PairingRecord pair(
            String host,
            int port,
            String certificateSha256,
            String installationId,
            String oneTimeToken,
            EncryptedPairingStore pairingStore)
            throws IOException, ProtocolException, PairingStoreException {
        if (host == null || host.trim().isEmpty() || port < 1 || port > 65535) {
            throw new ProtocolException("controller address is invalid");
        }
        String normalizedPin = PinnedTlsSocket.normalizeFingerprint(certificateSha256);
        String clientNonce = Nonce.create();
        JSONObject request = PairingMessages.pairRequest(
                installationId, clientNonce, oneTimeToken);

        try (SSLSocket socket = PinnedTlsSocket.connect(host.trim(), port, normalizedPin)) {
            LengthPrefixedJsonCodec.write(socket.getOutputStream(), request);
            JSONObject response = LengthPrefixedJsonCodec.read(socket.getInputStream());
            PairingRecord record = PairingMessages.pairResponse(
                    response,
                    host.trim(),
                    port,
                    normalizedPin,
                    installationId,
                    clientNonce);
            boolean completed = false;
            try {
                // Keep the currently active pairing intact until the PC has authenticated and
                // acknowledged the new key. A dropped repair connection cannot overwrite it.
                pairingStore.savePending(record);
                LengthPrefixedJsonCodec.write(
                        socket.getOutputStream(), PairingMessages.pairConfirm(record, clientNonce));
                PairingMessages.verifyPairConfirmAck(
                        LengthPrefixedJsonCodec.read(socket.getInputStream()), record, clientNonce);
                pairingStore.promotePending();
                LengthPrefixedJsonCodec.write(
                        socket.getOutputStream(), PairingMessages.pairCommit(record, clientNonce));
                PairingMessages.verifyPairCommitAck(
                        LengthPrefixedJsonCodec.read(socket.getInputStream()), record, clientNonce);
                completed = true;
                return record;
            } finally {
                if (!completed) {
                    record.destroy();
                }
            }
        }
    }
}
