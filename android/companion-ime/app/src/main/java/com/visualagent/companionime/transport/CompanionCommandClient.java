package com.visualagent.companionime.transport;

import com.visualagent.companionime.protocol.AuthenticatedMessages;
import com.visualagent.companionime.protocol.CommandEnvelope;
import com.visualagent.companionime.protocol.LengthPrefixedJsonCodec;
import com.visualagent.companionime.protocol.ProtocolException;
import com.visualagent.companionime.replay.ReplayGuard;
import com.visualagent.companionime.security.PairingRecord;
import com.visualagent.companionime.security.PinnedTlsSocket;

import org.json.JSONObject;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import javax.net.ssl.SSLSocket;

public final class CompanionCommandClient {
    private final AtomicReference<SSLSocket> activeSocket = new AtomicReference<>();
    private final ReplayProtectedExecution replayProtectedExecution =
            new ReplayProtectedExecution();

    public void processEditorSession(
            PairingRecord pairing,
            String editorSessionId,
            ReplayGuard replayGuard,
            CommandExecution execution,
            BooleanSupplier editorSessionActive) throws IOException, ProtocolException {
        byte[] sharedKey = pairing.sharedKey();
        SSLSocket socket = null;
        try {
            socket = PinnedTlsSocket.connect(
                    pairing.host(), pairing.port(), pairing.certificateSha256());
            activeSocket.set(socket);
            try (SSLSocket managedSocket = socket) {
                if (!editorSessionActive.getAsBoolean()) {
                    return;
                }
                double nowEpoch = System.currentTimeMillis() / 1000.0;
                JSONObject hello = AuthenticatedMessages.bridgeHello(
                        pairing.pairingId(),
                        pairing.deviceId(),
                        nowEpoch,
                        sharedKey);
                String helloNonce = AuthenticatedMessages.innerNonce(hello, "hello");
                LengthPrefixedJsonCodec.write(managedSocket.getOutputStream(), hello);
                AuthenticatedMessages.verifyBridgeHelloAck(
                        LengthPrefixedJsonCodec.read(managedSocket.getInputStream()),
                        pairing.pairingId(),
                        pairing.deviceId(),
                        helloNonce,
                        sharedKey);

                JSONObject ready = AuthenticatedMessages.editorReady(
                        pairing.pairingId(),
                        pairing.deviceId(),
                        editorSessionId,
                        System.currentTimeMillis() / 1000.0,
                        sharedKey);
                String readyNonce = AuthenticatedMessages.innerNonce(ready, "ready");
                LengthPrefixedJsonCodec.write(managedSocket.getOutputStream(), ready);
                AuthenticatedMessages.verifyEditorReadyAck(
                        LengthPrefixedJsonCodec.read(managedSocket.getInputStream()),
                        pairing.pairingId(),
                        pairing.deviceId(),
                        editorSessionId,
                        readyNonce,
                        sharedKey);

                managedSocket.setSoTimeout(0);
                while (true) {
                    JSONObject rawCommand = LengthPrefixedJsonCodec.read(
                            managedSocket.getInputStream());
                    CommandEnvelope command = CommandEnvelope.parseAndAuthenticate(
                            rawCommand,
                            sharedKey,
                            pairing.deviceId(),
                            editorSessionId,
                            System.currentTimeMillis() / 1000.0);

                    ReplayProtectedExecution.Outcome outcome = replayProtectedExecution.execute(
                            command, replayGuard, execution);
                    if (outcome == ReplayProtectedExecution.Outcome.UNKNOWN) {
                        return;
                    }
                    LengthPrefixedJsonCodec.write(
                            managedSocket.getOutputStream(),
                            AuthenticatedMessages.actionAck(
                                    command,
                                    "accepted",
                                    null,
                                    System.currentTimeMillis() / 1000.0,
                                    sharedKey));
                }
            }
        } finally {
            if (socket != null) {
                activeSocket.compareAndSet(socket, null);
            }
            Arrays.fill(sharedKey, (byte) 0);
        }
    }

    public void cancelActive() {
        SSLSocket socket = activeSocket.getAndSet(null);
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException ignored) {
                // Socket cancellation has no payload and requires no logging.
            }
        }
    }
}
