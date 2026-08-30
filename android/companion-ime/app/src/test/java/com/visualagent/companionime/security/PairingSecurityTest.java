package com.visualagent.companionime.security;

import com.visualagent.companionime.protocol.ProtocolException;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class PairingSecurityTest {
    @Test
    public void normalizesManuallyEnteredCertificatePin() throws Exception {
        String entered = "AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:"
                + "AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA";

        assertEquals(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                PinnedTlsSocket.normalizeFingerprint(entered));
        assertThrows(
                ProtocolException.class,
                () -> PinnedTlsSocket.normalizeFingerprint("not-a-fingerprint"));
    }

    @Test
    public void pairingRecordDoesNotExposeMutableKeyAndCanBeDestroyed() {
        byte[] original = new byte[32];
        original[0] = 7;
        PairingRecord record = new PairingRecord(
                "host", 443, repeat("aa", 32), "installation", "pairing", "device", original);
        byte[] exposed = record.sharedKey();
        exposed[0] = 9;
        assertArrayEquals(original, record.sharedKey());

        record.destroy();
        assertArrayEquals(new byte[32], record.sharedKey());
    }

    private static String repeat(String value, int count) {
        StringBuilder output = new StringBuilder();
        for (int index = 0; index < count; index++) {
            output.append(value);
        }
        return output.toString();
    }
}
