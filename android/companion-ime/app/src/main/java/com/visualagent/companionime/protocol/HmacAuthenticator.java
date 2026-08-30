package com.visualagent.companionime.protocol;

import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

public final class HmacAuthenticator {
    private HmacAuthenticator() {
    }

    public static String sign(JSONObject message, byte[] sharedKey) throws ProtocolException {
        if (sharedKey == null || sharedKey.length != 32) {
            throw new ProtocolException("shared key must contain 32 bytes");
        }
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(sharedKey, "HmacSHA256"));
            byte[] value = mac.doFinal(
                    CanonicalJson.serialize(message).getBytes(StandardCharsets.UTF_8));
            return Hashing.hex(value);
        } catch (GeneralSecurityException error) {
            throw new ProtocolException("HMAC-SHA256 is unavailable", error);
        }
    }

    public static void verify(JSONObject message, String supplied, byte[] sharedKey)
            throws ProtocolException {
        StrictJson.requireSignature(supplied);
        String expected = sign(message, sharedKey);
        if (!MessageDigest.isEqual(
                supplied.getBytes(StandardCharsets.US_ASCII),
                expected.getBytes(StandardCharsets.US_ASCII))) {
            throw new ProtocolException("message authentication failed");
        }
    }
}
