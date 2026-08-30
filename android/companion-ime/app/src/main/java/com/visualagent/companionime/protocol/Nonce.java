package com.visualagent.companionime.protocol;

import java.security.SecureRandom;
import java.util.Base64;

public final class Nonce {
    private static final SecureRandom RANDOM = new SecureRandom();

    private Nonce() {
    }

    public static String create() {
        while (true) {
            byte[] value = new byte[24];
            RANDOM.nextBytes(value);
            String encoded = Base64.getUrlEncoder().withoutPadding().encodeToString(value);
            if (Character.isLetterOrDigit(encoded.charAt(0))) {
                return encoded;
            }
        }
    }
}
