package com.visualagent.companionime.protocol;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.regex.Pattern;

public final class StrictJson {
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    private static final Pattern NONCE = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{15,127}");
    private static final Pattern SIGNATURE = Pattern.compile("[0-9a-f]{64}");
    private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,255}");

    private StrictJson() {
    }

    public static void requireExactKeys(JSONObject object, Set<String> expected)
            throws ProtocolException {
        Set<String> actual = new HashSet<>();
        Iterator<String> iterator = object.keys();
        while (iterator.hasNext()) {
            actual.add(iterator.next());
        }
        if (!actual.equals(expected)) {
            throw new ProtocolException("JSON keys do not match the protocol contract");
        }
    }

    public static String requireString(JSONObject object, String key, int maxLength)
            throws ProtocolException {
        try {
            Object raw = object.get(key);
            if (!(raw instanceof String)) {
                throw new ProtocolException(key + " must be a string");
            }
            String value = (String) raw;
            if (value.isEmpty() || value.length() > maxLength) {
                throw new ProtocolException(key + " has an invalid length");
            }
            return value;
        } catch (JSONException error) {
            throw new ProtocolException("missing key: " + key, error);
        }
    }

    public static String requireType(JSONObject object, String expected)
            throws ProtocolException {
        String actual = requireString(object, "type", 64);
        if (!expected.equals(actual)) {
            throw new ProtocolException("unexpected message type");
        }
        return actual;
    }

    public static void requireProtocolVersion(JSONObject object) throws ProtocolException {
        String value = requireString(object, "protocol_version", 64);
        if (!ProtocolConstants.VERSION.equals(value)) {
            throw new ProtocolException("unsupported protocol version");
        }
    }

    public static long requireLong(JSONObject object, String key, long minimum)
            throws ProtocolException {
        try {
            Object raw = object.get(key);
            if (!(raw instanceof Byte || raw instanceof Short || raw instanceof Integer
                    || raw instanceof Long)) {
                throw new ProtocolException(key + " must be an integer");
            }
            long value = ((Number) raw).longValue();
            if (value < minimum) {
                throw new ProtocolException(key + " is below the allowed minimum");
            }
            return value;
        } catch (JSONException error) {
            throw new ProtocolException("missing key: " + key, error);
        }
    }

    public static String requireSha256(JSONObject object, String key) throws ProtocolException {
        String value = requireString(object, key, 64);
        if (!SHA256.matcher(value).matches()) {
            throw new ProtocolException(key + " must be lowercase SHA-256 hex");
        }
        return value;
    }

    public static String requireNonce(JSONObject object, String key) throws ProtocolException {
        String value = requireString(object, key, 128);
        if (!NONCE.matcher(value).matches()) {
            throw new ProtocolException(key + " is not a valid nonce");
        }
        return value;
    }

    public static void requireSignature(String value) throws ProtocolException {
        if (value == null || !SIGNATURE.matcher(value).matches()) {
            throw new ProtocolException("signature is not a valid HMAC value");
        }
    }

    public static String requireIdentifier(JSONObject object, String key) throws ProtocolException {
        String value = requireString(object, key, 256);
        if (!IDENTIFIER.matcher(value).matches()) {
            throw new ProtocolException(key + " is not a valid identifier");
        }
        return value;
    }

    public static double requireFiniteDouble(JSONObject object, String key)
            throws ProtocolException {
        try {
            Object raw = object.get(key);
            if (!(raw instanceof Number)) {
                throw new ProtocolException(key + " must be numeric");
            }
            double value = ((Number) raw).doubleValue();
            if (!Double.isFinite(value)) {
                throw new ProtocolException(key + " must be finite");
            }
            return value;
        } catch (JSONException error) {
            throw new ProtocolException("missing key: " + key, error);
        }
    }
}
