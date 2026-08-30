package com.visualagent.companionime.protocol;

import org.json.JSONException;
import org.json.JSONObject;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

final class CanonicalJson {
    private CanonicalJson() {
    }

    static String serialize(JSONObject object) throws ProtocolException {
        StringBuilder result = new StringBuilder();
        appendObject(result, object);
        return result.toString();
    }

    private static void appendObject(StringBuilder result, JSONObject object)
            throws ProtocolException {
        List<String> keys = new ArrayList<>(object.keySet());
        Collections.sort(keys);
        result.append('{');
        boolean first = true;
        for (String key : keys) {
            if (!first) {
                result.append(',');
            }
            first = false;
            result.append(quote(key)).append(':');
            try {
                appendValue(result, object.get(key));
            } catch (JSONException error) {
                throw new ProtocolException("unable to canonicalize JSON", error);
            }
        }
        result.append('}');
    }

    private static void appendValue(StringBuilder result, Object value) throws ProtocolException {
        if (value == null || value == JSONObject.NULL) {
            result.append("null");
        } else if (value instanceof String) {
            result.append(quote((String) value));
        } else if (value instanceof Byte || value instanceof Short
                || value instanceof Integer || value instanceof Long
                || value instanceof BigInteger) {
            result.append(value.toString());
        } else if (value instanceof Float || value instanceof Double
                || value instanceof BigDecimal) {
            result.append(pythonFloatingPoint((Number) value));
        } else if (value instanceof Boolean) {
            result.append(Boolean.TRUE.equals(value) ? "true" : "false");
        } else if (value instanceof JSONObject) {
            appendObject(result, (JSONObject) value);
        } else {
            throw new ProtocolException("authenticated JSON values must be scalar objects");
        }
    }

    private static String pythonFloatingPoint(Number value) throws ProtocolException {
        double number = value.doubleValue();
        if (!Double.isFinite(number)) {
            throw new ProtocolException("authenticated JSON numbers must be finite");
        }
        if (number == 0.0d) {
            return Double.doubleToRawLongBits(number) == Double.doubleToRawLongBits(-0.0d)
                    ? "-0.0"
                    : "0.0";
        }

        BigDecimal decimal = value instanceof BigDecimal
                ? (BigDecimal) value
                : BigDecimal.valueOf(number);
        decimal = decimal.stripTrailingZeros();
        int exponent = decimal.precision() - decimal.scale() - 1;
        if (exponent >= 16 || exponent < -4) {
            return scientific(decimal, exponent);
        }
        String plain = decimal.toPlainString();
        return decimal.scale() <= 0 ? plain + ".0" : plain;
    }

    private static String scientific(BigDecimal decimal, int exponent) {
        String digits = decimal.unscaledValue().abs().toString();
        StringBuilder output = new StringBuilder(digits.length() + 8);
        if (decimal.signum() < 0) {
            output.append('-');
        }
        output.append(digits.charAt(0));
        if (digits.length() > 1) {
            output.append('.').append(digits, 1, digits.length());
        }
        output.append('e').append(exponent >= 0 ? '+' : '-');
        int absoluteExponent = Math.abs(exponent);
        if (absoluteExponent < 10) {
            output.append('0');
        }
        return output.append(absoluteExponent).toString();
    }

    private static String quote(String value) {
        StringBuilder output = new StringBuilder(value.length() + 2);
        output.append('"');
        for (int index = 0; index < value.length(); index++) {
            char current = value.charAt(index);
            switch (current) {
                case '"':
                    output.append("\\\"");
                    break;
                case '\\':
                    output.append("\\\\");
                    break;
                case '\b':
                    output.append("\\b");
                    break;
                case '\f':
                    output.append("\\f");
                    break;
                case '\n':
                    output.append("\\n");
                    break;
                case '\r':
                    output.append("\\r");
                    break;
                case '\t':
                    output.append("\\t");
                    break;
                default:
                    if (current < 0x20) {
                        output.append(String.format(java.util.Locale.ROOT, "\\u%04x", (int) current));
                    } else {
                        output.append(current);
                    }
            }
        }
        return output.append('"').toString();
    }
}
