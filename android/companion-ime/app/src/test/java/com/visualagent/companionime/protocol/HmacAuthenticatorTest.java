package com.visualagent.companionime.protocol;

import org.json.JSONObject;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class HmacAuthenticatorTest {
    @Test
    public void recursiveLexicalCanonicalizationIgnoresInsertionOrder() throws Exception {
        JSONObject firstNested = new JSONObject();
        firstNested.put("text", "你好🙂");
        JSONObject first = new JSONObject();
        first.put("z", "last");
        first.put("nested", firstNested);
        first.put("a", 1);

        JSONObject secondNested = new JSONObject();
        secondNested.put("text", "你好🙂");
        JSONObject second = new JSONObject();
        second.put("a", 1);
        second.put("nested", secondNested);
        second.put("z", "last");

        assertEquals(
                "{\"a\":1,\"nested\":{\"text\":\"你好🙂\"},\"z\":\"last\"}",
                CanonicalJson.serialize(first));
        assertEquals(
                HmacAuthenticator.sign(first, CommandFixtures.KEY),
                HmacAuthenticator.sign(second, CommandFixtures.KEY));
        assertEquals(
                "5613584748e6b0877954348c435959856ed9ffe7cd28afe2674ef444c97579d1",
                HmacAuthenticator.sign(first, CommandFixtures.KEY));
    }

    @Test
    public void epochNumberMatchesPythonPlainFloatEncoding() throws Exception {
        JSONObject value = new JSONObject();
        value.put("issued_at_epoch", 1_800_000_000.5);

        assertEquals(
                "{\"issued_at_epoch\":1800000000.5}",
                CanonicalJson.serialize(value));
    }

    @Test
    public void integralEpochAndScientificValuesMatchPythonFloatEncoding() throws Exception {
        JSONObject value = new JSONObject();
        value.put("integral_epoch", 1_800_000_000.0);
        value.put("small", 0.00001);
        value.put("large", 10_000_000_000_000_000.0);

        assertEquals(
                "{\"integral_epoch\":1800000000.0,\"large\":1e+16,\"small\":1e-05}",
                CanonicalJson.serialize(value));
    }

    @Test
    public void parsedPythonJsonRetainsCanonicalIntegralFloat() throws Exception {
        byte[] payload = "{\"issued_at_epoch\":1800000000.0}"
                .getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream framed = new ByteArrayOutputStream();
        DataOutputStream output = new DataOutputStream(framed);
        output.writeInt(payload.length);
        output.write(payload);

        JSONObject parsed = LengthPrefixedJsonCodec.read(
                new ByteArrayInputStream(framed.toByteArray()));
        assertEquals(
                "{\"issued_at_epoch\":1800000000.0}",
                CanonicalJson.serialize(parsed));
    }

    @Test
    public void signatureIsLowercaseHexAndRejectsChangedField() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("commit_text", "abc");
        JSONObject command = CommandFixtures.command(envelope);
        String signature = envelope.getString("signature");
        assertEquals(64, signature.length());

        command.put("operation", "clear_text");
        assertThrows(
                ProtocolException.class,
                () -> HmacAuthenticator.verify(command, signature, CommandFixtures.KEY));
    }
}
