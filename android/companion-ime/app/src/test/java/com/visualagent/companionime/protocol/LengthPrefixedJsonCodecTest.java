package com.visualagent.companionime.protocol;

import org.json.JSONObject;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class LengthPrefixedJsonCodecTest {
    @Test
    public void roundTripsUnicodeJsonObject() throws Exception {
        JSONObject original = new JSONObject();
        original.put("value", "你好🙂");
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        LengthPrefixedJsonCodec.write(bytes, original);
        JSONObject decoded = LengthPrefixedJsonCodec.read(
                new ByteArrayInputStream(bytes.toByteArray()));

        assertEquals("你好🙂", decoded.getString("value"));
    }

    @Test
    public void rejectsOversizedLengthBeforeAllocatingPayload() throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream data = new DataOutputStream(bytes);
        data.writeInt(ProtocolConstants.MAX_FRAME_BYTES + 1);

        assertThrows(
                ProtocolException.class,
                () -> LengthPrefixedJsonCodec.read(
                        new ByteArrayInputStream(bytes.toByteArray())));
    }

    @Test
    public void rejectsTrailingContentAfterJsonObject() throws Exception {
        byte[] payload = "{\"a\":1} trailing".getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream data = new DataOutputStream(bytes);
        data.writeInt(payload.length);
        data.write(payload);

        assertThrows(
                ProtocolException.class,
                () -> LengthPrefixedJsonCodec.read(
                        new ByteArrayInputStream(bytes.toByteArray())));
    }

    @Test
    public void rejectsMalformedUtf8() throws Exception {
        byte[] payload = new byte[]{'{', '"', 'a', '"', ':', '"', (byte) 0xc3, '"', '}'};
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream data = new DataOutputStream(bytes);
        data.writeInt(payload.length);
        data.write(payload);

        assertThrows(
                ProtocolException.class,
                () -> LengthPrefixedJsonCodec.read(
                        new ByteArrayInputStream(bytes.toByteArray())));
    }
}
