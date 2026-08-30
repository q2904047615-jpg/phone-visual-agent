package com.visualagent.companionime.protocol;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;

public final class LengthPrefixedJsonCodec {
    private LengthPrefixedJsonCodec() {
    }

    public static JSONObject read(InputStream input) throws IOException, ProtocolException {
        DataInputStream data = new DataInputStream(input);
        int length = data.readInt();
        if (length <= 0 || length > ProtocolConstants.MAX_FRAME_BYTES) {
            throw new ProtocolException("frame length is outside the allowed range");
        }
        byte[] encoded = new byte[length];
        data.readFully(encoded);
        String json;
        try {
            json = StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(encoded))
                    .toString();
        } catch (CharacterCodingException error) {
            throw new ProtocolException("frame is not valid UTF-8", error);
        }
        try {
            JSONTokener tokener = new JSONTokener(json);
            Object parsed = tokener.nextValue();
            if (!(parsed instanceof JSONObject) || tokener.nextClean() != 0) {
                throw new ProtocolException("frame must contain exactly one JSON object");
            }
            JSONObject object = (JSONObject) parsed;
            if (object.length() == 0) {
                throw new ProtocolException("empty JSON objects are not accepted");
            }
            return object;
        } catch (JSONException error) {
            throw new ProtocolException("frame is not a JSON object", error);
        }
    }

    public static void write(OutputStream output, JSONObject object) throws IOException,
            ProtocolException {
        byte[] encoded = object.toString().getBytes(StandardCharsets.UTF_8);
        if (encoded.length <= 0 || encoded.length > ProtocolConstants.MAX_FRAME_BYTES) {
            throw new ProtocolException("frame length is outside the allowed range");
        }
        DataOutputStream data = new DataOutputStream(output);
        data.writeInt(encoded.length);
        data.write(encoded);
        data.flush();
    }
}
