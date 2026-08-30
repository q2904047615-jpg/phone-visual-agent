package com.visualagent.companionime.protocol;

import org.json.JSONObject;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

public final class CommandEnvelopeTest {
    @Test
    public void acceptsUnicodeMultilineCommitBoundToCurrentEditor() throws Exception {
        String text = "你好🙂\nA?é";
        JSONObject envelope = CommandFixtures.envelope("commit_text", text);
        CommandEnvelope command = CommandFixtures.parse(envelope);

        assertEquals(CommandEnvelope.Operation.COMMIT_TEXT, command.operation());
        assertEquals(text, command.text());
        assertEquals(CommandFixtures.EDITOR, command.editorSessionId());
        assertEquals(
                "33458ae1458ac13c6633495939fbb09e571d630c906d1a68e7d4fdb4ede057b3",
                envelope.getString("signature"));
        assertEquals(
                "c2522a351ac276f56dbabd7c793fb00ff72abbc43f2a9e2f609bee1246b3f77d",
                command.commandDigest());
    }

    @Test
    public void clearUsesJsonNullAndIndependentOperation() throws Exception {
        CommandEnvelope command = CommandFixtures.parse(
                CommandFixtures.envelope("clear_text", null));

        assertEquals(CommandEnvelope.Operation.CLEAR_TEXT, command.operation());
        assertNull(command.text());
    }

    @Test
    public void rejectsUnknownNestedCommandKey() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("commit_text", "abc");
        CommandFixtures.command(envelope).put("fallback", "mechanical_keyboard");
        CommandFixtures.resign(envelope);

        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(envelope));
    }

    @Test
    public void rejectsUnknownScopeKeyEvenWhenAuthenticated() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("commit_text", "abc");
        CommandFixtures.scope(envelope).put("target_hint", "must-not-be-present");
        CommandFixtures.resign(envelope);

        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(envelope));
    }

    @Test
    public void rejectsTextOnClearEvenWhenAuthenticated() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("clear_text", null);
        CommandFixtures.command(envelope).put("text", "must-not-be-present");
        CommandFixtures.resign(envelope);

        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(envelope));
    }

    @Test
    public void rejectsTamperedText() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("commit_text", "original");
        CommandFixtures.command(envelope).put("text", "tampered");

        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(envelope));
    }

    @Test
    public void rejectsWrongEditorSession() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("commit_text", "abc");
        CommandFixtures.scope(envelope).put("editor_session_id", "stale-editor");
        CommandFixtures.resign(envelope);

        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(envelope));
    }

    @Test
    public void rejectsExpiredCommand() throws Exception {
        JSONObject envelope = CommandFixtures.envelope("commit_text", "abc");
        CommandFixtures.scope(envelope).put("expires_at_epoch", CommandFixtures.NOW - 0.1);
        CommandFixtures.resign(envelope);

        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(envelope));
    }

    @Test
    public void acceptsLongUnicodeTextWithoutChangingNewlinesOrEmoji() throws Exception {
        String fragment = "长文本🙂,!?é\n";
        StringBuilder text = new StringBuilder();
        for (int index = 0; index < 2_000; index++) {
            text.append(fragment);
        }
        CommandEnvelope command = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", text.toString()));

        assertEquals(text.toString(), command.text());
    }

    @Test
    public void rejectsCarriageReturnAndNonEmptyClearExpectedDigest() throws Exception {
        JSONObject carriageReturn = CommandFixtures.envelope("commit_text", "a\rb");
        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(carriageReturn));

        JSONObject clear = CommandFixtures.envelope("clear_text", null);
        CommandFixtures.scope(clear).put("expected_text_digest", Hashing.sha256Hex("not-empty"));
        CommandFixtures.resign(clear);
        assertThrows(ProtocolException.class, () -> CommandFixtures.parse(clear));
    }
}
