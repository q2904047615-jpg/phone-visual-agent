package com.visualagent.companionime.editor;

import com.visualagent.companionime.protocol.CommandEnvelope;
import com.visualagent.companionime.protocol.CommandFixtures;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public final class EditorActionRunnerTest {
    @Test
    public void commitFinishesCompositionBeforeSingleCommit() throws Exception {
        FakeConnection connection = new FakeConnection();
        CommandEnvelope command = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", "你好🙂\nA?é"));

        assertTrue(new EditorActionRunner().execute(command, connection));
        assertEquals(Arrays.asList("finish", "commit:你好🙂\nA?é"), connection.events);
    }

    @Test
    public void clearFinishesCompositionAndNeverCommitsNewText() throws Exception {
        FakeConnection connection = new FakeConnection();
        CommandEnvelope command = CommandFixtures.parse(
                CommandFixtures.envelope("clear_text", null));

        assertTrue(new EditorActionRunner().execute(command, connection));
        assertEquals(Arrays.asList("finish", "clear"), connection.events);
    }

    @Test
    public void failedCompositionFinishDoesNotCommitOrClear() throws Exception {
        FakeConnection connection = new FakeConnection();
        connection.finishAccepted = false;
        CommandEnvelope commit = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", "text"));
        CommandEnvelope clear = CommandFixtures.parse(
                CommandFixtures.envelope("clear_text", null));

        org.junit.Assert.assertFalse(new EditorActionRunner().execute(commit, connection));
        org.junit.Assert.assertFalse(new EditorActionRunner().execute(clear, connection));
        assertEquals(Arrays.asList("finish", "finish"), connection.events);
    }

    private static final class FakeConnection implements EditorConnection {
        private final List<String> events = new ArrayList<>();
        private boolean finishAccepted = true;

        @Override
        public boolean finishComposingText() {
            events.add("finish");
            return finishAccepted;
        }

        @Override
        public boolean commitText(String text) {
            events.add("commit:" + text);
            return true;
        }

        @Override
        public boolean clearText() {
            events.add("clear");
            return true;
        }
    }
}
