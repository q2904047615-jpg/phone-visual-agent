package com.visualagent.companionime.transport;

import com.visualagent.companionime.protocol.CommandEnvelope;
import com.visualagent.companionime.protocol.CommandFixtures;
import com.visualagent.companionime.replay.ReplayGuard;
import com.visualagent.companionime.replay.ReplayStore;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public final class ReplayProtectedExecutionTest {
    @Test
    public void reservesBeforeEditorCallAndNeverRepeatsSameOrMutatedAction() throws Exception {
        MemoryStore store = new MemoryStore();
        ReplayGuard guard = new ReplayGuard(store);
        ReplayProtectedExecution protectedExecution = new ReplayProtectedExecution();
        AtomicInteger physicalCalls = new AtomicInteger();
        CommandEnvelope first = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", "first"));

        assertEquals(
                ReplayProtectedExecution.Outcome.ACCEPTED,
                protectedExecution.execute(first, guard, command -> {
                    assertFalse("reservation must exist before InputConnection", store.values.isEmpty());
                    physicalCalls.incrementAndGet();
                    return true;
                }));
        assertEquals(
                ReplayProtectedExecution.Outcome.ACCEPTED,
                protectedExecution.execute(first, guard, command -> {
                    physicalCalls.incrementAndGet();
                    return true;
                }));

        CommandEnvelope mutatedSameAction = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", "second"));
        assertEquals(
                ReplayProtectedExecution.Outcome.UNKNOWN,
                protectedExecution.execute(mutatedSameAction, guard, command -> {
                    physicalCalls.incrementAndGet();
                    return true;
                }));
        assertEquals(1, physicalCalls.get());
    }

    @Test
    public void uncertainEditorResultRemainsUnknownAndIsNotRetried() throws Exception {
        ReplayGuard guard = new ReplayGuard(new MemoryStore());
        ReplayProtectedExecution protectedExecution = new ReplayProtectedExecution();
        AtomicInteger calls = new AtomicInteger();
        CommandEnvelope command = CommandFixtures.parse(
                CommandFixtures.envelope("commit_text", "text"));

        assertEquals(
                ReplayProtectedExecution.Outcome.UNKNOWN,
                protectedExecution.execute(command, guard, value -> {
                    calls.incrementAndGet();
                    return false;
                }));
        assertEquals(
                ReplayProtectedExecution.Outcome.UNKNOWN,
                protectedExecution.execute(command, guard, value -> {
                    calls.incrementAndGet();
                    return true;
                }));
        assertEquals(1, calls.get());
    }

    private static final class MemoryStore implements ReplayStore {
        private final Map<String, String> values = new HashMap<>();

        @Override
        public String read(String commandDigest) {
            return values.get(commandDigest);
        }

        @Override
        public boolean write(String commandDigest, String state) {
            values.put(commandDigest, state);
            return true;
        }
    }
}
