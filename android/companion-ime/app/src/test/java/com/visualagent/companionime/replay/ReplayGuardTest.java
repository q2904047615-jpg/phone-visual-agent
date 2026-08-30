package com.visualagent.companionime.replay;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class ReplayGuardTest {
    @Test
    public void duplicateReservedCommandIsUnknownAndNeverReservedAgain() {
        MemoryStore store = new MemoryStore();
        ReplayGuard guard = new ReplayGuard(store);

        String digest = repeat('a');
        assertEquals(ReplayGuard.Reservation.NEW, guard.reserve("action-1", digest));
        assertEquals(
                ReplayGuard.Reservation.DUPLICATE_UNKNOWN,
                guard.reserve("action-1", digest));
        assertTrue(guard.markAccepted("action-1", digest));
        assertEquals(
                ReplayGuard.Reservation.DUPLICATE_ACCEPTED,
                new ReplayGuard(store).reserve("action-1", digest));
        assertEquals(
                ReplayGuard.Reservation.DUPLICATE_UNKNOWN,
                guard.reserve("action-1", repeat('b')));
    }

    @Test
    public void durableReservedStateRemainsUnknownAfterProcessRestart() {
        MemoryStore store = new MemoryStore();
        String digest = repeat('a');
        assertEquals(ReplayGuard.Reservation.NEW,
                new ReplayGuard(store).reserve("action-3", digest));

        assertEquals(ReplayGuard.Reservation.DUPLICATE_UNKNOWN,
                new ReplayGuard(store).reserve("action-3", digest));
    }

    @Test
    public void failedDurableReservationFailsClosedForProcessLifetime() {
        MemoryStore store = new MemoryStore();
        store.failWrites = true;
        ReplayGuard guard = new ReplayGuard(store);

        assertEquals(
                ReplayGuard.Reservation.DUPLICATE_UNKNOWN,
                guard.reserve("action-2", repeat('a')));
        store.failWrites = false;
        assertEquals(
                ReplayGuard.Reservation.DUPLICATE_UNKNOWN,
                guard.reserve("action-2", repeat('a')));
        assertFalse(guard.markAccepted("action-2", repeat('a')));
    }

    private static String repeat(char value) {
        char[] output = new char[64];
        java.util.Arrays.fill(output, value);
        return new String(output);
    }

    private static final class MemoryStore implements ReplayStore {
        private final Map<String, String> values = new HashMap<>();
        private boolean failWrites;

        @Override
        public String read(String commandDigest) {
            return values.get(commandDigest);
        }

        @Override
        public boolean write(String commandDigest, String state) {
            if (failWrites) {
                return false;
            }
            values.put(commandDigest, state);
            return true;
        }
    }
}
