package com.visualagent.companionime.replay;

import com.visualagent.companionime.protocol.Hashing;

import java.util.HashSet;
import java.util.Set;

public final class ReplayGuard {
    public enum Reservation {
        NEW,
        DUPLICATE_ACCEPTED,
        DUPLICATE_UNKNOWN
    }

    private static final String RESERVED = "reserved:";
    private static final String ACCEPTED = "accepted:";

    private final ReplayStore store;
    private final Set<String> processUnknown = new HashSet<>();

    public ReplayGuard(ReplayStore store) {
        this.store = store;
    }

    public synchronized Reservation reserve(String actionIdentity, String commandDigest) {
        String identityDigest = Hashing.sha256Hex(actionIdentity);
        if (processUnknown.contains(identityDigest)) {
            return Reservation.DUPLICATE_UNKNOWN;
        }
        String existing = store.read(identityDigest);
        if ((ACCEPTED + commandDigest).equals(existing)) {
            return Reservation.DUPLICATE_ACCEPTED;
        }
        if (existing != null) {
            return Reservation.DUPLICATE_UNKNOWN;
        }
        if (!store.write(identityDigest, RESERVED + commandDigest)) {
            processUnknown.add(identityDigest);
            return Reservation.DUPLICATE_UNKNOWN;
        }
        return Reservation.NEW;
    }

    public synchronized boolean markAccepted(String actionIdentity, String commandDigest) {
        String identityDigest = Hashing.sha256Hex(actionIdentity);
        if (!(RESERVED + commandDigest).equals(store.read(identityDigest))) {
            processUnknown.add(identityDigest);
            return false;
        }
        boolean stored = store.write(identityDigest, ACCEPTED + commandDigest);
        if (!stored) {
            processUnknown.add(identityDigest);
        }
        return stored;
    }
}
