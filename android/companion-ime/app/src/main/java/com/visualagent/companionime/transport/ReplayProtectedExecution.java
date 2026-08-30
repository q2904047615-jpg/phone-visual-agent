package com.visualagent.companionime.transport;

import com.visualagent.companionime.protocol.CommandEnvelope;
import com.visualagent.companionime.replay.ReplayGuard;

public final class ReplayProtectedExecution {
    public enum Outcome {
        ACCEPTED,
        UNKNOWN
    }

    public Outcome execute(
            CommandEnvelope command,
            ReplayGuard replayGuard,
            CommandExecution execution) {
        ReplayGuard.Reservation reservation = replayGuard.reserve(
                command.replayIdentity(), command.commandDigest());
        if (reservation == ReplayGuard.Reservation.DUPLICATE_ACCEPTED) {
            return Outcome.ACCEPTED;
        }
        if (reservation == ReplayGuard.Reservation.DUPLICATE_UNKNOWN) {
            return Outcome.UNKNOWN;
        }
        if (!execution.execute(command)) {
            return Outcome.UNKNOWN;
        }
        return replayGuard.markAccepted(command.replayIdentity(), command.commandDigest())
                ? Outcome.ACCEPTED
                : Outcome.UNKNOWN;
    }
}
