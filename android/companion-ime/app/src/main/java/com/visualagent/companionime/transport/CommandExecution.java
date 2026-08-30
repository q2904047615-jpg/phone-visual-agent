package com.visualagent.companionime.transport;

import com.visualagent.companionime.protocol.CommandEnvelope;

public interface CommandExecution {
    boolean execute(CommandEnvelope command);
}
