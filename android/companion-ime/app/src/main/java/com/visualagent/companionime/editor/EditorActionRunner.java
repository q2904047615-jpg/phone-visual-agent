package com.visualagent.companionime.editor;

import com.visualagent.companionime.protocol.CommandEnvelope;

public final class EditorActionRunner {
    public boolean execute(CommandEnvelope command, EditorConnection connection) {
        if (!connection.finishComposingText()) {
            return false;
        }
        if (command.operation() == CommandEnvelope.Operation.COMMIT_TEXT) {
            return connection.commitText(command.text());
        }
        return connection.clearText();
    }
}
