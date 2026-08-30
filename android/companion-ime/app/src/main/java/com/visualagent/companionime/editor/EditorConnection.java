package com.visualagent.companionime.editor;

public interface EditorConnection {
    boolean finishComposingText();

    boolean commitText(String text);

    boolean clearText();
}
