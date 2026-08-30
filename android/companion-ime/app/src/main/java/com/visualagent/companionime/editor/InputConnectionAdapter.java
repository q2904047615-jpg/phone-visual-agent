package com.visualagent.companionime.editor;

import android.view.inputmethod.ExtractedText;
import android.view.inputmethod.ExtractedTextRequest;
import android.view.inputmethod.InputConnection;

public final class InputConnectionAdapter implements EditorConnection {
    private final InputConnection inputConnection;

    public InputConnectionAdapter(InputConnection inputConnection) {
        this.inputConnection = inputConnection;
    }

    @Override
    public boolean finishComposingText() {
        return inputConnection.finishComposingText();
    }

    @Override
    public boolean commitText(String text) {
        return inputConnection.commitText(text, 1);
    }

    @Override
    public boolean clearText() {
        ExtractedTextRequest request = new ExtractedTextRequest();
        request.hintMaxChars = 0;
        request.hintMaxLines = 0;
        ExtractedText extracted = inputConnection.getExtractedText(request, 0);
        if (extracted != null
                && extracted.text != null
                && extracted.startOffset >= 0
                && extracted.partialStartOffset < 0
                && extracted.partialEndOffset < 0) {
            int start = extracted.startOffset;
            int length = extracted.text.length();
            if (inputConnection.setSelection(start, start + length)) {
                return inputConnection.commitText("", 1);
            }
        }
        boolean selected = inputConnection.performContextMenuAction(android.R.id.selectAll);
        return selected && inputConnection.commitText("", 1);
    }
}
