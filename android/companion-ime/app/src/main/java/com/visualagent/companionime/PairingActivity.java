package com.visualagent.companionime;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.text.InputType;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.visualagent.companionime.security.EncryptedPairingStore;
import com.visualagent.companionime.security.PairingClient;
import com.visualagent.companionime.security.PairingRecord;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class PairingActivity extends Activity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private EncryptedPairingStore pairingStore;
    private EditText host;
    private EditText port;
    private EditText certificate;
    private EditText oneTimeToken;
    private Button pairButton;
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_SECURE);
        pairingStore = new EncryptedPairingStore(this);
        setContentView(buildContent());
        refreshPairingStatus();
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private ScrollView buildContent() {
        int padding = Math.round(20 * getResources().getDisplayMetrics().density);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("Visual Agent Companion IME");
        title.setTextSize(24);
        content.addView(title, matchWrap());

        TextView explanation = new TextView(this);
        explanation.setText(
                "Pair once over pinned TLS, then enable and select this input method. "
                        + "The one-time token and text commands are never persisted.");
        explanation.setPadding(0, padding / 2, 0, padding / 2);
        content.addView(explanation, matchWrap());

        host = field("Controller host", InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_VARIATION_URI);
        port = field("TLS port", InputType.TYPE_CLASS_NUMBER);
        certificate = field("Certificate SHA-256 fingerprint", InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        oneTimeToken = field("One-time pairing token", InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        oneTimeToken.setSaveEnabled(false);
        oneTimeToken.setImportantForAutofill(
                android.view.View.IMPORTANT_FOR_AUTOFILL_NO_EXCLUDE_DESCENDANTS);
        content.addView(host, matchWrap());
        content.addView(port, matchWrap());
        content.addView(certificate, matchWrap());
        content.addView(oneTimeToken, matchWrap());

        pairButton = button("Pair", view -> pair());
        content.addView(pairButton, matchWrap());
        content.addView(button("Open input-method settings", view ->
                startActivity(new Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))), matchWrap());
        content.addView(button("Select input method", view -> {
            InputMethodManager manager =
                    (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
            manager.showInputMethodPicker();
        }), matchWrap());
        content.addView(button("Remove pairing", view -> removePairing()), matchWrap());

        status = new TextView(this);
        status.setPadding(0, padding / 2, 0, 0);
        content.addView(status, matchWrap());

        ScrollView scroll = new ScrollView(this);
        scroll.addView(content, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return scroll;
    }

    private EditText field(String hint, int inputType) {
        EditText result = new EditText(this);
        result.setHint(hint);
        result.setInputType(inputType);
        result.setSingleLine(true);
        return result;
    }

    private Button button(String label, android.view.View.OnClickListener listener) {
        Button result = new Button(this);
        result.setText(label);
        result.setOnClickListener(listener);
        return result;
    }

    private static LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private void pair() {
        String controllerHost = host.getText().toString().trim();
        String portText = port.getText().toString().trim();
        String fingerprint = certificate.getText().toString().trim();
        char[] tokenCharacters = oneTimeToken.getText().toString().toCharArray();
        oneTimeToken.setText("");
        pairButton.setEnabled(false);
        status.setText("Pairing…");
        executor.execute(() -> {
            try {
                int controllerPort = Integer.parseInt(portText);
                String installationId = pairingStore.getOrCreateInstallationId();
                String token = new String(tokenCharacters);
                PairingRecord record;
                try {
                    record = new PairingClient().pair(
                            controllerHost,
                            controllerPort,
                            fingerprint,
                            installationId,
                            token,
                            pairingStore);
                } finally {
                    Arrays.fill(tokenCharacters, '\0');
                }
                record.destroy();
                runOnUiThread(() -> status.setText(
                        "Paired. Enable and select the Companion IME to use it."));
            } catch (Exception ignored) {
                Arrays.fill(tokenCharacters, '\0');
                runOnUiThread(() -> status.setText(
                        "Pairing failed. Check the host, port, certificate pin, and token."));
            } finally {
                runOnUiThread(() -> pairButton.setEnabled(true));
            }
        });
    }

    private void removePairing() {
        try {
            pairingStore.clear();
            status.setText("Pairing removed.");
        } catch (Exception ignored) {
            status.setText("Pairing could not be removed.");
        }
    }

    private void refreshPairingStatus() {
        PairingRecord record = null;
        try {
            record = pairingStore.load();
            status.setText(record == null ? "Not paired." : "Paired.");
        } catch (Exception ignored) {
            status.setText("Stored pairing is unavailable; remove it and pair again.");
        } finally {
            if (record != null) {
                record.destroy();
            }
        }
    }
}
