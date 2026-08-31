package com.visualagent.companionime;

import android.inputmethodservice.InputMethodService;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;

import com.visualagent.companionime.editor.EditorActionRunner;
import com.visualagent.companionime.editor.InputConnectionAdapter;
import com.visualagent.companionime.foreground.ForegroundAppIdentity;
import com.visualagent.companionime.foreground.ForegroundAppIdentityReader;
import com.visualagent.companionime.protocol.CommandEnvelope;
import com.visualagent.companionime.replay.ReplayGuard;
import com.visualagent.companionime.replay.SharedPreferencesReplayStore;
import com.visualagent.companionime.security.EncryptedPairingStore;
import com.visualagent.companionime.security.PairingRecord;
import com.visualagent.companionime.transport.CompanionCommandClient;

import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class CompanionImeService extends InputMethodService {
    private static final long IDLE_WAIT_MS = 500L;
    private static final long EXECUTION_WAIT_MS = 5_000L;

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Object editorLock = new Object();

    private volatile String currentEditorSessionId;
    private volatile String currentEditorPackageName;
    private volatile double currentEditorEventAtEpoch = Double.NaN;
    private Thread worker;
    private Thread foregroundWorker;
    private EncryptedPairingStore pairingStore;
    private SharedPreferences.OnSharedPreferenceChangeListener pairingListener;
    private ReplayGuard replayGuard;
    private final CompanionCommandClient commandClient = new CompanionCommandClient();
    private final CompanionCommandClient foregroundClient = new CompanionCommandClient();
    private final EditorActionRunner actionRunner = new EditorActionRunner();
    private ForegroundAppIdentityReader foregroundIdentityReader;

    @Override
    public void onCreate() {
        super.onCreate();
        pairingStore = new EncryptedPairingStore(this);
        pairingListener = (preferences, key) -> {
            commandClient.cancelActive();
            foregroundClient.cancelActive();
        };
        pairingStore.registerChangeListener(pairingListener);
        replayGuard = new ReplayGuard(new SharedPreferencesReplayStore(this));
        foregroundIdentityReader = new ForegroundAppIdentityReader(this);
        running.set(true);
        worker = new Thread(this::runCommandLoop, "companion-ime-command-loop");
        worker.start();
        foregroundWorker = new Thread(
                this::runForegroundLoop, "companion-ime-foreground-loop");
        foregroundWorker.start();
    }

    @Override
    public void onStartInput(EditorInfo attribute, boolean restarting) {
        super.onStartInput(attribute, restarting);
        synchronized (editorLock) {
            currentEditorSessionId = UUID.randomUUID().toString();
            currentEditorPackageName = attribute == null ? null : attribute.packageName;
            currentEditorEventAtEpoch = System.currentTimeMillis() / 1000.0;
        }
        commandClient.cancelActive();
        foregroundClient.cancelActive();
        if (worker != null) {
            worker.interrupt();
        }
        if (foregroundWorker != null) {
            foregroundWorker.interrupt();
        }
    }

    @Override
    public void onFinishInput() {
        synchronized (editorLock) {
            currentEditorSessionId = null;
            currentEditorPackageName = null;
            currentEditorEventAtEpoch = Double.NaN;
        }
        commandClient.cancelActive();
        foregroundClient.cancelActive();
        super.onFinishInput();
    }

    @Override
    public void onDestroy() {
        running.set(false);
        synchronized (editorLock) {
            currentEditorSessionId = null;
            currentEditorPackageName = null;
            currentEditorEventAtEpoch = Double.NaN;
        }
        commandClient.cancelActive();
        foregroundClient.cancelActive();
        if (pairingStore != null && pairingListener != null) {
            pairingStore.unregisterChangeListener(pairingListener);
        }
        if (worker != null) {
            worker.interrupt();
        }
        if (foregroundWorker != null) {
            foregroundWorker.interrupt();
        }
        super.onDestroy();
    }

    private void runCommandLoop() {
        while (running.get()) {
            String editorSessionId;
            String editorPackageName;
            double editorEventAtEpoch;
            synchronized (editorLock) {
                editorSessionId = currentEditorSessionId;
                editorPackageName = currentEditorPackageName;
                editorEventAtEpoch = currentEditorEventAtEpoch;
            }
            if (editorSessionId == null) {
                waitWithoutLogging();
                continue;
            }
            try {
                PairingRecord pairing = pairingStore.load();
                if (pairing == null) {
                    waitWithoutLogging();
                    continue;
                }
                try {
                    ForegroundAppIdentity identity = foregroundIdentityReader.read(
                            editorPackageName, editorEventAtEpoch);
                    commandClient.processEditorSession(
                            pairing,
                            editorSessionId,
                            identity,
                            replayGuard,
                            command -> executeOnCurrentEditor(editorSessionId, command),
                            () -> running.get()
                                    && editorSessionId.equals(currentEditorSessionId));
                } finally {
                    pairing.destroy();
                }
            } catch (Exception ignored) {
                // Deliberately do not log protocol payloads or exception messages.
            }
            waitWithoutLogging();
        }
    }

    private void runForegroundLoop() {
        while (running.get()) {
            try {
                PairingRecord pairing = pairingStore.load();
                if (pairing == null) {
                    waitWithoutLogging();
                    continue;
                }
                try {
                    String editorPackageName;
                    double editorEventAtEpoch;
                    synchronized (editorLock) {
                        editorPackageName = currentEditorPackageName;
                        editorEventAtEpoch = currentEditorEventAtEpoch;
                    }
                    ForegroundAppIdentity identity = foregroundIdentityReader.read(
                            editorPackageName, editorEventAtEpoch);
                    foregroundClient.publishForegroundState(pairing, identity);
                } finally {
                    pairing.destroy();
                }
            } catch (Exception ignored) {
                // Deliberately do not log identities, protocol payloads, or exceptions.
            }
            waitWithoutLogging();
        }
    }

    private boolean executeOnCurrentEditor(
            String expectedEditorSessionId,
            CommandEnvelope command) {
        CountDownLatch complete = new CountDownLatch(1);
        AtomicBoolean accepted = new AtomicBoolean(false);
        AtomicBoolean cancelled = new AtomicBoolean(false);
        mainHandler.post(() -> {
            try {
                if (cancelled.get()
                        || !expectedEditorSessionId.equals(currentEditorSessionId)) {
                    return;
                }
                InputConnection current = getCurrentInputConnection();
                if (current == null) {
                    return;
                }
                accepted.set(actionRunner.execute(command, new InputConnectionAdapter(current)));
            } catch (RuntimeException ignored) {
                accepted.set(false);
            } finally {
                complete.countDown();
            }
        });
        try {
            if (!complete.await(EXECUTION_WAIT_MS, TimeUnit.MILLISECONDS)) {
                cancelled.set(true);
                return false;
            }
            return accepted.get();
        } catch (InterruptedException interrupted) {
            cancelled.set(true);
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private void waitWithoutLogging() {
        try {
            Thread.sleep(IDLE_WAIT_MS);
        } catch (InterruptedException ignored) {
            // Editor changes wake the worker so it can bind a fresh editor_session_id.
        }
    }
}
