package com.visualagent.companionime.replay;

import android.content.Context;
import android.content.SharedPreferences;

public final class SharedPreferencesReplayStore implements ReplayStore {
    private static final String PREFERENCES = "companion_ime_replay_v1";
    private final SharedPreferences preferences;

    public SharedPreferencesReplayStore(Context context) {
        preferences = context.getApplicationContext()
                .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
    }

    @Override
    public synchronized String read(String commandDigest) {
        return preferences.getString(commandDigest, null);
    }

    @Override
    public synchronized boolean write(String commandDigest, String state) {
        return preferences.edit().putString(commandDigest, state).commit();
    }
}
