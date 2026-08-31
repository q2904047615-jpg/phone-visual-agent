package com.visualagent.companionime.foreground;

import android.app.AppOpsManager;
import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.Context;
import android.os.Process;

import com.visualagent.companionime.protocol.ProtocolException;

public final class ForegroundAppIdentityReader {
    private static final long LOOKBACK_MILLIS = 24L * 60L * 60L * 1000L;
    // ACTIVITY_RESUMED and the deprecated MOVE_TO_FOREGROUND share value 1.
    private static final int ACTIVITY_RESUMED_EVENT = 1;

    private final Context context;

    public ForegroundAppIdentityReader(Context context) {
        this.context = context.getApplicationContext();
    }

    public ForegroundAppIdentity read(
            String editorPackageName,
            double editorEventAtEpoch) throws ProtocolException {
        double nowEpoch = System.currentTimeMillis() / 1000.0;
        if (ForegroundAppIdentity.isValidPackageName(editorPackageName)) {
            double eventAt = Double.isFinite(editorEventAtEpoch)
                    ? Math.min(editorEventAtEpoch, nowEpoch) : nowEpoch;
            return ForegroundAppIdentity.editorInfo(editorPackageName, eventAt, nowEpoch);
        }
        if (!hasUsageAccess(context)) {
            return ForegroundAppIdentity.unavailable(
                    "usage_access_not_granted", nowEpoch);
        }
        UsageStatsManager manager = (UsageStatsManager) context.getSystemService(
                Context.USAGE_STATS_SERVICE);
        if (manager == null) {
            return ForegroundAppIdentity.unavailable(
                    "usage_stats_unavailable", nowEpoch);
        }
        try {
            long nowMillis = System.currentTimeMillis();
            UsageEvents events = manager.queryEvents(
                    Math.max(0L, nowMillis - LOOKBACK_MILLIS), nowMillis);
            UsageEvents.Event event = new UsageEvents.Event();
            String latestPackage = null;
            long latestTimestamp = -1L;
            while (events != null && events.hasNextEvent()) {
                events.getNextEvent(event);
                String packageName = event.getPackageName();
                if (event.getEventType() == ACTIVITY_RESUMED_EVENT
                        && event.getTimeStamp() >= latestTimestamp
                        && ForegroundAppIdentity.isValidPackageName(packageName)) {
                    latestPackage = packageName;
                    latestTimestamp = event.getTimeStamp();
                }
            }
            if (latestPackage == null) {
                return ForegroundAppIdentity.unavailable(
                        "foreground_event_unavailable", nowEpoch);
            }
            return ForegroundAppIdentity.usageStats(
                    latestPackage, latestTimestamp / 1000.0, nowEpoch);
        } catch (RuntimeException ignored) {
            return ForegroundAppIdentity.unavailable(
                    "usage_stats_unavailable", nowEpoch);
        }
    }

    public static boolean hasUsageAccess(Context context) {
        AppOpsManager manager = (AppOpsManager) context.getSystemService(
                Context.APP_OPS_SERVICE);
        if (manager == null) {
            return false;
        }
        try {
            int mode = manager.checkOpNoThrow(
                    AppOpsManager.OPSTR_GET_USAGE_STATS,
                    Process.myUid(),
                    context.getPackageName());
            return mode == AppOpsManager.MODE_ALLOWED;
        } catch (RuntimeException ignored) {
            return false;
        }
    }
}
