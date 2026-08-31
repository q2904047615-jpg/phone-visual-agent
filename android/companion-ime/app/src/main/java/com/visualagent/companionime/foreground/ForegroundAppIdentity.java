package com.visualagent.companionime.foreground;

import com.visualagent.companionime.protocol.ProtocolException;

import java.util.regex.Pattern;

public final class ForegroundAppIdentity {
    public static final String SOURCE_EDITOR_INFO = "editor_info";
    public static final String SOURCE_USAGE_STATS = "usage_stats";
    public static final String SOURCE_UNAVAILABLE = "unavailable";

    private static final Pattern PACKAGE_NAME = Pattern.compile(
            "[A-Za-z][A-Za-z0-9_]*(?:\\.[A-Za-z0-9_]+)+");
    private static final Pattern REASON_CODE = Pattern.compile(
            "[a-z][a-z0-9_.-]{0,63}");

    private final String packageName;
    private final String source;
    private final Double eventAtEpoch;
    private final double observedAtEpoch;
    private final String reasonCode;

    private ForegroundAppIdentity(
            String packageName,
            String source,
            Double eventAtEpoch,
            double observedAtEpoch,
            String reasonCode) throws ProtocolException {
        this.packageName = packageName;
        this.source = source;
        this.eventAtEpoch = eventAtEpoch;
        this.observedAtEpoch = observedAtEpoch;
        this.reasonCode = reasonCode;
        validate();
    }

    public static ForegroundAppIdentity editorInfo(
            String packageName,
            double eventAtEpoch,
            double observedAtEpoch) throws ProtocolException {
        return new ForegroundAppIdentity(
                packageName, SOURCE_EDITOR_INFO, eventAtEpoch, observedAtEpoch, null);
    }

    public static ForegroundAppIdentity usageStats(
            String packageName,
            double eventAtEpoch,
            double observedAtEpoch) throws ProtocolException {
        return new ForegroundAppIdentity(
                packageName, SOURCE_USAGE_STATS, eventAtEpoch, observedAtEpoch, null);
    }

    public static ForegroundAppIdentity unavailable(
            String reasonCode,
            double observedAtEpoch) throws ProtocolException {
        return new ForegroundAppIdentity(
                null, SOURCE_UNAVAILABLE, null, observedAtEpoch, reasonCode);
    }

    public String packageName() {
        return packageName;
    }

    public String source() {
        return source;
    }

    public Double eventAtEpoch() {
        return eventAtEpoch;
    }

    public double observedAtEpoch() {
        return observedAtEpoch;
    }

    public String reasonCode() {
        return reasonCode;
    }

    public boolean available() {
        return !SOURCE_UNAVAILABLE.equals(source);
    }

    public static boolean isValidPackageName(String value) {
        return value != null && PACKAGE_NAME.matcher(value).matches();
    }

    private void validate() throws ProtocolException {
        if (!Double.isFinite(observedAtEpoch)) {
            throw new ProtocolException("foreground observed time is invalid");
        }
        if (SOURCE_UNAVAILABLE.equals(source)) {
            if (packageName != null || eventAtEpoch != null || reasonCode == null
                    || !REASON_CODE.matcher(reasonCode).matches()) {
                throw new ProtocolException("unavailable foreground identity is invalid");
            }
            return;
        }
        if ((!SOURCE_EDITOR_INFO.equals(source) && !SOURCE_USAGE_STATS.equals(source))
                || !isValidPackageName(packageName) || eventAtEpoch == null
                || !Double.isFinite(eventAtEpoch) || eventAtEpoch > observedAtEpoch
                || reasonCode != null) {
            throw new ProtocolException("foreground identity is invalid");
        }
    }
}
