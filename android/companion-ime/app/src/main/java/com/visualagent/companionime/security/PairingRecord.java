package com.visualagent.companionime.security;

import java.util.Arrays;

public final class PairingRecord {
    private final String host;
    private final int port;
    private final String certificateSha256;
    private final String installationId;
    private final String pairingId;
    private final String deviceId;
    private final byte[] sharedKey;

    public PairingRecord(
            String host,
            int port,
            String certificateSha256,
            String installationId,
            String pairingId,
            String deviceId,
            byte[] sharedKey) {
        this.host = host;
        this.port = port;
        this.certificateSha256 = certificateSha256;
        this.installationId = installationId;
        this.pairingId = pairingId;
        this.deviceId = deviceId;
        this.sharedKey = sharedKey.clone();
    }

    public String host() {
        return host;
    }

    public int port() {
        return port;
    }

    public String certificateSha256() {
        return certificateSha256;
    }

    public String installationId() {
        return installationId;
    }

    public String pairingId() {
        return pairingId;
    }

    public String deviceId() {
        return deviceId;
    }

    public byte[] sharedKey() {
        return sharedKey.clone();
    }

    public void destroy() {
        Arrays.fill(sharedKey, (byte) 0);
    }
}
