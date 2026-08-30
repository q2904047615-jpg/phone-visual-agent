package com.visualagent.companionime.security;

import com.visualagent.companionime.protocol.Hashing;
import com.visualagent.companionime.protocol.ProtocolException;

import java.net.InetSocketAddress;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Locale;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

public final class PinnedTlsSocket {
    private static final int CONNECT_TIMEOUT_MS = 5_000;
    private static final int READ_TIMEOUT_MS = 15_000;

    private PinnedTlsSocket() {
    }

    public static SSLSocket connect(String host, int port, String certificateSha256)
            throws java.io.IOException, ProtocolException {
        String normalized = normalizeFingerprint(certificateSha256);
        SSLSocket socket = null;
        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, new TrustManager[]{new PinTrustManager(normalized)}, null);
            socket = (SSLSocket) context.getSocketFactory().createSocket();
            List<String> supported = Arrays.asList(socket.getSupportedProtocols());
            List<String> enabled = new ArrayList<>();
            if (supported.contains("TLSv1.3")) {
                enabled.add("TLSv1.3");
            }
            if (supported.contains("TLSv1.2")) {
                enabled.add("TLSv1.2");
            }
            if (enabled.isEmpty()) {
                throw new GeneralSecurityException("TLS 1.2 or newer is unavailable");
            }
            socket.setEnabledProtocols(enabled.toArray(new String[0]));
            socket.connect(new InetSocketAddress(host, port), CONNECT_TIMEOUT_MS);
            socket.setSoTimeout(READ_TIMEOUT_MS);
            socket.startHandshake();
            return socket;
        } catch (GeneralSecurityException error) {
            if (socket != null) {
                try {
                    socket.close();
                } catch (java.io.IOException ignored) {
                    // Preserve the TLS setup failure.
                }
            }
            throw new ProtocolException("unable to initialize pinned TLS", error);
        } catch (java.io.IOException error) {
            if (socket != null) {
                try {
                    socket.close();
                } catch (java.io.IOException ignored) {
                    // Preserve the connection or handshake failure.
                }
            }
            throw error;
        }
    }

    public static String normalizeFingerprint(String fingerprint) throws ProtocolException {
        String normalized = fingerprint.replace(":", "")
                .replace("-", "")
                .replace(" ", "")
                .toLowerCase(Locale.ROOT);
        if (!normalized.matches("[0-9a-f]{64}")) {
            throw new ProtocolException("certificate fingerprint must be SHA-256 hex");
        }
        return normalized;
    }

    private static final class PinTrustManager implements X509TrustManager {
        private final String expected;

        private PinTrustManager(String expected) {
            this.expected = expected;
        }

        @Override
        public void checkClientTrusted(X509Certificate[] chain, String authType)
                throws CertificateException {
            throw new CertificateException("client certificates are not accepted");
        }

        @Override
        public void checkServerTrusted(X509Certificate[] chain, String authType)
                throws CertificateException {
            if (chain == null || chain.length == 0) {
                throw new CertificateException("server did not provide a certificate");
            }
            chain[0].checkValidity();
            try {
                String actual = Hashing.sha256Hex(chain[0].getEncoded());
                if (!MessageDigest.isEqual(
                        actual.getBytes(java.nio.charset.StandardCharsets.US_ASCII),
                        expected.getBytes(java.nio.charset.StandardCharsets.US_ASCII))) {
                    throw new CertificateException("server certificate pin did not match");
                }
            } catch (java.security.cert.CertificateEncodingException error) {
                throw new CertificateException("unable to encode server certificate", error);
            }
        }

        @Override
        public X509Certificate[] getAcceptedIssuers() {
            return new X509Certificate[0];
        }
    }
}
