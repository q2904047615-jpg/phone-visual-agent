package com.visualagent.companionime.security;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.util.Base64;
import java.util.Arrays;
import java.util.UUID;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

public final class EncryptedPairingStore {
    private static final String PREFERENCES = "companion_ime_pairing_v1";
    private static final String KEY_ALIAS = "visual_agent_companion_ime_pairing_key_v1";
    private static final String AAD = "visual-agent-companion-ime-pairing-v1";

    private static final String HOST = "host";
    private static final String PORT = "port";
    private static final String CERTIFICATE = "certificate_sha256";
    private static final String INSTALLATION = "installation_id";
    private static final String PAIR = "pairing_id";
    private static final String DEVICE = "device_id";
    private static final String KEY_IV = "shared_key_iv";
    private static final String KEY_CIPHERTEXT = "shared_key_ciphertext";
    private static final String PENDING_PREFIX = "pending_";

    private final SharedPreferences preferences;

    public EncryptedPairingStore(Context context) {
        preferences = context.getApplicationContext()
                .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
    }

    public void registerChangeListener(
            SharedPreferences.OnSharedPreferenceChangeListener listener) {
        preferences.registerOnSharedPreferenceChangeListener(listener);
    }

    public void unregisterChangeListener(
            SharedPreferences.OnSharedPreferenceChangeListener listener) {
        preferences.unregisterOnSharedPreferenceChangeListener(listener);
    }

    public synchronized String getOrCreateInstallationId() throws PairingStoreException {
        String existing = preferences.getString(INSTALLATION, null);
        if (existing != null && !existing.isEmpty()) {
            return existing;
        }
        String created = UUID.randomUUID().toString();
        if (!preferences.edit().putString(INSTALLATION, created).commit()) {
            throw new PairingStoreException("unable to persist installation identity");
        }
        return created;
    }

    public synchronized void save(PairingRecord record) throws PairingStoreException {
        saveSlot(record, false);
    }

    public synchronized void savePending(PairingRecord record) throws PairingStoreException {
        saveSlot(record, true);
    }

    private void saveSlot(PairingRecord record, boolean pending) throws PairingStoreException {
        byte[] plaintextKey = record.sharedKey();
        try {
            if (plaintextKey.length != 32) {
                throw new PairingStoreException("pairing key must contain 32 bytes");
            }
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, getOrCreateSecretKey());
            cipher.updateAAD(AAD.getBytes(StandardCharsets.UTF_8));
            byte[] ciphertext = cipher.doFinal(plaintextKey);
            String iv = Base64.getEncoder().encodeToString(cipher.getIV());
            String encrypted = Base64.getEncoder().encodeToString(ciphertext);
            boolean committed = preferences.edit()
                    .putString(slot(pending, HOST), record.host())
                    .putInt(slot(pending, PORT), record.port())
                    .putString(slot(pending, CERTIFICATE), record.certificateSha256())
                    .putString(slot(pending, INSTALLATION), record.installationId())
                    .putString(slot(pending, PAIR), record.pairingId())
                    .putString(slot(pending, DEVICE), record.deviceId())
                    .putString(slot(pending, KEY_IV), iv)
                    .putString(slot(pending, KEY_CIPHERTEXT), encrypted)
                    .commit();
            if (!committed) {
                throw new PairingStoreException("unable to persist encrypted pairing data");
            }
        } catch (GeneralSecurityException | IllegalArgumentException error) {
            throw new PairingStoreException("unable to encrypt pairing data", error);
        } finally {
            Arrays.fill(plaintextKey, (byte) 0);
        }
    }

    public synchronized PairingRecord load() throws PairingStoreException {
        return loadSlot(false);
    }

    public synchronized PairingRecord loadPending() throws PairingStoreException {
        return loadSlot(true);
    }

    private PairingRecord loadSlot(boolean pending) throws PairingStoreException {
        String host = preferences.getString(slot(pending, HOST), null);
        int port = preferences.getInt(slot(pending, PORT), -1);
        String certificate = preferences.getString(slot(pending, CERTIFICATE), null);
        String installation = preferences.getString(slot(pending, INSTALLATION), null);
        String pair = preferences.getString(slot(pending, PAIR), null);
        String device = preferences.getString(slot(pending, DEVICE), null);
        String encodedIv = preferences.getString(slot(pending, KEY_IV), null);
        String encodedCiphertext = preferences.getString(slot(pending, KEY_CIPHERTEXT), null);
        if (host == null || port < 1 || certificate == null || installation == null
                || pair == null || device == null || encodedIv == null
                || encodedCiphertext == null) {
            return null;
        }
        byte[] key = null;
        try {
            byte[] iv = Base64.getDecoder().decode(encodedIv);
            byte[] ciphertext = Base64.getDecoder().decode(encodedCiphertext);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, getOrCreateSecretKey(),
                    new GCMParameterSpec(128, iv));
            cipher.updateAAD(AAD.getBytes(StandardCharsets.UTF_8));
            key = cipher.doFinal(ciphertext);
            if (key.length != 32) {
                throw new PairingStoreException("decrypted pairing key has an invalid length");
            }
            PairingRecord result = new PairingRecord(
                    host, port, certificate, installation, pair, device, key);
            return result;
        } catch (GeneralSecurityException | IllegalArgumentException error) {
            throw new PairingStoreException("unable to decrypt pairing data", error);
        } finally {
            if (key != null) {
                Arrays.fill(key, (byte) 0);
            }
        }
    }

    public synchronized void promotePending() throws PairingStoreException {
        PairingRecord pending = loadPending();
        if (pending == null) {
            throw new PairingStoreException("pending pairing is unavailable");
        }
        try {
            save(pending);
            discardPending();
        } finally {
            pending.destroy();
        }
    }

    public synchronized void discardPending() throws PairingStoreException {
        boolean committed = preferences.edit()
                .remove(slot(true, HOST))
                .remove(slot(true, PORT))
                .remove(slot(true, CERTIFICATE))
                .remove(slot(true, INSTALLATION))
                .remove(slot(true, PAIR))
                .remove(slot(true, DEVICE))
                .remove(slot(true, KEY_IV))
                .remove(slot(true, KEY_CIPHERTEXT))
                .commit();
        if (!committed) {
            throw new PairingStoreException("unable to discard pending pairing data");
        }
    }

    public synchronized void clear() throws PairingStoreException {
        if (!preferences.edit().clear().commit()) {
            throw new PairingStoreException("unable to clear pairing data");
        }
        try {
            KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
            keyStore.load(null);
            if (keyStore.containsAlias(KEY_ALIAS)) {
                keyStore.deleteEntry(KEY_ALIAS);
            }
        } catch (GeneralSecurityException | java.io.IOException error) {
            throw new PairingStoreException("unable to delete pairing encryption key", error);
        }
    }

    private static String slot(boolean pending, String key) {
        return pending ? PENDING_PREFIX + key : key;
    }

    private static SecretKey getOrCreateSecretKey() throws GeneralSecurityException {
        try {
            KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
            keyStore.load(null);
            java.security.Key existing = keyStore.getKey(KEY_ALIAS, null);
            if (existing instanceof SecretKey) {
                return (SecretKey) existing;
            }
        } catch (java.io.IOException error) {
            throw new GeneralSecurityException("unable to open Android Keystore", error);
        }

        KeyGenerator generator = KeyGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build());
        return generator.generateKey();
    }
}
