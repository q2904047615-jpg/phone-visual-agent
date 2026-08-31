package com.visualagent.companionime.foreground;

import com.visualagent.companionime.protocol.ProtocolException;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public final class ForegroundAppIdentityTest {
    @Test
    public void editorInfoIsTheExplicitHigherPriorityIdentity() throws Exception {
        ForegroundAppIdentity value = ForegroundAppIdentity.editorInfo(
                "com.tencent.mm", 1000.0, 1001.0);

        assertTrue(value.available());
        assertEquals("com.tencent.mm", value.packageName());
        assertEquals(ForegroundAppIdentity.SOURCE_EDITOR_INFO, value.source());
        assertNull(value.reasonCode());
    }

    @Test
    public void usageStatsAndUnavailableHaveDisjointShapes() throws Exception {
        ForegroundAppIdentity usage = ForegroundAppIdentity.usageStats(
                "com.android.settings", 990.0, 1000.0);
        ForegroundAppIdentity unavailable = ForegroundAppIdentity.unavailable(
                "usage_access_not_granted", 1000.0);

        assertEquals(ForegroundAppIdentity.SOURCE_USAGE_STATS, usage.source());
        assertFalse(unavailable.available());
        assertNull(unavailable.packageName());
        assertEquals("usage_access_not_granted", unavailable.reasonCode());
    }

    @Test
    public void invalidPackageAndFutureEventAreRejected() throws Exception {
        try {
            ForegroundAppIdentity.editorInfo("微信", 1000.0, 1000.0);
            fail("invalid package must be rejected");
        } catch (ProtocolException expected) {
            assertTrue(expected.getMessage().contains("foreground identity"));
        }
        try {
            ForegroundAppIdentity.usageStats("com.tencent.mm", 1001.0, 1000.0);
            fail("future event must be rejected");
        } catch (ProtocolException expected) {
            assertTrue(expected.getMessage().contains("foreground identity"));
        }
    }
}
