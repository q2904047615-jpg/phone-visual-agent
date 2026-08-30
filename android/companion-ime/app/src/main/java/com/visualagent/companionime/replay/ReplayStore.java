package com.visualagent.companionime.replay;

public interface ReplayStore {
    String read(String commandDigest);

    boolean write(String commandDigest, String state);
}
