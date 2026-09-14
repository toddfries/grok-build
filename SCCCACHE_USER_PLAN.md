# SCCache User Solution Plan

## Problem
The `_pbuild` user cannot execute Todd-owned sccache binaries (`/obfarm0.a/todd/bin/sccache-rustc`) due to permission restrictions, causing sccache server startup to fail with "no current exe available".

## Solution: Option 4 Implementation

### Step 1: Create dedicated sccache user
```bash
# As root on obfarm0:
doas useradd -m -d /obfarm0.a/sccache -s /sbin/nologin _sccache
```

### Step 2: Move existing cache to new user's home
```bash
# Stop any existing sccache processes first
doas pkill -f sccache || true

# Move cache directory
doas mv /obfarm0.a/todd/sccache /obfarm0.a/sccache/
doas chown -R _sccache:_sccache /obfarm0.a/sccache
```

### Step 3: Update farm-build-grok-both with user-switching pattern
Add to the top of `/home/todd/bin/farm-build-grok-both`:
```bash
#!/bin/ksh
# ... existing header ...

# User-switching pattern like vfb
if [ "$(id -u)" != "0" ] && [ "$(id -u)" != "$(id -u _sccache 2>/dev/null || echo 999)" ]; then
    echo "Relaunching as _sccache user"
    exec doas -u _sccache $0 "$@"
fi

# Continue as _sccache user or root
```

### Step 4: Update paths to use new sccache home
In the sccache setup section:
```bash
# Use _sccache user's home for cache
SCCACHE_DIR=${SCCACHE_DIR:-/obfarm0.a/sccache}
```

### Step 5: Ensure proper permissions
```bash
# Ensure the sccache binaries are executable by _sccache
doas chmod +rx /obfarm0.a/todd/bin/sccache-rustc
doas chmod +rx /obfarm0.a/todd/bin/ensure-sccache-server
```

### Step 6: Test the solution
```bash
# Test as regular user (should relaunch as _sccache)
doas -u todd /home/todd/bin/farm-build-grok-both status

# Should see proper sccache stats instead of all zeros
```

## Benefits
- Centralized sccache service under dedicated user
- Proper separation of concerns
- No permission conflicts between users
- Cache preserved (no re-caching needed)
- Follows established pattern from vfb helper

## Implementation Notes
- The `_sccache` user will own and run the sccache server
- All sccache binaries remain under Todd's ownership but are executable by _sccache
- Cache directory moved to `/obfarm0.a/sccache` for clear ownership
- The user-switching pattern ensures the script always runs with appropriate privileges