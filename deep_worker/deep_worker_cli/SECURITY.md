# Security Policy for deep-worker-cli

## Overview

deep-worker-cli is designed to operate **EXCLUSIVELY in a trusted local environment**. The security model assumes that:

1. All processes run on the same local machine
2. The local environment is trusted
3. No malicious processes can access localhost services
4. Network isolation prevents external access

## ⚠️ CRITICAL SECURITY REQUIREMENTS

### 1. Localhost-Only Binding

**ALL network services MUST bind ONLY to localhost (127.0.0.1)**

- ✅ RPyC server: Bound to `127.0.0.1` (enforced in code)
- ❌ NEVER bind to `0.0.0.0` or external interfaces
- ❌ NEVER expose ports to external networks

### 2. Network Isolation

The following network services are used **for localhost communication only**:

#### RPyC Server (Port 18861 by default)
- **Binding**: `127.0.0.1` (localhost only)
- **Purpose**: All IPC communication between agent and dwcli
- **Authentication**: NONE (relies on localhost isolation)
- **Protocol Config**:
  - `allow_public_attrs: True` - Safe ONLY on localhost
  - `allow_pickle: True` - Safe ONLY on localhost
- **Security**: Protected by localhost-only binding

### 3. Trust Assumptions

The system assumes:

- ✅ All processes on localhost are trusted
- ✅ No malicious code can connect to localhost services
- ✅ The user running deep-worker has legitimate intent

### 4. Insecure Configurations (DO NOT USE)

The following configurations are **EXPLICITLY FORBIDDEN**:

```python
# ❌ FORBIDDEN: External interface binding
ThreadedServer(service, hostname="0.0.0.0", port=18861)

# ❌ FORBIDDEN: Wildcard binding
ThreadedServer(service, hostname="", port=18861)

# ❌ FORBIDDEN: External IP binding
ThreadedServer(service, hostname="192.168.1.100", port=18861)
```

**Always use:**

```python
# ✅ CORRECT: Localhost only
ThreadedServer(service, hostname="127.0.0.1", port=18861)
```

## Threat Model

### Protected Against

1. ✅ **External network access**: All services bind to localhost only
2. ✅ **Remote exploitation**: No external network exposure
3. ✅ **Port scanning**: Ports not visible from external networks
4. ✅ **Network-based attacks**: Services not reachable externally

### NOT Protected Against

1. ❌ **Malicious local processes**: Any process on localhost can connect
2. ❌ **Privilege escalation**: No authentication between local processes
3. ❌ **Code injection**: Pickle deserialization accepts arbitrary code from localhost

## Security Best Practices

### For Users

1. **Run in trusted environments only**
   - Use on personal workstations
   - Use in isolated development containers
   - DO NOT use on shared multi-user systems

2. **Verify localhost binding**
   ```bash
   # Check that RPyC server is bound to localhost only
   netstat -tuln | grep 18861
   # Should show: 127.0.0.1:18861, NOT 0.0.0.0:18861
   ```

3. **Use firewall rules** (defense in depth)
   ```bash
   # Block external access to deep-worker ports
   sudo ufw deny from any to any port 18861
   ```

### For Developers

1. **Never change localhost binding**
   - Always use `hostname="127.0.0.1"`
   - Add code comments explaining security requirements
   - Reject PRs that bind to external interfaces

2. **Code review checklist**
   - ✅ All servers bind to `127.0.0.1`
   - ✅ No `0.0.0.0` bindings
   - ✅ No external network access
   - ✅ Security comments are present

3. **Testing security**
   ```bash
   # Test that external access is blocked
   # From another machine:
   telnet <machine-ip> 18861  # Should fail/timeout

   # From localhost:
   telnet 127.0.0.1 18861  # Should connect
   ```

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public GitHub issue
2. **DO** report privately to the maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Compliance

This security model is suitable for:

- ✅ Personal development environments
- ✅ Isolated development containers
- ✅ CI/CD pipelines (isolated runners)
- ✅ Local testing and experimentation

This security model is **NOT suitable** for:

- ❌ Production servers accessible from networks
- ❌ Multi-user systems with untrusted users
- ❌ Cloud environments with shared infrastructure
- ❌ Any scenario requiring authentication/authorization

## Version History

- **v0.0.19+**: RPyC-only architecture with localhost-only binding (Unix sockets removed)

## References

- [RPyC Security Documentation](https://rpyc.readthedocs.io/en/latest/docs/security.html)
- [OWASP Localhost Security](https://owasp.org/www-community/attacks/Localhost_IP_Address_Attack)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
