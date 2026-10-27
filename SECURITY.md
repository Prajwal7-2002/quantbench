# Security Policy 🔒

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `0.1.x` | :white_check_mark: |

---

## Safe Tensor Guidelines

1. **Pure Tensor Math**: `quantbench-engine` performs purely arithmetic bit-packing and matrix operations. It requires zero network connectivity.
2. **Safe Serialization**: Saved quantized checkpoints strictly serialize standard integer and float tensor dictionaries (`state_dict`). Untrusted pickle formats should never be loaded.

---

## Reporting a Vulnerability

If you discover a security vulnerability or numerical flaw that impacts host security:

* **Please DO NOT file a public issue.**
* Report the issue privately through [GitHub's Private Vulnerability Reporting](https://github.com/Prajwal7-2002/quantbench/security/advisories/new) or email maintainer Prajwal Pujari at `prajwalpujari16@gmail.com`.

### Submission Checklist
Please include:
1. Clear description of the vulnerability.
2. Minimal reproduction script.
3. Potential security impact.

You will receive an acknowledgment within 48 hours, followed by updates on the fix and coordinated disclosure.
