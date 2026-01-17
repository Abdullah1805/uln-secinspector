class ProofBuilder:
    def build(self, target, finding):
        e = finding["evidence"]

        return f"""
# 🛑 Time-Based SQL Injection

## 🎯 Target (In Scope)
{target}

## 📍 Endpoint
{finding['url']}

## 🔑 Parameter
{finding['param']}

## 🧪 Evidence
| Request Type | Response Time |
|-------------|---------------|
| Baseline    | {e['baseline']}s |
| Injected    | {e['injected']}s |

## 🔥 Proof of Concept
curl "{finding['url']}?{finding['param']}=1' AND SLEEP(5)--"
## ⚠️ Impact
An attacker can perform time-based inference attacks, potentially leading to full database compromise.

## ✅ Remediation
Use parameterized queries / prepared statements and input validation.
"""

    def build_bulk(self, target, findings):
        return "\n---\n".join(
            self.build(target, f) for f in findings
        )
