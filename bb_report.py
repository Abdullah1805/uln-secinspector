class ProofBuilder:
    def build(self, target, f):
        return f"""
# 🛑 {f['impact']}

## 🎯 Target
{target}

## 📍 Endpoint
{f['url']}

## 🔑 Parameter
{f['param']}

## 📊 Confidence
{f['confidence']}%

## 🧪 Evidence
{f.get('evidence', 'N/A')}

## ⚠️ Impact
This issue may lead to unauthorized access or data exposure.

## ✅ Recommendation
Apply strict validation, authorization checks, and secure coding practices.
"""

    def build_bulk(self, target, findings):
        return "\n---\n".join(self.build(target, f) for f in findings)
