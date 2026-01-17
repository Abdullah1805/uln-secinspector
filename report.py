class ProofBuilder:
    def build(self, target, finding):
        return f"""
# 🛑 {finding['impact']}

## Target
{target}

## Affected Endpoint
{finding['url']}

## Parameter
{finding['param']}

## Evidence
Delay: {finding['evidence']['delay']}s

## Proof of Concept
```bash
curl "{finding['url']}?{finding['param']}=1'%20AND%20SLEEP(5)--"
