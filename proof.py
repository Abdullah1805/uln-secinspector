class ProofBuilder:
    def build(self, target, f, evidence, impact):
        return f"""
# 🛑 {impact['impact']}

## Target
{target}

## Affected Endpoint
{f['url']}

## Parameter
{f['param']}

## Impact
{impact['impact']} (Confidence: {impact['confidence']})

## Proof of Concept
```bash
curl "{f['url']}?{f['param']}=1'%20AND%20SLEEP(5)--"
