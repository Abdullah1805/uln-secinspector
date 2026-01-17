class ProofBuilder:
    def build(self, target, f, evidence, impact):
        return f"""
---
### 🛑 {impact['impact']} تم اكتشاف ثغرة
**الموقع المستهدف:** `{target}`  
**نقطة النهاية:** `{f['url']}`  
**الباراميتر المصاب:** `{f['param']}`  

**دليل الإثبات (PoC):**
```bash
curl "{f['url']}?{f['param']}=1'%20AND%20SLEEP(5)--"
