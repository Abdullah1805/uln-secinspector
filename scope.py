---

## 🛡️ scope.py (حماية قانونية)
```python
class ScopeGuard:
    def allowed(self, url):
        allowed_domains = ["example.com"]
        return any(d in url for d in allowed_domains)
