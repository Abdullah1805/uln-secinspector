class ProofBuilder:
    def build(self, target, finding):
        lines = []

        lines.append(f"# 🛑 {finding.get('impact', 'Security Impact')}")
        lines.append("")
        lines.append("## 🎯 Target (In Scope)")
        lines.append(target)
        lines.append("")
        lines.append("## 📍 Endpoint")
        lines.append(finding.get("url", "N/A"))
        lines.append("")
        lines.append("## 🔑 Parameter")
        lines.append(finding.get("param", "N/A"))
        lines.append("")
        lines.append("## 🧪 Evidence")
        delay = finding.get("evidence", {}).get("delay", "unknown")
        lines.append(f"Response delayed by {delay} seconds")
        lines.append("")
        lines.append("## 🔥 Proof of Concept")
        lines.append(
            f'curl "{finding.get("url")}?{finding.get("param")}=1\' AND SLEEP(5)--"'
        )
        lines.append("")
        lines.append("## ⚠️ Impact")
        lines.append(
            "This vulnerability allows time-based inference and may lead to full data extraction."
        )

        return "\n".join(lines)

    def build_bulk(self, target, findings):
        reports = []
        for f in findings:
            reports.append(self.build(target, f))
            reports.append("\n---\n")
        return "\n".join(reports)
