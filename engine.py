import streamlit as st
from recon import ReconEngine
from validators import ValidatorEngine
from impact import ImpactEngine
from proof import ProofBuilder
from scope import ScopeGuard

class SovereignBBEngine:
    def __init__(self, concurrency=40):
        self.recon = ReconEngine(concurrency)
        self.validator = ValidatorEngine()
        self.impact = ImpactEngine()
        self.proof = ProofBuilder()
        self.scope = ScopeGuard()

    async def run(self, target):
        if not self.scope.allowed(target):
            st.error(f"[!] Target out of scope: {target}")
            return False

        findings = await self.recon.discover(target)
        found_any = False

        for f in findings:
            valid, evidence = await self.validator.validate(f)
            if not valid:
                continue

            impact = self.impact.escalate(f, evidence)
            if not impact:
                continue

            report = self.proof.build(target, f, evidence, impact)
            st.markdown(report)
            found_any = True
        
        return found_any
