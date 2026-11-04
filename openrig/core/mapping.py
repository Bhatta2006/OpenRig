"""Humanoid bone-name -> role mapping.

Works from names plus hierarchy depth only, so it is testable without Blender.
Handles the common naming schemes (Mixamo, Unreal mannequin, 3ds Max Biped,
VRoid, Character Creator, Daz, generic ``.L/.R`` rigs). Anything it cannot
classify is simply left unmapped; the user can override the result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SIDES = ("L", "R")
LIMB_ROLES = ("clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe")

_EXCLUDE_TOKENS = {
    "ik", "pole", "ctrl", "control", "controller", "target", "twist", "roll",
    "helper", "mch", "tweak", "end", "top", "nub", "proxy", "corrective", "jiggle",
}
_FINGER = ("thumb", "index", "middle", "ring", "pinky", "pinkie", "little", "finger")
_LEFT = {"l", "left", "lt"}
_RIGHT = {"r", "right", "rt"}


def tokenize(name: str) -> list[str]:
    """Split ``mixamorig:LeftUpLeg`` / ``thigh_l`` / ``Bip01 L Thigh`` into lowercase words."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def classify(name: str, *, allow_twist: bool = False):
    """Return ``(role, side)`` for a bone name, or ``None``. Side is 'L', 'R' or 'C'."""
    tokens = tokenize(name)
    tset = set(tokens)
    excluded = tset & _EXCLUDE_TOKENS
    if allow_twist:
        excluded -= {"twist"}
    if excluded:
        return None
    joined = "".join(tokens)
    if any(f in joined for f in _FINGER):
        return None

    side = "C"
    if tset & _LEFT:
        side = "L"
    elif tset & _RIGHT:
        side = "R"

    if "forearm" in joined or "lowerarm" in joined:
        role = "forearm"
    elif "shldr" in joined or "upperarm" in joined or "arm" in tset:
        role = "upperarm"
    elif "hand" in joined:
        role = "hand"
    elif "clavicle" in joined or "collar" in joined or "shoulder" in joined:
        role = "clavicle"
    elif "thigh" in joined or "upleg" in joined or "upperleg" in joined:
        role = "thigh"
    elif "calf" in joined or "shin" in joined or "lowerleg" in joined or "leg" in tset:
        role = "shin"
    elif "foot" in joined:
        role = "foot"
    elif "toe" in joined or "ball" in tset:
        role = "toe"
    elif "neck" in joined:
        role = "neck"
    elif "head" in tset:
        role = "head"
    elif tset & {"pelvis", "hips", "hip"}:
        role = "hips"
    elif any(k in joined for k in ("spine", "abdomen", "waist", "chest", "torso", "stomach")):
        role = "spine"
    else:
        return None

    if role in LIMB_ROLES and side == "C":
        return None  # limb bones need a side
    if role not in LIMB_ROLES:
        side = "C"
    return role, side


@dataclass
class RigMap:
    """Result of mapping: role -> bone name(s)."""

    hips: str | None = None
    spine: list[str] = field(default_factory=list)
    neck: list[str] = field(default_factory=list)
    head: str | None = None
    limbs: dict[str, dict[str, str]] = field(default_factory=lambda: {"L": {}, "R": {}})

    def missing(self) -> list[str]:
        out = []
        if not self.hips:
            out.append("hips")
        if not self.spine:
            out.append("spine")
        if not self.head:
            out.append("head")
        for s in SIDES:
            for r in ("upperarm", "forearm", "hand", "thigh", "shin", "foot"):
                if r not in self.limbs[s]:
                    out.append(f"{r}.{s}")
        return out

    @property
    def ok(self) -> bool:
        return not self.missing()

    def to_dict(self) -> dict:
        return {"hips": self.hips, "spine": self.spine, "neck": self.neck,
                "head": self.head, "limbs": self.limbs}

    @classmethod
    def from_dict(cls, d: dict) -> "RigMap":
        m = cls()
        m.hips = d.get("hips")
        m.spine = list(d.get("spine", []))
        m.neck = list(d.get("neck", []))
        m.head = d.get("head")
        limbs = d.get("limbs", {})
        m.limbs = {"L": dict(limbs.get("L", {})), "R": dict(limbs.get("R", {}))}
        return m


def _depths(parents: dict[str, str | None]) -> dict[str, int]:
    depth: dict[str, int] = {}
    for name in parents:
        d, cur, seen = 0, name, set()
        while parents.get(cur) is not None and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            d += 1
        depth[name] = d
    return depth


def map_rig(parents: dict[str, str | None]) -> RigMap:
    """``parents`` maps every bone name to its parent name (or None)."""
    depth = _depths(parents)
    buckets: dict[tuple[str, str], list[str]] = {}
    for name in parents:
        c = classify(name)
        if c:
            buckets.setdefault(c, []).append(name)

    if ("neck", "C") not in buckets:  # e.g. Character Creator only has neck twist bones
        for name in parents:
            c = classify(name, allow_twist=True)
            if c == ("neck", "C"):
                buckets.setdefault(c, []).append(name)

    def shallowest(names):
        return sorted(names, key=lambda n: (depth[n], len(n), n))[0]

    def chain(names):
        out, used = [], set()
        for n in sorted(names, key=lambda n: (depth[n], n)):
            if depth[n] not in used:
                used.add(depth[n])
                out.append(n)
        return out

    m = RigMap()
    if ("hips", "C") in buckets:
        m.hips = shallowest(buckets[("hips", "C")])
    if ("spine", "C") in buckets:
        m.spine = [n for n in chain(buckets[("spine", "C")]) if n != m.hips]
    if ("neck", "C") in buckets:
        m.neck = chain(buckets[("neck", "C")])
    if ("head", "C") in buckets:
        m.head = shallowest(buckets[("head", "C")])
    for side in SIDES:
        for role in LIMB_ROLES:
            names = buckets.get((role, side))
            if names:
                m.limbs[side][role] = shallowest(names)
    return m
