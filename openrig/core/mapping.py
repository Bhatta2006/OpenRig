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

    def assign(self, role: str, bones: list[str]) -> None:
        """Set one role by hand. ``role`` is 'hips', 'head', 'spine', 'neck' or
        '<limb role>.<L|R>' (e.g. 'forearm.L'). ``spine``/``neck`` take bones ordered
        root to tip; the others take exactly one bone (an empty list clears optional roles)."""
        single = role in ("hips", "head") or role.partition(".")[0] in LIMB_ROLES
        if single and len(bones) > 1:
            raise ValueError(f"{role} takes exactly one bone, got {len(bones)}")
        if len(set(bones)) != len(bones):
            raise ValueError("the same bone was given twice")
        used = self._used_by_others(role)
        clash = [b for b in bones if b in used]
        if clash:
            raise ValueError(f"{clash[0]} is already used as {used[clash[0]]}")
        if role == "hips":
            self.hips = bones[0] if bones else None
        elif role == "head":
            self.head = bones[0] if bones else None
        elif role == "spine":
            self.spine = list(bones)
        elif role == "neck":
            self.neck = list(bones)
        else:
            name, _, side = role.partition(".")
            if name not in LIMB_ROLES or side not in SIDES:
                raise ValueError(f"unknown role {role!r}")
            if bones:
                self.limbs[side][name] = bones[0]
            else:
                self.limbs[side].pop(name, None)

    def _used_by_others(self, role: str) -> dict[str, str]:
        used = {}
        if role != "hips" and self.hips:
            used[self.hips] = "hips"
        if role != "head" and self.head:
            used[self.head] = "head"
        if role != "spine":
            used.update({b: "spine" for b in self.spine})
        if role != "neck":
            used.update({b: "neck" for b in self.neck})
        for side, limbs in self.limbs.items():
            for r, b in limbs.items():
                if f"{r}.{side}" != role:
                    used[b] = f"{r}.{side}"
        return used

    def rows(self) -> list[tuple[str, str]]:
        """(label, bone names) for display."""
        out = [("Hips", self.hips or "-"), ("Spine", ", ".join(self.spine) or "-"),
               ("Neck", ", ".join(self.neck) or "-"), ("Head", self.head or "-")]
        for s in SIDES:
            for r in LIMB_ROLES:
                out.append((f"{r.title()} {s}", self.limbs[s].get(r, "-")))
        return out

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


_RIGIFY_LIMBS = {"clavicle": "shoulder", "upperarm": "upper_arm", "forearm": "forearm", "hand": "hand",
                 "thigh": "thigh", "shin": "shin", "foot": "foot", "toe": "toe"}


def _map_rigify(parents: dict[str, str | None]) -> RigMap | None:
    """Rigify deformation (DEF-), original (ORG-) or meta rig. Its spine bones are numbered
    ``spine`` (pelvis), ``spine.001``-``.003`` (back), ``.004``-``.005`` (neck), ``.006`` (head),
    which name matching alone cannot tell apart."""
    for prefix in ("DEF-", "ORG-", ""):
        if f"{prefix}spine" in parents and f"{prefix}spine.006" in parents:
            break
    else:
        return None
    m = RigMap()
    m.hips = f"{prefix}spine"
    m.spine = [f"{prefix}spine.{i:03d}" for i in (1, 2, 3)]
    m.neck = [f"{prefix}spine.{i:03d}" for i in (4, 5)]
    m.head = f"{prefix}spine.006"
    for side in SIDES:
        for role, base in _RIGIFY_LIMBS.items():
            name = f"{prefix}{base}.{side}"
            if name in parents:
                m.limbs[side][role] = name
    m.spine = [n for n in m.spine if n in parents]
    m.neck = [n for n in m.neck if n in parents]
    return m


def map_rig(parents: dict[str, str | None]) -> RigMap:
    """``parents`` maps every bone name to its parent name (or None)."""
    rigify = _map_rigify(parents)
    if rigify is not None and rigify.ok:  # a partial match falls through to name matching
        return rigify
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
