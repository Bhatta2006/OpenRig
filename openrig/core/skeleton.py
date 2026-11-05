"""Turn a RigMap into particles (joints) and edges (bones) for the solver."""
from __future__ import annotations

from dataclasses import dataclass, field

from .mapping import RigMap, SIDES


@dataclass
class Edge:
    key: str                  # role key, e.g. "upperarm.L"
    bone: str                 # Blender bone name
    a: int                    # particle index of bone head
    b: int                    # particle index of the joint it points at
    parent: str | None        # role key of the parent edge (mapped ancestor), or None


@dataclass
class Skeleton:
    names: list[str] = field(default_factory=list)            # particle names
    sources: list[tuple[str, str]] = field(default_factory=list)  # (bone, 'head'|'tail')
    edges: list[Edge] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    handles: dict[str, str] = field(default_factory=dict)     # handle label -> particle name

    def add(self, name: str, bone: str, which: str) -> int:
        self.index[name] = len(self.names)
        self.names.append(name)
        self.sources.append((bone, which))
        return self.index[name]

    def has(self, name: str) -> bool:
        return name in self.index

    @property
    def edge_by_key(self) -> dict[str, Edge]:
        return {e.key: e for e in self.edges}


def build_skeleton(m: RigMap) -> Skeleton:
    if not m.ok:
        raise ValueError("Rig mapping incomplete: " + ", ".join(m.missing()))
    sk = Skeleton()
    axial = [("hips", m.hips)] + [(f"spine{i}", b) for i, b in enumerate(m.spine)]
    axial += [(f"neck{i}", b) for i, b in enumerate(m.neck)] + [("head", m.head)]
    for pname, bone in axial:
        sk.add(pname, bone, "head")
    sk.add("head_top", m.head, "tail")

    # axial edges
    prev_key = None
    for i, (pname, bone) in enumerate(axial):
        a = sk.index[pname]
        b = sk.index[axial[i + 1][0]] if i + 1 < len(axial) else sk.index["head_top"]
        sk.edges.append(Edge(pname, bone, a, b, prev_key))
        prev_key = pname
    last_spine = f"spine{len(m.spine) - 1}"

    for s in SIDES:
        L = m.limbs[s]
        torso_parent = last_spine
        if "clavicle" in L:
            sk.add(f"clavicle.{s}", L["clavicle"], "head")
        for r in ("upperarm", "forearm", "hand"):
            sk.add(f"{r}.{s}", L[r], "head")
        sk.add(f"hand_end.{s}", L["hand"], "tail")
        for r in ("thigh", "shin", "foot"):
            sk.add(f"{r}.{s}", L[r], "head")
        if "toe" in L:
            sk.add(f"toe.{s}", L["toe"], "head")
            sk.add(f"toe_end.{s}", L["toe"], "tail")
        else:
            sk.add(f"toe_end.{s}", L["foot"], "tail")

        ix = sk.index
        parent = torso_parent
        if "clavicle" in L:
            sk.edges.append(Edge(f"clavicle.{s}", L["clavicle"], ix[f"clavicle.{s}"], ix[f"upperarm.{s}"], torso_parent))
            parent = f"clavicle.{s}"
        sk.edges.append(Edge(f"upperarm.{s}", L["upperarm"], ix[f"upperarm.{s}"], ix[f"forearm.{s}"], parent))
        sk.edges.append(Edge(f"forearm.{s}", L["forearm"], ix[f"forearm.{s}"], ix[f"hand.{s}"], f"upperarm.{s}"))
        sk.edges.append(Edge(f"hand.{s}", L["hand"], ix[f"hand.{s}"], ix[f"hand_end.{s}"], f"forearm.{s}"))
        sk.edges.append(Edge(f"thigh.{s}", L["thigh"], ix[f"thigh.{s}"], ix[f"shin.{s}"], "hips"))
        sk.edges.append(Edge(f"shin.{s}", L["shin"], ix[f"shin.{s}"], ix[f"foot.{s}"], f"thigh.{s}"))
        if "toe" in L:
            sk.edges.append(Edge(f"foot.{s}", L["foot"], ix[f"foot.{s}"], ix[f"toe.{s}"], f"shin.{s}"))
            sk.edges.append(Edge(f"toe.{s}", L["toe"], ix[f"toe.{s}"], ix[f"toe_end.{s}"], f"foot.{s}"))
        else:
            sk.edges.append(Edge(f"foot.{s}", L["foot"], ix[f"foot.{s}"], ix[f"toe_end.{s}"], f"shin.{s}"))

    sk.handles = {
        "Hips": "hips", "Head": "head_top",
        "Hand L": "hand.L", "Hand R": "hand.R",
        "Foot L": "foot.L", "Foot R": "foot.R",
    }
    return sk
